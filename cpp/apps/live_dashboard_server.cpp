#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <psapi.h>

#include "ps26053/io/lidar_io.hpp"
#include "ps26053/runtime/pipeline.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <climits>
#include <cmath>
#include <filesystem>

namespace {

static std::atomic<bool> g_running{true};

BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT) {
        std::cout << "\n[Server] Shutting down C++ Live Dashboard Server...\n";
        g_running = false;
        return TRUE;
    }
    return FALSE;
}

double getProcessRamMb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
}

std::string readFileContents(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) return "";
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string sanitizeJsonPath(const std::string& p) {
    std::string s = p;
    for (char& c : s) {
        if (c == '\\') c = '/';
    }
    return s;
}

} // namespace

class LiveDashboardServer {
public:
    LiveDashboardServer(int port = 8080, const std::string& scan_arg = "data/raw",
                        bool watch_enabled = false, int watch_interval_sec = 5,
                        bool loop_enabled = false)
        : port_(port), scan_arg_(scan_arg), watch_enabled_(watch_enabled),
          watch_interval_sec_(watch_interval_sec > 0 ? watch_interval_sec : 5),
          loop_enabled_(loop_enabled),
          model_path_("models/onnx/pointnet2_semseg.onnx") {
        pipeline_ = std::make_unique<ps26053::MappingPipeline>(model_path_);
    }

    bool start() {
        // 1. Initialize Winsock
        WSADATA wsaData;
        int iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (iResult != 0) {
            std::cerr << "[Error] WSAStartup failed: " << iResult << "\n";
            return false;
        }

        // 2. Pre-initialize native C++ perception pipeline
        std::cout << "[Pipeline] Initializing PointNet++ ONNX runtime in C++...\n";
        if (!pipeline_->initialize()) {
            std::cerr << "[Warn] Could not load ONNX model; running in heuristic perception mode.\n";
        }
        pipeline_->loadConfig("config");

        // Preload LiDAR scan(s): single file or sequence directory. Frame
        // timestamps assume the nominal 10 Hz from config/lidar.yaml.
        seq_scans_ = ps26053::LidarIO::listSequenceScans(scan_arg_);
        if (seq_scans_.empty()) seq_scans_.push_back(scan_arg_);
        for (const auto& p : seq_scans_) known_scans_.insert(p);
        std::cout << "[Dataset] " << seq_scans_.size() << " LiDAR scan(s) from: " << scan_arg_ << " ...\n";
        seq_index_ = 0;
        if (processFrameAt(0)) {
            scan_processed_ = true;
        }

        // 3. Create server socket
        listen_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_socket_ == INVALID_SOCKET) {
            std::cerr << "[Error] Socket creation failed: " << WSAGetLastError() << "\n";
            WSACleanup();
            return false;
        }

        // Set SO_REUSEADDR
        int opt = 1;
        setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        // Bind
        sockaddr_in server_addr;
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(static_cast<u_short>(port_));

        if (bind(listen_socket_, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
            std::cerr << "[Error] Bind failed on port " << port_ << ": " << WSAGetLastError() << "\n";
            closesocket(listen_socket_);
            WSACleanup();
            return false;
        }

        if (listen(listen_socket_, SOMAXCONN) == SOCKET_ERROR) {
            std::cerr << "[Error] Listen failed: " << WSAGetLastError() << "\n";
            closesocket(listen_socket_);
            WSACleanup();
            return false;
        }

        std::cout << "==============================================================\n";
        std::cout << "  PS 26053: C++ NATIVE LIDAR DASHBOARD SERVER ONLINE           \n";
        std::cout << "  URL: http://localhost:" << port_ << "/                       \n";
        std::cout << "  Input: Pure LiDAR Point Cloud Sequences (.bin/.pcd)        \n";
        std::cout << "  Engine: Native C++17/20 | RAM: " << std::fixed << std::setprecision(1)
                  << getProcessRamMb() << " MB (C++-only runtime)\n";
        std::cout << "  Scans: " << seq_scans_.size() << " frame(s) loaded from " << scan_arg_ << "\n";
        std::cout << "==============================================================\n";

        if (watch_enabled_ || loop_enabled_) {
            std::cout << "[Watch] Auto-ingest " << (watch_enabled_ ? "ON" : "off")
                      << " (every " << watch_interval_sec_ << " s), loop "
                      << (loop_enabled_ ? "ON (replays reported)" : "off") << ".\n";
            startWatchThread();
        }

        // Listen loop
        while (g_running) {
            fd_set read_fds;
            FD_ZERO(&read_fds);
            FD_SET(listen_socket_, &read_fds);

            timeval timeout{};
            timeout.tv_sec = 0;
            timeout.tv_usec = 200000; // 200ms timeout for non-blocking shutdown check

            int activity = select(0, &read_fds, nullptr, nullptr, &timeout);
            if (activity > 0 && FD_ISSET(listen_socket_, &read_fds)) {
                sockaddr_in client_addr;
                int client_len = sizeof(client_addr);
                SOCKET client_socket = accept(listen_socket_, (sockaddr*)&client_addr, &client_len);
                if (client_socket != INVALID_SOCKET) {
                    char ip_buf[INET_ADDRSTRLEN] = {0};
                    inet_ntop(AF_INET, &(client_addr.sin_addr), ip_buf, INET_ADDRSTRLEN);
                    std::string client_ip(ip_buf);
                    std::thread(&LiveDashboardServer::handleClient, this, client_socket, client_ip).detach();
                }
            }
        }

        closesocket(listen_socket_);
        WSACleanup();
        return true;
    }

private:
    int port_{8080};
    SOCKET listen_socket_{INVALID_SOCKET};
    std::unique_ptr<ps26053::MappingPipeline> pipeline_;
    ps26053::PointCloud raw_scan_;
    ps26053::FrameMetrics latest_metrics_;
    bool scan_processed_{false};
    std::mutex mtx_;

    // LiDAR sequence state: ordered scan paths plus the current frame index.
    std::string scan_arg_;
    std::string model_path_;  // ONNX model path — stored so reset() can reconstruct
    std::vector<std::string> seq_scans_;
    size_t seq_index_{0};
    static constexpr double kFrameIntervalSec = 0.1; // nominal 10 Hz (lidar.yaml)
    bool watch_enabled_{false};
    int watch_interval_sec_{5};
    bool loop_enabled_{false};
    size_t loop_wraps_{0};
    double loop_time_offset_{0.0};
    std::set<std::string> known_scans_;

    void handleClient(SOCKET sock, const std::string& client_ip) {
        std::vector<char> buf(16384);
        std::string request;
        size_t content_length = 0;
        size_t header_end = std::string::npos;
        bool headers_parsed = false;

        // Loop to receive complete HTTP headers and body
        while (true) {
            int n = recv(sock, buf.data(), static_cast<int>(buf.size()), 0);
            if (n <= 0) break;
            request.append(buf.data(), n);

            if (!headers_parsed) {
                header_end = request.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    headers_parsed = true;
                    std::string headers = request.substr(0, header_end);
                    std::string cl_key = "Content-Length: ";
                    size_t pos = headers.find(cl_key);
                    if (pos == std::string::npos) {
                        cl_key = "content-length: ";
                        pos = headers.find(cl_key);
                    }
                    if (pos != std::string::npos) {
                        size_t start = pos + cl_key.length();
                        size_t end = headers.find("\r\n", start);
                        try {
                            content_length = std::stoul(headers.substr(start, end - start));
                        } catch (...) {
                            content_length = 0;
                        }
                    }
                }
            }

            if (headers_parsed) {
                size_t body_len = request.size() - (header_end + 4);
                if (body_len >= content_length) {
                    break;
                }
            }
        }

        if (request.empty()) {
            closesocket(sock);
            return;
        }

        std::istringstream req_stream(request);
        std::string method, path, proto;
        req_stream >> method >> path >> proto;

        // Parse query params if any
        std::string pure_path = path;
        std::string query_str;
        size_t qmark = path.find('?');
        if (qmark != std::string::npos) {
            pure_path = path.substr(0, qmark);
            query_str = path.substr(qmark + 1);
        }

        // Route requests (Strictly LiDAR-Only)
        if (method == "GET" && (pure_path == "/" || pure_path == "/index.html")) {
            serveIndexHtml(sock);
        } else if (method == "GET" && pure_path == "/api/status") {
            serveApiStatus(sock);
        } else if (method == "GET" && pure_path == "/api/lidar/scan") {
            serveApiLidarScan(sock);
        } else if (method == "GET" && pure_path == "/api/frames") {
            serveApiFrames(sock);
        } else if (method == "POST" && pure_path == "/api/frame/next") {
            serveApiFrameNext(sock);
        } else if (method == "POST" && pure_path == "/api/frame/prev") {
            serveApiFramePrev(sock);
        } else if (method == "POST" && pure_path == "/api/frame/select") {
            serveApiFrameSelect(sock, query_str, request);
        } else if (method == "POST" && pure_path == "/api/lidar/upload") {
            serveApiLidarUpload(sock, request, header_end, query_str);
        } else if (method == "POST" && pure_path == "/api/reset") {
            serveApiReset(sock);
        } else if (method == "GET" && (pure_path == "/api/ply" || pure_path == "/api/ply/export")) {
            serveApiPly(sock);
        } else if (method == "GET" && pure_path.rfind("/vendor/", 0) == 0) {
            serveVendorFile(sock, pure_path);
        } else if (method == "OPTIONS") {
            serveCorsPreflight(sock);
        } else {
            sendResponse(sock, 404, "text/plain", "404 Not Found");
        }

        closesocket(sock);
    }

    void serveIndexHtml(SOCKET sock) {
        std::string html = readFileContents("cpp/web/index.html");
        if (html.empty()) {
            html = "<html><body><h2>PS 26053 C++ Server Running</h2><p>Please place index.html in cpp/web/index.html.</p></body></html>";
        }
        sendResponse(sock, 200, "text/html; charset=utf-8", html);
    }

    void serveVendorFile(SOCKET sock, const std::string& pure_path) {
        std::string name = pure_path.substr(std::string("/vendor/").size());
        if (name.empty() || name.find("..") != std::string::npos ||
            name.find('/') != std::string::npos || name.find('\\') != std::string::npos) {
            sendResponse(sock, 404, "text/plain", "404 Not Found");
            return;
        }
        std::string content_type = "application/octet-stream";
        if (name.size() >= 3 && name.compare(name.size() - 3, 3, ".js") == 0) {
            content_type = "application/javascript";
        } else if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".css") == 0) {
            content_type = "text/css";
        }
        std::string body = readFileContents("cpp/web/vendor/" + name);
        if (body.empty()) {
            sendResponse(sock, 404, "text/plain", "404 Not Found");
            return;
        }
        sendResponse(sock, 200, content_type, body);
    }

    void serveApiStatus(SOCKET sock) {
        double ram_mb = getProcessRamMb();

        size_t active_pts = 0;
        size_t active_cells = 0;
        size_t active_tracks = 0;
        double pre_ms = 0.0, inf_ms = 0.0, trk_ms = 0.0, map_ms = 0.0, tot_ms = 0.0, fps_val = 0.0;
        size_t net_pts = 0, fall_pts = 0;
        size_t b_errors = 0;

        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (scan_processed_ && !raw_scan_.empty()) {
                active_pts = raw_scan_.size();
                active_cells = latest_metrics_.active_cells;
                active_tracks = latest_metrics_.active_tracks;
                pre_ms = latest_metrics_.preprocess_time_ms;
                inf_ms = latest_metrics_.inference_time_ms;
                trk_ms = latest_metrics_.tracking_time_ms;
                map_ms = latest_metrics_.mapping_time_ms;
                tot_ms = latest_metrics_.total_time_ms;
                fps_val = latest_metrics_.fps;
                net_pts = latest_metrics_.inference_network_points;
                fall_pts = latest_metrics_.inference_fallback_points;
                b_errors = pipeline_->getGrid().checkBoundaryAlignment().totalErrors();
            }
        }

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"engine\": \"Native C++17/20 LiDAR Perception Engine\",\n";
        ss << "  \"ram_mb\": " << ram_mb << ",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"current_scan\": \"" << (seq_scans_.empty() ? "" : sanitizeJsonPath(seq_scans_[seq_index_])) << "\",\n";
        ss << "  \"lidar_points\": " << active_pts << ",\n";
        ss << "  \"active_cells\": " << active_cells << ",\n";
        ss << "  \"active_tracks\": " << active_tracks << ",\n";
        ss << "  \"network_points\": " << net_pts << ",\n";
        ss << "  \"fallback_points\": " << fall_pts << ",\n";
        ss << "  \"preprocess_time_ms\": " << pre_ms << ",\n";
        ss << "  \"inference_time_ms\": " << inf_ms << ",\n";
        ss << "  \"tracking_time_ms\": " << trk_ms << ",\n";
        ss << "  \"mapping_time_ms\": " << map_ms << ",\n";
        ss << "  \"total_latency_ms\": " << tot_ms << ",\n";
        ss << "  \"fps\": " << fps_val << ",\n";
        ss << "  \"boundary_errors\": " << b_errors << "\n";
        ss << "}";

        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiLidarScan(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!scan_processed_ && !raw_scan_.empty()) {
            latest_metrics_ = pipeline_->processFrame(raw_scan_, 0.0, 0);
            scan_processed_ = true;
        }

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"metrics\": {\n";
        ss << "    \"input_points\": " << raw_scan_.size() << ",\n";
        ss << "    \"frame_index\": " << seq_index_ << ",\n";
        ss << "    \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "    \"current_scan\": \"" << (seq_scans_.empty() ? "" : sanitizeJsonPath(seq_scans_[seq_index_])) << "\",\n";
        ss << "    \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "    \"active_tracks\": " << latest_metrics_.active_tracks << ",\n";
        ss << "    \"preprocess_time_ms\": " << latest_metrics_.preprocess_time_ms << ",\n";
        ss << "    \"inference_time_ms\": " << latest_metrics_.inference_time_ms << ",\n";
        ss << "    \"tracking_time_ms\": " << latest_metrics_.tracking_time_ms << ",\n";
        ss << "    \"mapping_time_ms\": " << latest_metrics_.mapping_time_ms << ",\n";
        ss << "    \"total_time_ms\": " << latest_metrics_.total_time_ms << ",\n";
        ss << "    \"fps\": " << latest_metrics_.fps << ",\n";
        ss << "    \"network_points\": " << latest_metrics_.inference_network_points << ",\n";
        ss << "    \"fallback_points\": " << latest_metrics_.inference_fallback_points << ",\n";
        ss << "    \"boundary_errors\": " << pipeline_->getGrid().checkBoundaryAlignment().totalErrors() << ",\n";
        ss << "    \"ram_mb\": " << getProcessRamMb() << "\n";
        ss << "  },\n";

        // 1. Point cloud with authentic semantic classification from PointNet++
        const auto& cloud_to_send = pipeline_->getProcessedCloud().empty() ? raw_scan_ : pipeline_->getProcessedCloud();
        ss << "  \"points\": [\n";
        bool first_pt = true;

        // Dynamic obstacles (vehicles) priority emission
        for (size_t i = 0; i < cloud_to_send.size(); ++i) {
            const auto& pt = cloud_to_send[i];
            if (pt.semantic_class == ps26053::SemanticClass::DYNAMIC_OBSTACLE) {
                if (!first_pt) ss << ",\n";
                first_pt = false;
                ss << "    [" << pt.x << "," << pt.y << "," << pt.z << "," << pt.intensity << ",2," << pt.confidence << "]";
            }
        }
        // Subsample terrain and static background points for efficient JSON transport
        size_t step = std::max<size_t>(1, cloud_to_send.size() / 15000);
        for (size_t i = 0; i < cloud_to_send.size(); i += step) {
            const auto& pt = cloud_to_send[i];
            if (pt.semantic_class != ps26053::SemanticClass::DYNAMIC_OBSTACLE) {
                if (!first_pt) ss << ",\n";
                first_pt = false;
                int sem_cls = static_cast<int>(pt.semantic_class);
                ss << "    [" << pt.x << "," << pt.y << "," << pt.z << "," << pt.intensity << "," << sem_cls << "," << pt.confidence << "]";
            }
        }
        ss << "\n  ],\n";

        // 2. Active 2.5D cells from Quadtree
        ss << "  \"cells\": [\n";
        auto cells = pipeline_->getGrid().getAllCells();
        bool first_cell = true;
        size_t cell_stride = std::max<size_t>(1, cells.size() / 5000);
        for (size_t i = 0; i < cells.size(); i += cell_stride) {
            const auto& c = cells[i];
            if (c.point_count == 0 && c.occupancy < 0.1f) continue;
            if (!first_cell) ss << ",\n";
            first_cell = false;
            float cx = (c.bounds.min_x + c.bounds.max_x) * 0.5f;
            float cy = (c.bounds.min_y + c.bounds.max_y) * 0.5f;
            ss << "    {\"x\":" << cx << ",\"y\":" << cy << ",\"res\":" << c.resolution
               << ",\"elev\":" << c.elevation << ",\"occ\":" << c.occupancy
               << ",\"class\":" << static_cast<int>(c.semantic_class)
               << ",\"conf\":" << c.semantic_confidence
               << ",\"object_id\":" << c.object_id
               << ",\"vx\":" << c.velocity_x << ",\"vy\":" << c.velocity_y << "}";
        }
        ss << "\n  ],\n";

        // 3. Tracked obstacles from Kalman Filter across consecutive frames
        ss << "  \"tracks\": [\n";
        const auto& tracks = pipeline_->getTracks();
        bool first_trk = true;
        for (const auto& tr : tracks) {
            if (!first_trk) ss << ",\n";
            first_trk = false;
            std::string cls_str = (tr.semantic_class == ps26053::SemanticClass::DYNAMIC_OBSTACLE) ? "Vehicle" : "Obstacle";
            ss << "    {\"id\":" << tr.id << ",\"class\":\"" << cls_str << "\",\"x\":" << tr.position.x()
               << ",\"y\":" << tr.position.y() << ",\"z\":" << tr.position_z
               << ",\"vx\":" << tr.velocity.x() << ",\"vy\":" << tr.velocity.y()
               << ",\"confidence\":" << tr.confidence
               << ",\"length\":" << tr.bbox.width() << ",\"width\":" << tr.bbox.height() << "}";
        }
        ss << "\n  ]\n";
        ss << "}";

        sendResponse(sock, 200, "application/json", ss.str());
    }

    // Load and process one sequence frame (grid/tracker state persists).
    bool processFrameAt(size_t index) {
        if (index >= seq_scans_.size()) return false;
        if (!ps26053::LidarIO::loadPointCloud(seq_scans_[index], raw_scan_)) {
            std::cerr << "[Server] Failed to load scan: " << seq_scans_[index] << "\n";
            return false;
        }
        seq_index_ = index;
        latest_metrics_ = pipeline_->processFrame(
            raw_scan_, loop_time_offset_ + index * kFrameIntervalSec, static_cast<int>(index));
        scan_processed_ = true;
        return true;
    }

    void startWatchThread() {
        std::error_code ec;
        if (!std::filesystem::is_directory(scan_arg_, ec)) {
            if (watch_enabled_) {
                std::cerr << "[Watch] Not a directory, watch disabled: " << scan_arg_ << "\n";
                watch_enabled_ = false;
            }
            if (loop_enabled_ && seq_scans_.size() <= 1) {
                std::cerr << "[Watch] Loop needs a sequence directory; disabled.\n";
                loop_enabled_ = false;
            }
            return;
        }
        std::thread([this]() {
            while (g_running) {
                std::this_thread::sleep_for(std::chrono::seconds(watch_interval_sec_));
                std::lock_guard<std::mutex> lock(mtx_);
                if (!g_running) break;
                auto current = ps26053::LidarIO::listSequenceScans(scan_arg_);
                bool advanced = false;
                for (const auto& p : current) {
                    if (known_scans_.count(p)) continue;
                    seq_scans_.push_back(p);
                    known_scans_.insert(p);
                    if (processFrameAt(seq_scans_.size() - 1)) {
                        std::cout << "[Watch] Auto-ingested frame " << (seq_scans_.size() - 1)
                                  << ": " << p << "\n";
                        advanced = true;
                    }
                }
                if (!advanced && loop_enabled_ && !seq_scans_.empty() &&
                    seq_index_ + 1 >= seq_scans_.size()) {
                    loop_time_offset_ += seq_scans_.size() * kFrameIntervalSec;
                    ++loop_wraps_;
                    if (processFrameAt(0)) {
                        std::cout << "[Watch] Looping sequence (replay #"
                                  << loop_wraps_ << ", NOT live data).\n";
                    }
                }
            }
        }).detach();
    }

    void serveApiFrames(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"current_scan\": \"" << (seq_scans_.empty() ? "" : sanitizeJsonPath(seq_scans_[seq_index_])) << "\",\n";
        ss << "  \"scans\": [\n";
        for (size_t i = 0; i < seq_scans_.size(); ++i) {
            if (i > 0) ss << ",\n";
            ss << "    \"" << sanitizeJsonPath(seq_scans_[i]) << "\"";
        }
        ss << "\n  ],\n";
        ss << "  \"watch_enabled\": " << (watch_enabled_ ? "true" : "false") << ",\n";
        ss << "  \"watch_dir\": \"" << (watch_enabled_ ? scan_arg_ : "") << "\",\n";
        ss << "  \"loop_enabled\": " << (loop_enabled_ ? "true" : "false") << ",\n";
        ss << "  \"replay\": " << ((loop_enabled_ && loop_wraps_ > 0) ? "true" : "false") << ",\n";
        ss << "  \"wraps\": " << loop_wraps_ << "\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiFrameNext(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (seq_index_ + 1 >= seq_scans_.size()) {
            if (loop_enabled_ && !seq_scans_.empty()) {
                loop_time_offset_ += seq_scans_.size() * kFrameIntervalSec;
                ++loop_wraps_;
                processFrameAt(0);
            } else {
                ss << "{\n";
                ss << "  \"status\": \"end_of_sequence\",\n";
                ss << "  \"frame_index\": " << seq_index_ << ",\n";
                ss << "  \"frame_count\": " << seq_scans_.size() << "\n";
                ss << "}";
                sendResponse(sock, 200, "application/json", ss.str());
                return;
            }
        } else {
            processFrameAt(seq_index_ + 1);
        }

        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"current_scan\": \"" << sanitizeJsonPath(seq_scans_[seq_index_]) << "\",\n";
        ss << "  \"metrics\": {\n";
        ss << "    \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "    \"active_tracks\": " << latest_metrics_.active_tracks << ",\n";
        ss << "    \"total_time_ms\": " << latest_metrics_.total_time_ms << "\n";
        ss << "  }\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiFramePrev(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (seq_index_ > 0) {
            processFrameAt(seq_index_ - 1);
        }

        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"current_scan\": \"" << sanitizeJsonPath(seq_scans_[seq_index_]) << "\",\n";
        ss << "  \"metrics\": {\n";
        ss << "    \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "    \"active_tracks\": " << latest_metrics_.active_tracks << ",\n";
        ss << "    \"total_time_ms\": " << latest_metrics_.total_time_ms << "\n";
        ss << "  }\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiFrameSelect(SOCKET sock, const std::string& query, const std::string& body) {
        size_t target_idx = 0;
        bool found_idx = false;

        // Try parsing query param ?index=N
        size_t idx_pos = query.find("index=");
        if (idx_pos != std::string::npos) {
            try {
                target_idx = std::stoul(query.substr(idx_pos + 6));
                found_idx = true;
            } catch (...) {}
        }
        if (!found_idx) {
            size_t b_idx = body.find("\"index\":");
            if (b_idx != std::string::npos) {
                try {
                    target_idx = std::stoul(body.substr(b_idx + 8));
                    found_idx = true;
                } catch (...) {}
            }
        }

        std::lock_guard<std::mutex> lock(mtx_);
        if (seq_scans_.empty() || target_idx >= seq_scans_.size()) {
            sendResponse(sock, 400, "application/json", "{\"error\":\"Invalid frame index\"}");
            return;
        }

        processFrameAt(target_idx);

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"current_scan\": \"" << sanitizeJsonPath(seq_scans_[seq_index_]) << "\",\n";
        ss << "  \"metrics\": {\n";
        ss << "    \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "    \"active_tracks\": " << latest_metrics_.active_tracks << ",\n";
        ss << "    \"total_time_ms\": " << latest_metrics_.total_time_ms << "\n";
        ss << "  }\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    // ── LiDAR Scan Upload ──────────────────────────────────────────────────
    // POST /api/lidar/upload
    // Body: raw bytes of a single Velodyne .bin scan (4×float32 per point).
    // Optional query: ?filename=<name.bin>  (used for display only)
    //
    // The uploaded scan is:
    //   1. Saved to uploads/<timestamp>_<filename>.bin
    //   2. Appended to the live sequence (seq_scans_)
    //   3. Processed immediately as the next frame
    //   4. Returned as a normal /api/lidar/scan JSON response
    void serveApiLidarUpload(SOCKET sock, const std::string& request,
                             size_t header_end, const std::string& query_str) {
        // Extract filename hint from query string
        std::string filename_hint = "uploaded.bin";
        auto parse_param = [&](const std::string& key) -> std::string {
            std::string prefix = key + "=";
            size_t p = query_str.find(prefix);
            if (p == std::string::npos) return "";
            p += prefix.size();
            size_t e = query_str.find('&', p);
            return (e == std::string::npos) ? query_str.substr(p) : query_str.substr(p, e - p);
        };
        std::string fn = parse_param("filename");
        if (!fn.empty()) filename_hint = fn;

        // Validate extension — only .bin, .pcd, and .ply allowed
        auto ext_ok = [](const std::string& s) {
            std::string lower = s;
            for (char& c : lower) c = static_cast<char>(std::tolower(c));
            if (lower.size() >= 4 && lower.substr(lower.size()-4) == ".bin") return true;
            if (lower.size() >= 4 && lower.substr(lower.size()-4) == ".pcd") return true;
            if (lower.size() >= 4 && lower.substr(lower.size()-4) == ".ply") return true;
            return false;
        };
        if (!ext_ok(filename_hint)) {
            sendResponse(sock, 400, "application/json",
                "{\"error\":\"Only LiDAR point cloud files (.bin, .pcd, .ply) are accepted. "
                "Camera images, RGB video, and non-LiDAR formats are strictly rejected.\"}");
            return;
        }

        // Extract body bytes (raw scan data)
        size_t body_start = header_end + 4;
        std::string body = (body_start < request.size()) ? request.substr(body_start) : "";

        if (body.empty()) {
            sendResponse(sock, 400, "application/json",
                "{\"error\":\"Empty request body — send raw LiDAR point cloud bytes.\"}" );
            return;
        }

        // Minimum size check: must have at least 1 point
        if (body.size() < 16) {
            sendResponse(sock, 400, "application/json",
                "{\"error\":\"File too small — must be a valid LiDAR point cloud (>=16 bytes).\"}" );
            return;
        }

        // Save to uploads/ with timestamp prefix
        std::filesystem::create_directories("uploads");
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::string save_path = "uploads/" + std::to_string(now_ms) + "_" + filename_hint;
        {
            std::ofstream out(save_path, std::ios::binary);
            if (!out.is_open()) {
                sendResponse(sock, 500, "application/json",
                    "{\"error\":\"Failed to save uploaded file to disk.\"}" );
                return;
            }
            out.write(body.data(), static_cast<std::streamsize>(body.size()));
        }

        std::lock_guard<std::mutex> lock(mtx_);

        // Process the uploaded scan immediately through the pipeline
        ps26053::PointCloud raw_scan;
        if (!ps26053::LidarIO::loadPointCloud(save_path, raw_scan) || raw_scan.empty()) {
            sendResponse(sock, 422, "application/json",
                "{\"error\":\"File saved but could not be parsed as a valid LiDAR point cloud (.bin, .pcd, .ply).\"}" );
            return;
        }

        // Append to the live sequence
        seq_scans_.push_back(save_path);
        size_t new_frame_idx = seq_scans_.size() - 1;
        seq_index_ = new_frame_idx;
        raw_scan_ = raw_scan;

        // Run full pipeline
        double ts = new_frame_idx * 0.1;
        latest_metrics_ = pipeline_->processFrame(raw_scan_, ts, static_cast<int>(new_frame_idx));
        scan_processed_ = true;

        // Export PLY for this frame
        char fname_buf[64];
        std::snprintf(fname_buf, sizeof(fname_buf),
            "results/maps/adaptive_map_frame%06zu.ply", new_frame_idx);
        std::filesystem::create_directories("results/maps");
        pipeline_->getGrid().exportToPLY(fname_buf);

        // Return scan JSON (same format as /api/lidar/scan)
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"upload\": {\n";
        ss << "    \"filename\": \"" << filename_hint << "\",\n";
        ss << "    \"bytes\": " << body.size() << ",\n";
        ss << "    \"points_loaded\": " << raw_scan.size() << ",\n";
        ss << "    \"frame_index\": " << new_frame_idx << ",\n";
        ss << "    \"total_frames\": " << seq_scans_.size() << "\n";
        ss << "  },\n";
        ss << "  \"metrics\": {\n";
        ss << "    \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "    \"active_tracks\": " << latest_metrics_.active_tracks << ",\n";
        ss << "    \"total_time_ms\": " << latest_metrics_.total_time_ms << "\n";
        ss << "  }\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiReset(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        // Reinitialize pipeline state — MappingPipeline has no reset() method;
        // reconstruct it from the same model path and re-run initialize().
        pipeline_ = std::make_unique<ps26053::MappingPipeline>(model_path_);
        pipeline_->initialize();
        pipeline_->loadConfig("config");
        seq_index_ = 0;
        loop_time_offset_ = 0.0;
        loop_wraps_ = 0;
        latest_metrics_ = ps26053::FrameMetrics{};
        if (!seq_scans_.empty()) {
            processFrameAt(0);
        }
        sendResponse(sock, 200, "application/json", "{\"status\":\"ok\",\"message\":\"Pipeline reset to frame 0\"}");
    }

    void serveApiPly(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        // Build zero-padded filename using snprintf to avoid char* concat errors.
        char fname_buf[64];
        std::snprintf(fname_buf, sizeof(fname_buf),
            "results/maps/adaptive_map_frame%06zu.ply", seq_index_);
        std::string tmp_ply(fname_buf);
        pipeline_->getGrid().exportToPLY(tmp_ply);
        std::string ply_data = readFileContents(tmp_ply);
        if (ply_data.empty()) {
            sendResponse(sock, 500, "text/plain", "Failed to export PLY");
            return;
        }
        sendResponse(sock, 200, "application/x-ply", ply_data);
    }

    void serveCorsPreflight(SOCKET sock) {
        std::ostringstream ss;
        ss << "HTTP/1.1 204 No Content\r\n";
        ss << "Access-Control-Allow-Origin: *\r\n";
        ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        ss << "Access-Control-Allow-Headers: Content-Type\r\n";
        ss << "Content-Length: 0\r\n\r\n";
        std::string res = ss.str();
        send(sock, res.c_str(), static_cast<int>(res.size()), 0);
    }

    void sendResponse(SOCKET sock, int status_code, const std::string& content_type, const std::string& body) {
        std::ostringstream ss;
        std::string status_msg = (status_code == 200) ? "OK" : (status_code == 404 ? "Not Found" : "Error");
        ss << "HTTP/1.1 " << status_code << " " << status_msg << "\r\n";
        ss << "Content-Type: " << content_type << "\r\n";
        ss << "Access-Control-Allow-Origin: *\r\n";
        ss << "Connection: close\r\n";
        ss << "Cache-Control: no-store\r\n";
        ss << "Content-Length: " << body.size() << "\r\n\r\n";
        ss << body;

        std::string resp = ss.str();
        size_t sent_total = 0;
        while (sent_total < resp.size()) {
            int chunk = static_cast<int>(std::min<size_t>(resp.size() - sent_total, INT_MAX));
            int n = send(sock, resp.c_str() + sent_total, chunk, 0);
            if (n <= 0) {
                std::cerr << "[Server] send() failed after " << sent_total << " of "
                          << resp.size() << " bytes (WSA " << WSAGetLastError() << ").\n";
                return;
            }
            sent_total += static_cast<size_t>(n);
        }
    }
};

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    int port = 8080;
    std::string scan_arg = "data/raw/sequences/00/velodyne";
    if (!std::filesystem::exists(scan_arg)) {
        scan_arg = "data/raw";
    }
    bool watch = false;
    int watch_interval = 5;
    bool loop = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--watch") {
            watch = true;
        } else if (arg == "--loop") {
            loop = true;
        } else if (arg.rfind("--interval=", 0) == 0) {
            watch_interval = std::stoi(arg.substr(11));
        } else if (arg.rfind("--port=", 0) == 0) {
            port = std::stoi(arg.substr(7));
        } else if (!arg.empty() && arg[0] != '-') {
            if (i == 1) {
                // Positional port or scan_arg
                if (std::all_of(arg.begin(), arg.end(), ::isdigit)) {
                    port = std::stoi(arg);
                } else {
                    scan_arg = arg;
                }
            } else if (i == 2) {
                scan_arg = arg;
            }
        }
    }

    LiveDashboardServer server(port, scan_arg, watch, watch_interval, loop);
    if (!server.start()) {
        return 1;
    }

    return 0;
}
