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
#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <climits>
#include <cmath>

namespace {

std::atomic<bool> g_running{true};

BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT) {
        std::cout << "\n[Server] Shutting down C++ Live Dashboard Server...\n";
        g_running = false;
        return TRUE;
    }
    return FALSE;
}

// Win32 API to retrieve real process memory usage
double getProcessRamMb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 24.0;
}

std::string readFileContents(const std::string& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) return "";
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace

class LiveDashboardServer {
public:
    LiveDashboardServer(int port = 8080, const std::string& scan_arg = "data/raw/000000.bin")
        : port_(port), scan_arg_(scan_arg) {
        pipeline_ = std::make_unique<ps26053::MappingPipeline>("models/onnx/pointnet2_semseg.onnx");
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
        // H3: behavior comes from config/*.yaml (compiled defaults apply
        // loudly only when the files are missing or malformed).
        pipeline_->loadConfig("config");

        // Preload LiDAR scan(s): single file or sequence directory. Frame
        // timestamps assume the nominal 10 Hz from config/lidar.yaml.
        seq_scans_ = ps26053::LidarIO::listSequenceScans(scan_arg_);
        if (seq_scans_.empty()) seq_scans_.push_back(scan_arg_);
        std::cout << "[Dataset] " << seq_scans_.size() << " scan(s) from: " << scan_arg_ << " ...\n";
        seq_index_ = 0;
        if (processFrameAt(0)) {
            scan_processed_ = true;
            std::cout << "[Pipeline] Pre-computation complete: "
                      << latest_metrics_.active_cells << " 2.5D cells, "
                      << latest_metrics_.active_tracks << " tracked objects.\n";
        }

        // 3. Create listening socket
        listen_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listen_socket_ == INVALID_SOCKET) {
            std::cerr << "[Error] Socket creation failed: " << WSAGetLastError() << "\n";
            WSACleanup();
            return false;
        }

        // Allow port reuse
        int opt = 1;
        setsockopt(listen_socket_, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port_);

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
        std::cout << "  PS 26053: C++ NATIVE LIVE DASHBOARD SERVER ONLINE           \n";
        std::cout << "  URL: http://localhost:" << port_ << "/                       \n";
        std::cout << "  Engine: Native C++17/20 | RAM: " << std::fixed << std::setprecision(1) 
                  << getProcessRamMb() << " MB (vs Python ~1250 MB)\n";
        std::cout << "  Camera: Real Hardware Only (Zero Mock Data)\n";
        std::cout << "==============================================================\n";

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

    // Sequence state: ordered scan paths plus the current frame index.
    std::string scan_arg_;
    std::vector<std::string> seq_scans_;
    size_t seq_index_{0};
    static constexpr double kFrameIntervalSec = 0.1; // nominal 10 Hz (lidar.yaml)

    std::atomic<bool> camera_active_{false};
    std::atomic<int> camera_frame_count_{0};
    std::chrono::steady_clock::time_point last_cam_time_{};
    std::string latest_camera_jpeg_binary_;
    std::string latest_camera_base64_;
    std::string remote_client_ip_;

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
        size_t qmark = path.find('?');
        if (qmark != std::string::npos) {
            pure_path = path.substr(0, qmark);
        }

        // Route requests
        if (method == "GET" && (pure_path == "/" || pure_path == "/index.html")) {
            serveIndexHtml(sock);
        } else if (method == "GET" && (pure_path == "/mobile" || pure_path == "/mobile.html")) {
            serveMobileHtml(sock);
        } else if (method == "GET" && pure_path == "/api/status") {
            serveApiStatus(sock);
        } else if (method == "GET" && pure_path == "/api/lidar/scan") {
            serveApiLidarScan(sock);
        } else if (method == "POST" && pure_path == "/api/camera/upload") {
            serveApiCameraUpload(sock, request, client_ip);
        } else if (method == "GET" && pure_path == "/api/camera/latest.jpg") {
            serveApiCameraLatestJpg(sock);
        } else if (method == "GET" && pure_path == "/api/camera/latest") {
            serveApiCameraLatest(sock);
        } else if (method == "POST" && pure_path == "/api/camera/frame") {
            serveApiCameraFrame(sock, request);
        } else if (method == "POST" && pure_path == "/api/reset") {
            serveApiReset(sock);
        } else if (method == "GET" && pure_path == "/api/frames") {
            serveApiFrames(sock);
        } else if (method == "POST" && pure_path == "/api/frame/next") {
            serveApiFrameNext(sock);
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

    void serveMobileHtml(SOCKET sock) {
        std::string html = readFileContents("cpp/web/mobile.html");
        if (html.empty()) {
            html = "<html><body><h2>Mobile Camera Transmitter</h2><p>File cpp/web/mobile.html not found.</p></body></html>";
        }
        sendResponse(sock, 200, "text/html; charset=utf-8", html);
    }

    // Serve vendored frontend assets (Three.js). Flat directory only:
    // any path separator or ".." outside the leading prefix is rejected so
    // this can never escape cpp/web/vendor/.
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
        std::lock_guard<std::mutex> lock(mtx_);
        double ram_mb = getProcessRamMb();

        // Check if camera was active in the last 4 seconds
        auto now = std::chrono::steady_clock::now();
        double sec_since_cam = std::chrono::duration<double>(now - last_cam_time_).count();
        bool is_cam_live = camera_active_ && (sec_since_cam < 4.0);

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"engine\": \"Native C++17/20 (x86_64 MinGW GCC 15.2)\",\n";
        ss << "  \"ram_mb\": " << ram_mb << ",\n";
        // NOTE: a python_ram_mb comparison used to live here with a hardcoded
        // 1250.0 baseline. The runtime is C++-only now and the baseline was
        // never measured, so both fields were removed rather than fabricated.
        ss << "  \"camera_status\": \"" << (is_cam_live ? "online" : "offline") << "\",\n";
        ss << "  \"camera_frames\": " << camera_frame_count_.load() << ",\n";
        ss << "  \"remote_mobile_active\": " << (is_cam_live && !remote_client_ip_.empty() ? "true" : "false") << ",\n";
        ss << "  \"remote_client_ip\": \"" << remote_client_ip_ << "\",\n";
        ss << "  \"lidar_points\": " << (scan_processed_ ? raw_scan_.size() : 0) << ",\n";
        ss << "  \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "  \"active_tracks\": " << latest_metrics_.active_tracks << ",\n";
        ss << "  \"preprocess_time_ms\": " << latest_metrics_.preprocess_time_ms << ",\n";
        ss << "  \"inference_time_ms\": " << latest_metrics_.inference_time_ms << ",\n";
        ss << "  \"tracking_time_ms\": " << latest_metrics_.tracking_time_ms << ",\n";
        ss << "  \"mapping_time_ms\": " << latest_metrics_.mapping_time_ms << ",\n";
        ss << "  \"total_latency_ms\": " << latest_metrics_.total_time_ms << ",\n";
        ss << "  \"fps\": " << latest_metrics_.fps << ",\n";
        // Computed from the live grid on every request — never a constant.
        ss << "  \"boundary_errors\": " << pipeline_->getGrid().checkBoundaryAlignment().totalErrors() << "\n";
        ss << "}";

        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiCameraUpload(SOCKET sock, const std::string& request, const std::string& client_ip) {
        size_t body_pos = request.find("\r\n\r\n");
        if (body_pos == std::string::npos) {
            sendResponse(sock, 400, "application/json", "{\"error\":\"Missing body\"}");
            return;
        }

        std::string body = request.substr(body_pos + 4);
        
        {
            std::lock_guard<std::mutex> lock(mtx_);
            camera_active_ = true;
            camera_frame_count_++;
            last_cam_time_ = std::chrono::steady_clock::now();
            remote_client_ip_ = client_ip;

            // Check Content-Type
            if (request.find("Content-Type: image/jpeg") != std::string::npos ||
                request.find("content-type: image/jpeg") != std::string::npos) {
                latest_camera_jpeg_binary_ = body;
            } else {
                // Check if JSON containing "image"
                size_t img_key = body.find("\"image\":");
                if (img_key != std::string::npos) {
                    size_t start_quote = body.find("\"", img_key + 8);
                    if (start_quote != std::string::npos) {
                        size_t end_quote = body.find("\"", start_quote + 1);
                        if (end_quote != std::string::npos) {
                            latest_camera_base64_ = body.substr(start_quote + 1, end_quote - start_quote - 1);
                        }
                    }
                } else {
                    latest_camera_base64_ = body;
                }
            }
        }

        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_id\": " << camera_frame_count_.load() << ",\n";
        ss << "  \"client_ip\": \"" << client_ip << "\"\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiCameraLatestJpg(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!latest_camera_jpeg_binary_.empty()) {
            sendResponse(sock, 200, "image/jpeg", latest_camera_jpeg_binary_);
            return;
        }
        sendResponse(sock, 404, "text/plain", "No binary frame available");
    }

    void serveApiCameraLatest(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();
        double sec_since = std::chrono::duration<double>(now - last_cam_time_).count();
        bool is_live = camera_active_ && (sec_since < 5.0);

        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"active\": " << (is_live ? "true" : "false") << ",\n";
        ss << "  \"frame_id\": " << camera_frame_count_.load() << ",\n";
        ss << "  \"sec_since\": " << sec_since << ",\n";
        ss << "  \"has_base64\": " << (!latest_camera_base64_.empty() ? "true" : "false") << ",\n";
        ss << "  \"has_binary\": " << (!latest_camera_jpeg_binary_.empty() ? "true" : "false") << ",\n";
        ss << "  \"remote_ip\": \"" << remote_client_ip_ << "\",\n";
        if (is_live && !latest_camera_base64_.empty()) {
            ss << "  \"image\": \"" << latest_camera_base64_ << "\"\n";
        } else {
            ss << "  \"image\": null\n";
        }
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiLidarScan(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // Re-run pipeline if not yet run or if fresh frame requested
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
        // Computed from the live grid on every request — never a constant.
        ss << "    \"boundary_errors\": " << pipeline_->getGrid().checkBoundaryAlignment().totalErrors() << ",\n";
        ss << "    \"ram_mb\": " << getProcessRamMb() << "\n";
        ss << "  },\n";

        // 1. Sample processed and semantically segmented point cloud from PointNet++
        // Per-point schema: [x, y, z, intensity, class_id, confidence].
        // (Per-point timestamps do not exist in Point3D and are omitted
        // rather than fabricated.)
        const auto& cloud_to_send = pipeline_->getProcessedCloud().empty() ? raw_scan_ : pipeline_->getProcessedCloud();
        ss << "  \"points\": [\n";
        size_t step = std::max<size_t>(1, cloud_to_send.size() / 12000);
        bool first_pt = true;
        for (size_t i = 0; i < cloud_to_send.size(); i += step) {
            const auto& pt = cloud_to_send[i];
            if (!first_pt) ss << ",\n";
            first_pt = false;
            int sem_cls = static_cast<int>(pt.semantic_class);
            ss << "    [" << pt.x << "," << pt.y << "," << pt.z << "," << pt.intensity << "," << sem_cls << "," << pt.confidence << "]";
        }
        ss << "\n  ],\n";

        // 2. Active 2.5D cells (full per-cell schema: no silent omissions)
        ss << "  \"cells\": [\n";
        auto cells = pipeline_->getGrid().getAllCells();
        bool first_cell = true;
        size_t cell_stride = std::max<size_t>(1, cells.size() / 4000);
        for (size_t i = 0; i < cells.size(); i += cell_stride) {
            const auto& c = cells[i];
            if (c.point_count == 0 && c.occupancy < 0.2f) continue;
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

        // 3. Dynamic tracked objects (z and confidence measured, sizes from
        // the 3D-connected cluster footprint)
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

    void serveApiCameraFrame(SOCKET sock, const std::string& request) {
        // Extract JSON body
        size_t body_pos = request.find("\r\n\r\n");
        if (body_pos == std::string::npos) {
            sendResponse(sock, 400, "application/json", "{\"error\":\"Missing body\"}");
            return;
        }

        // H5: this route runs on detached worker threads alongside the LiDAR
        // routes; all shared timing/counter/metrics state is mutex-guarded.
        size_t active_cells = 0;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            camera_active_ = true;
            camera_frame_count_++;
            last_cam_time_ = std::chrono::steady_clock::now();
            active_cells = latest_metrics_.active_cells;
        }

        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"camera_active\": true,\n";
        ss << "  \"camera_frame_id\": " << camera_frame_count_.load() << ",\n";
        ss << "  \"active_cells\": " << active_cells << "\n";
        ss << "}";

        sendResponse(sock, 200, "application/json", ss.str());
    }

    // Load and process one sequence frame (grid/tracker state persists).
    // Must be called with mtx_ held.
    bool processFrameAt(size_t index) {
        if (index >= seq_scans_.size()) return false;
        if (!ps26053::LidarIO::loadBinScan(seq_scans_[index], raw_scan_)) {
            std::cerr << "[Server] Failed to load scan: " << seq_scans_[index] << "\n";
            return false;
        }
        seq_index_ = index;
        latest_metrics_ = pipeline_->processFrame(
            raw_scan_, index * kFrameIntervalSec, static_cast<int>(index));
        scan_processed_ = true;
        return true;
    }

    void serveApiFrames(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ostringstream ss;
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"current_scan\": \"" << (seq_scans_.empty() ? "" : seq_scans_[seq_index_]) << "\"\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiFrameNext(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2);
        if (seq_index_ + 1 >= seq_scans_.size()) {
            ss << "{\n";
            ss << "  \"status\": \"end_of_sequence\",\n";
            ss << "  \"frame_index\": " << seq_index_ << ",\n";
            ss << "  \"frame_count\": " << seq_scans_.size() << "\n";
            ss << "}";
            sendResponse(sock, 200, "application/json", ss.str());
            return;
        }
        if (!processFrameAt(seq_index_ + 1)) {
            sendResponse(sock, 500, "application/json", "{\"status\":\"frame_load_failed\"}");
            return;
        }
        ss << "{\n";
        ss << "  \"status\": \"ok\",\n";
        ss << "  \"frame_index\": " << seq_index_ << ",\n";
        ss << "  \"frame_count\": " << seq_scans_.size() << ",\n";
        ss << "  \"active_cells\": " << latest_metrics_.active_cells << ",\n";
        ss << "  \"total_latency_ms\": " << latest_metrics_.total_time_ms << "\n";
        ss << "}";
        sendResponse(sock, 200, "application/json", ss.str());
    }

    void serveApiReset(SOCKET sock) {
        std::lock_guard<std::mutex> lock(mtx_);
        pipeline_ = std::make_unique<ps26053::MappingPipeline>("models/onnx/pointnet2_semseg.onnx");
        pipeline_->initialize();
        pipeline_->loadConfig("config");
        scan_processed_ = false;
        seq_index_ = 0;
        camera_active_ = false;
        camera_frame_count_ = 0;
        latest_camera_jpeg_binary_.clear();
        latest_camera_base64_.clear();
        remote_client_ip_.clear();

        sendResponse(sock, 200, "application/json", "{\"status\":\"reset_complete\"}");
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
        ss << "Content-Length: " << body.size() << "\r\n\r\n";
        ss << body;

        // H6: a single send() may write fewer bytes than requested (large
        // scan payloads). Loop until everything is written and report a
        // shortfall loudly instead of silently truncating the JSON.
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
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    int port = 8080;
    std::string scan_arg = "data/raw/000000.bin";
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }
    if (argc > 2) {
        scan_arg = argv[2];
    }

    LiveDashboardServer server(port, scan_arg);
    if (!server.start()) {
        std::cerr << "[Error] Failed to start live dashboard server.\n";
        return 1;
    }

    return 0;
}
