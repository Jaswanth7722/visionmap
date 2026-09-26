// Worst-case regression test for row-band artifacts: project a real photo
// with EVERY point marked dynamic (as if the model labeled all of it), then
// assert the width gate admits no oversize cluster. CTest runs with CWD =
// build dir, hence the ../images/ path.
#include "ps26053/io/camera_ingest.hpp"
#include "ps26053/tracking/kalman_tracker.hpp"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

std::vector<uint8_t> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f.is_open());
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    return std::vector<uint8_t>(s.begin(), s.end());
}

} // namespace

int main() {
    std::cout << "[Test] Running test_photo_tracking ...\n";

    auto bytes = readFile("../images/car.png");
    ps26053::RgbImage img;
    assert(ps26053::CameraIngest::decodeImage(bytes.data(), bytes.size(), img));
    std::cout << "  decoded photo: " << img.width << "x" << img.height << "\n";

    auto cloud = ps26053::CameraIngest::imageToPointCloud(img, 4096);
    assert(!cloud.empty());
    for (auto& pt : cloud) {
        pt.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
    }

    ps26053::KalmanTracker tracker;
    tracker.update(cloud, 0.0);
    const auto& tracks = tracker.getActiveTracks();
    std::cout << "  worst-case tracks admitted: " << tracks.size() << "\n";
    for (const auto& tr : tracks) {
        float w = tr.bbox.max_x - tr.bbox.min_x;
        std::cout << "  track #" << tr.id << " width=" << w << "\n";
        assert(w <= 6.0f);
    }

    std::cout << "[Test PASS] test_photo_tracking succeeded!\n";
    return 0;
}
