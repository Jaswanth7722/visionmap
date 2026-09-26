#include "ps26053/tracking/kalman_tracker.hpp"
#include <cassert>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_tracking ...\n";

    ps26053::KalmanTracker tracker;

    // Simulate dynamic obstacle moving along X axis: x = 10.0 + 2.0 * t
    for (int frame = 0; frame < 5; ++frame) {
        double t = frame * 0.1;
        float x_pos = 10.0f + 2.0f * static_cast<float>(t);

        ps26053::PointCloud dynamic_points;
        for (int i = 0; i < 10; ++i) {
            float dx = (i % 3) * 0.2f - 0.2f;
            float dy = (i / 3) * 0.2f - 0.2f;
            ps26053::Point3D pt;
            pt.x = x_pos + dx;
            pt.y = 5.0f + dy;
            pt.z = 0.5f;
            pt.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
            dynamic_points.push_back(pt);
        }

        tracker.update(dynamic_points, t);
    }

    auto tracks = tracker.getActiveTracks();
    assert(!tracks.empty());
    assert(tracks[0].confirmed == true);
    // Estimated velocity should be close to 2.0 m/s along X
    std::cout << "Track #" << tracks[0].id << " velocity: vx=" << tracks[0].velocity.x()
              << " vy=" << tracks[0].velocity.y() << std::endl;
    assert(tracks[0].velocity.x() > 0.5f);

    std::cout << "[Test PASS] test_tracking succeeded!\n";
    return 0;
}
