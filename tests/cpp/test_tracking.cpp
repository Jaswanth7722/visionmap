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
        for (int i = 0; i < 16; ++i) {
            float dx = (i % 4) * 0.2f - 0.3f;
            float dy = (i / 4) * 0.2f - 0.3f;
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
    // Estimated velocity should be close to 2.0 m/s along X.
    // L2: the bound must mean something — 0.5x would pass at quarter speed.
    std::cout << "Track #" << tracks[0].id << " velocity: vx=" << tracks[0].velocity.x()
              << " vy=" << tracks[0].velocity.y() << std::endl;
    assert(tracks[0].velocity.x() > 1.0f);
    // H1: confidence is computed from history, never the default 0.
    assert(tracks[0].confidence > 0.5f);

    // H1: objects stacked in Z must not merge. Two 10-point clusters share
    // the same XY footprint but sit 2.5 m apart in height; the old 1 m XY
    // binning produced one track, 3D clustering must produce two.
    {
        ps26053::KalmanTracker zt;
        ps26053::PointCloud pts;
        for (int i = 0; i < 16; ++i) {
            ps26053::Point3D p;
            p.x = 10.0f + (i % 4) * 0.2f;
            p.y = 5.0f + (i / 4) * 0.2f;
            p.z = 0.5f;
            p.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
            pts.push_back(p);
        }
        for (int i = 0; i < 16; ++i) {
            ps26053::Point3D p;
            p.x = 10.0f + (i % 4) * 0.2f;
            p.y = 5.0f + (i / 4) * 0.2f;
            p.z = 3.0f;
            p.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
            pts.push_back(p);
        }
        zt.update(pts, 0.0);
        auto ztr = zt.getActiveTracks();
        assert(ztr.size() == 2);
        assert(ztr[0].confidence == 0.5f && ztr[1].confidence == 0.5f);
    }

    // H1: one 4.5 m vehicle must form one track, not one per 1 m bin.
    {
        ps26053::KalmanTracker vt;
        ps26053::PointCloud pts;
        for (float x = 10.0f; x < 14.5f; x += 0.3f) {
            for (float y = 4.6f; y < 5.4f; y += 0.3f) {
                ps26053::Point3D p;
                p.x = x; p.y = y; p.z = 0.5f;
                p.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
                pts.push_back(p);
            }
        }
        vt.update(pts, 0.0);
        auto vtr = vt.getActiveTracks();
        assert(vtr.size() == 1);
        // bbox spans the vehicle, far beyond a single 1 m bin
        assert(vtr[0].bbox.max_x - vtr[0].bbox.min_x > 3.0f);
    }

    // H1-width-gate: a 12 m wide (laterally) flat band — the shape of a
    // mislabeled monocular-depth image row, which spans the image width —
    // must not become a track, while the 4.5 m long vehicle above still does.
    {
        ps26053::KalmanTracker bt;
        ps26053::PointCloud pts;
        for (float y = 0.0f; y < 12.0f; y += 0.5f) {
            ps26053::Point3D p;
            p.x = 10.0f; p.y = y; p.z = 0.5f;
            p.semantic_class = ps26053::SemanticClass::DYNAMIC_OBSTACLE;
            pts.push_back(p);
        }
        assert(pts.size() >= 15);
        bt.update(pts, 0.0);
        assert(bt.getActiveTracks().empty());
    }

    std::cout << "[Test PASS] test_tracking succeeded!\n";
    return 0;
}
