#include "ps26053/preprocessing/coordinate_transform.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    std::cout << "[Test] Running test_coordinate_transform ...\n";

    // Identity pose preserves every point exactly.
    {
        ps26053::CoordinateTransform tf;
        ps26053::PointCloud cloud;
        ps26053::Point3D p;
        p.x = 3.0f; p.y = -4.0f; p.z = 1.5f; p.intensity = 0.7f;
        p.semantic_class = ps26053::SemanticClass::STATIC_OBSTACLE;
        p.confidence = 0.9f;
        cloud.push_back(p);
        tf.toWorld(cloud);
        assert(cloud[0].x == 3.0f && cloud[0].y == -4.0f && cloud[0].z == 1.5f);
        assert(cloud[0].intensity == 0.7f);
        assert(cloud[0].semantic_class == ps26053::SemanticClass::STATIC_OBSTACLE);
        assert(cloud[0].confidence == 0.9f);
        assert(tf.sensorPosition().isZero(0));
    }

    // Yaw = +90 deg maps sensor +X to world +Y; translation then applies.
    {
        const float kHalfPi = std::acos(-1.0f) * 0.5f;
        ps26053::SensorPose pose;
        pose.position = Eigen::Vector3f(10.0f, 0.0f, 0.0f);
        pose.yaw = kHalfPi;
        ps26053::CoordinateTransform tf(pose);
        ps26053::PointCloud cloud;
        ps26053::Point3D p;
        p.x = 1.0f; p.y = 0.0f; p.z = 0.0f;
        cloud.push_back(p);
        tf.toWorld(cloud);
        assert(std::fabs(cloud[0].x - 10.0f) < 1e-5f);
        assert(std::fabs(cloud[0].y - 1.0f) < 1e-5f);
        assert(std::fabs(cloud[0].z - 0.0f) < 1e-5f);
        assert((tf.sensorPosition() - Eigen::Vector3f(10.0f, 0.0f, 0.0f)).norm() < 1e-6f);
    }

    std::cout << "[Test PASS] test_coordinate_transform succeeded!\n";
    return 0;
}
