#pragma once

#include "ps26053/common/types.hpp"
#include <Eigen/Dense>
#include <cmath>

namespace ps26053 {

/**
 * @brief Sensor pose in the world/vehicle frame (architecture spec, pipeline
 * stage 3: coordinate/pose transform).
 *
 * position is the sensor origin in world coordinates; yaw is the rotation
 * about the Z axis in radians (right-handed, counter-clockwise looking down
 * +Z). The default is the identity pose (sensor at the world origin,
 * axes aligned), which reproduces the historical behaviour explicitly
 * instead of through a hardcoded origin assumption.
 */
struct SensorPose {
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    float yaw{0.0f};

    Eigen::Matrix4f toMatrix() const {
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        float c = std::cos(yaw);
        float s = std::sin(yaw);
        m(0, 0) = c;  m(0, 1) = -s;
        m(1, 0) = s;  m(1, 1) = c;
        m.block<3, 1>(0, 3) = position;
        return m;
    }
};

class CoordinateTransform {
public:
    explicit CoordinateTransform(const SensorPose& pose = SensorPose());

    void setPose(const SensorPose& pose) { pose_ = pose; }
    const SensorPose& pose() const { return pose_; }

    /**
     * @brief Map a sensor-frame cloud into the world frame, in place.
     * Only xyz coordinates move; intensity, class and confidence are untouched.
     */
    void toWorld(PointCloud& cloud) const;

    /**
     * @brief Sensor origin in world coordinates (drives resolution-band
     * distances downstream — no longer assumed to be the origin).
     */
    Eigen::Vector3f sensorPosition() const { return pose_.position; }

private:
    SensorPose pose_;
};

} // namespace ps26053
