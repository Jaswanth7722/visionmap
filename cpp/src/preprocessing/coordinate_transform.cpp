#include "ps26053/preprocessing/coordinate_transform.hpp"
#include "ps26053/preprocessing/point_cloud_ops.hpp"

namespace ps26053 {

CoordinateTransform::CoordinateTransform(const SensorPose& pose) : pose_(pose) {}

void CoordinateTransform::toWorld(PointCloud& cloud) const {
    Eigen::Matrix4f tf = pose_.toMatrix();
    PointCloudOps::transformPointCloud(cloud, tf);
}

} // namespace ps26053
