#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <math.h>
#include <memory.h>
#include <iostream>
#include <thread>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>

#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/registration/transforms.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/ndt.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/io/pcd_io.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#define PI 3.14159265

typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> Pointcloud;
typedef sensor_msgs::Imu msgs_Imu;
typedef sensor_msgs::PointCloud2 msgs_Point;
typedef nav_msgs::Odometry msgs_Odom;

typedef Eigen::Vector3f Vector3f;
typedef Eigen::Vector4f Vector4f;
typedef Eigen::Matrix3f Matrix3f;
typedef Eigen::Matrix4f Matrix4f;

struct PointXYZMAP
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    unsigned int index_;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZMAP,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (unsigned int, index_, index_)
                                   )

typedef PointXYZMAP MapInfo;
typedef pcl::PointCloud<MapInfo> PointMap;