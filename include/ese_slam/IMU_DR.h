#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <math.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#define PI 3.14159265

struct PointOS1
{
	PCL_ADD_POINT4D;
	float t;
  uint16_t reflectivity;
  uint16_t intensity;
  uint8_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointOS1,
																		(float, x, x)
																		(float, y, y)
																		(float, z, z)
																		(float, t, t)
																		(uint16_t, reflectivity, reflectivity)
																		(uint16_t, intensity, intensity)
																		(uint8_t, ring, ring)
)

typedef pcl::PointXYZI PointI;
typedef pcl::PointCloud<PointI> Pointcloud;
typedef sensor_msgs::PointCloud2 msgs_Point;
typedef sensor_msgs::Imu msgs_Imu;
typedef Eigen::Vector3f Vector3;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Matrix4f Matrix4;

typedef pcl::PointCloud<PointOS1> Point_OS1;

Vector3 original_accel, original_ang_vel;
Vector3 current_accel, current_ang_vel;         // 0->x 1->y 2->z 
Vector3 past_accel, last_ang_vel;
Vector3 Position, last_Position, Velocity; 

Pointcloud::Ptr IMU_odom (new Pointcloud);
Pointcloud::Ptr align_cloud (new Pointcloud);

double time_interval = 0;

Matrix3 Roll, Pitch, Yaw, IMU_rotation_matrix;

ros::Time imu_current_time, imu_last_time;

void TimeUpdate (void)
{
    imu_last_time = imu_current_time;
    imu_current_time = ros::Time::now();
    time_interval = (imu_current_time - imu_last_time).toSec();
}

void initRotationMatrix  
(const float imu_roll, const float imu_pitch, const float imu_yaw)
{
    float roll_  = imu_roll  ;//* PI/180;
    float pitch_ = imu_pitch ;//* PI/180;
    float yaw_   = imu_yaw   ;//* PI/180;

    Roll  << 1,0,0,
             0,cos(roll_),-1*sin(roll_),
             0,sin(roll_),cos(roll_);

    Pitch << cos(pitch_),0,sin(pitch_),
             0,1,0,
             -1*sin(pitch_),0,cos(pitch_);

    Yaw   << cos(yaw_),-1*sin(yaw_),0,
             sin(yaw_),cos(yaw_),0,
             0,0,1; 

    IMU_rotation_matrix = Roll * Pitch * Yaw;

    return ;
}

void IMUrotation (void)
{
    current_accel = IMU_rotation_matrix * original_accel;
    current_ang_vel = IMU_rotation_matrix * original_ang_vel;

    /*double Accel = sqrt(pow(current_accel(0),2)+pow(current_accel(1),2)+pow(current_accel(2),2));

    current_accel(0) =  current_accel(0) * -9.8/Accel;
    current_accel(1) =  current_accel(1) * -9.8/Accel;
    current_ang_vel(0) = -1 * current_ang_vel(0);
    current_ang_vel(1) = -1 * current_ang_vel(1); */
}

void estimateOdomFromImu (void)
{
    Position = Position + (Velocity * time_interval) + (current_accel * time_interval * time_interval /2);
    Velocity = Velocity + (current_accel * time_interval);

    Pointcloud::Ptr tempPose (new Pointcloud);
    tempPose->points.resize(1);
    tempPose->points[0].x = Position(1);
    tempPose->points[0].y = Position(0);
    tempPose->points[0].z = 0;
    tempPose->points[0].intensity = 0;

   std::cout << "original x :" << original_accel(0) << "\n"
             << "original y :" << original_accel(1) << "\n"
             << "original z :" << original_accel(2) << "\n\n"
             << "aligned x :" << current_accel(0) << "\n"
             << "aligned y :" << current_accel(1) << "\n"
             << "aligned z :" << current_accel(2) << "\n\n" 
             << "velocity x :" << Velocity(0) << "\n"
             << "velocity y :" << Velocity(1) << "\n"
             << "velocity z :" << Velocity(2) << "\n" << std::endl;

    *IMU_odom += *tempPose;
}
