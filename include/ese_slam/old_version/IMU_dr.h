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
typedef pcl::PointCloud<PointOS1> Point_OS1;
typedef sensor_msgs::PointCloud2 msgs_Point;
typedef sensor_msgs::Imu msgs_Imu;
typedef Eigen::Vector3f Vector3;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::Matrix4f Matrix4;

Vector3 recent_accel, last_accel, recent_angVel, last_angVel, computed_accel;
Vector3 Velocity, Position;
ros::Time imu_recent_time, imu_last_time;

double dt, yaw_rate, theta, delta_theta;

Pointcloud::Ptr IMU_odom (new Pointcloud);

void imu_dr_init(void)
{
    recent_accel << 0,0,0;
    last_accel << 0,0,0;
    recent_angVel << 0,0,0;
    last_angVel << 0,0,0;
    computed_accel << 0,0,0;

    imu_recent_time = ros::Time::now();
    imu_last_time = ros::Time::now();
}

void timeUpdate(void)
{
    imu_last_time = imu_recent_time;
    imu_recent_time = ros::Time::now();
    dt = (imu_recent_time - imu_last_time).toSec();
}

void EstimateVelocity(void)
{
    //Vector3 d_vel = (recent_accel - last_accel)*dt*dt/2;

    //Velocity += d_vel;

    Velocity = Velocity + (last_accel + (recent_accel - last_accel)/2)*dt;

    std::cout << "accel x :" << recent_accel(0) << "\n"
              << "accel y :" << recent_accel(1) << "\n"
              << "accel z :" << recent_accel(2) << "\n" 
              << "time interval :" << dt << "\n" << std::endl;

    std::cout << "velo x :" << Velocity(0) << "\n"
              << "velo y :" << Velocity(1) << "\n"
              << "velo z :" << Velocity(2) << "\n" << std::endl;
}

/*void EstimateVelocity(void)
{
    Vector3 d_accel = recent_accel - last_accel;

    Velocity(0) = Velocity(0) + last_accel(0)*dt + d_accel(0)*dt/2;
    Velocity(1) = Velocity(1) + last_accel(1)*dt + d_accel(1)*dt/2;
    Velocity(2) = Velocity(2) + last_accel(2)*dt + d_accel(2)*dt/2;

    std::cout << "accel x :" << recent_accel(0) << "\n"
              << "accel y :" << recent_accel(1) << "\n"
              << "accel z :" << recent_accel(2) << "\n" 
              << "time interval :" << dt << "\n" << std::endl;

    std::cout << "velo x :" << Velocity(0) << "\n"
              << "velo y :" << Velocity(1) << "\n"
              << "velo z :" << Velocity(2) << "\n" << std::endl;
}*/

void imuDeadReckoning (void)
{
    yaw_rate = recent_angVel(2);
    //double Vel = sqrt(pow(Velocity(0),2) + pow(Velocity(1),2));

    //double delta_x = Vel*dt*cos(theta + (delta_theta/2));
    //double delta_y = Vel*dt*sin(theta + (delta_theta/2));

    double delta_x = Velocity(0)*dt;
    double delta_y = Velocity(1)*dt;

    if(delta_x != 0 && delta_y != 0)
        delta_theta = yaw_rate * dt;
    Position(0) += delta_x;
    Position(1) += delta_y;
    theta += delta_theta;

    Pointcloud::Ptr tempPose (new Pointcloud);
    tempPose->points.resize(1);
    tempPose->points[0].x = Position(1);
    tempPose->points[0].y = Position(0);
    tempPose->points[0].z = 0;
    tempPose->points[0].intensity = 0;
    *IMU_odom += *tempPose;

    std::cout << "Pos x :" << Position(0) << "\n"
              << "Pos y :" << Position(1) << "\n"
              << "Pos z :" << Position(2) << "\n" << std::endl;
}