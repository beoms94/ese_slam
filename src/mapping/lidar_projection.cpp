#include <ese_slam/lidar_projection.hpp>

int main (int argc, char** argv)
{
    ros::init(argc,argv,"lidar_projection");

    LidarProjection LP;

    ros::Rate rate_(10);
    while(ros::ok())
    {
        ros::spinOnce();

        LP.runLidarProjection();

        rate_.sleep();
    }

    return 0;
}