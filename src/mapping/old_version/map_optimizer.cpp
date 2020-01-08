//#include <ese_slam/slam_util.h>
#include <ese_slam/mapOptimizer.hpp>

int main (int argc, char** argv)
{
    ros::init (argc, argv, "map_optimizer");

    MapOptimizer MO;

    std::thread publishGlobalMapThread(&MapOptimizer::publishSampleMapThread, &MO);

    ros::Rate rate_(1);
    while(ros::ok())
    {
        ros::spinOnce();

        MO.runMapOptimizer();

        rate_.sleep();
    }

    return 0;
}