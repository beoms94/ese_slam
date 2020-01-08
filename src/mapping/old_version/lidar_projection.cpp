#include <ese_slam/lidar_projection.hpp>
#include <can_parser/GWAY1.h>

#include <unistd.h>

int projection_init = 0;

void canMsgsCallBack(const can_parser::GWAY1::ConstPtr& gw1_msgs)
{
    if(projection_init == 99)
    {
        return;
    }

    double wheel_RR = gw1_msgs->Gway_Wheel_Velocity_RR;
    double wheel_RL = gw1_msgs->Gway_Wheel_Velocity_RL;
    double vx = (wheel_RR + wheel_RL)/(3.6*2);
    
    if(vx > 0)
    {
        projection_init = 1;
    }

    return;
}

int main (int argc, char** argv)
{
    ros::init(argc,argv,"lidar_projection");

    ros::NodeHandle ns;

    ros::Subscriber CANMsgs = ns.subscribe("CAN/GWAY1",100,canMsgsCallBack);

    LidarProjection LP;

    ros::Rate rate_(10);
    while(ros::ok())
    {
        ros::spinOnce();

        if(projection_init == 1)
        {
            LP.projection_init = projection_init;
            projection_init = 99;
            sleep(2);
        }

        LP.runLidarProjection();

        rate_.sleep();
    }

    return 0;
}