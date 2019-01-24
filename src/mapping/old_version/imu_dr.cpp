#include <ese_slam/IMU_dr.h>

unsigned int imu_cnt = 0;
unsigned int cloud_flag = 0;

void imuCallback(const msgs_Imu::ConstPtr& imuMsgs)
{
    last_accel = recent_accel;
    last_angVel = recent_angVel;

    timeUpdate();
    if(imu_cnt == 0)
    {
        last_accel(0) = imuMsgs->linear_acceleration.x;
        last_accel(1) = imuMsgs->linear_acceleration.y;
        last_accel(2) = imuMsgs->linear_acceleration.z;

        last_angVel(0) = imuMsgs->angular_velocity.x;
        last_angVel(1) = imuMsgs->angular_velocity.y;
        last_angVel(2) = imuMsgs->angular_velocity.z;

        imu_cnt = 1;
    }

    recent_accel(0) = imuMsgs->linear_acceleration.x;
    recent_accel(1) = imuMsgs->linear_acceleration.y;
    recent_accel(2) = imuMsgs->linear_acceleration.z;

    recent_angVel(0) = imuMsgs->angular_velocity.x;
    recent_angVel(1) = imuMsgs->angular_velocity.y;
    recent_angVel(2) = imuMsgs->angular_velocity.z;
}

void cloudCallback (const msgs_Point::ConstPtr& cloudMsgs)
{

}

void publishOdom (ros::Publisher imu_odom)
{
    msgs_Point imuOdom;
    pcl::toROSMsg (*IMU_odom, imuOdom);
    imuOdom.header.frame_id = "base_link";
    imuOdom.header.stamp = ros::Time::now();

    imu_odom.publish(imuOdom);
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "imu_dr");
    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    std::string imu_msg, cloud_msg;
    float imu_roll, imu_pitch, imu_yaw;

    ph.param<std::string>("imu_msg",imu_msg,"os1_node/imu");
    ph.param<std::string>("cloud_msg",cloud_msg,"os1_node/points");
    ph.param("imu_roll",imu_roll, 0.0f);
    ph.param("imu_pitch",imu_pitch, 0.0f);
    ph.param("imu_yaw",imu_yaw, 0.0f);
    
    ros::Publisher imu_odom = nh.advertise<msgs_Point>("odom",50);
    ros::Publisher cloud_pub = nh.advertise<msgs_Point>("map_cloud",1000);
    
    ros::Subscriber imu_sub = nh.subscribe(imu_msg,1000,imuCallback);
    ros::Subscriber cloud_sub = nh.subscribe(cloud_msg,1000,cloudCallback);

    imu_dr_init();
    
    ros::Rate rate(10);
    while(ros::ok())
    {
        ros::spinOnce();

        EstimateVelocity();
        imuDeadReckoning();

        publishOdom(imu_odom);

        rate.sleep();
    }
    
}

/*
void imuCallback (const msgs_Imu::ConstPtr& imuMsgs)
{
    //ROS_INFO("CallBack!");
    original_accel[0] = imuMsgs->linear_acceleration.x;
    original_accel[1] = imuMsgs->linear_acceleration.y;
    original_accel[2] = imuMsgs->linear_acceleration.z;

    original_ang_vel[0] = imuMsgs->angular_velocity.x;
    original_ang_vel[1] = imuMsgs->angular_velocity.y;
    original_ang_vel[2] = imuMsgs->angular_velocity.z;

    IMUrotation();

    return ;
}

void cloudCallback (const msgs_Point::ConstPtr& cloudMsgs)
{
    Point_OS1::Ptr temp_cloud (new Point_OS1);
    pcl::fromROSMsg(*cloudMsgs,*temp_cloud);

    size_t point_num = temp_cloud->points.size();
    align_cloud->points.resize(point_num);

    for(size_t i = 0; i<point_num; i++)
    {
        align_cloud->points[i].x = temp_cloud->points[i].x + Position(1);
        align_cloud->points[i].y = temp_cloud->points[i].y + Position(0);
        align_cloud->points[i].z = temp_cloud->points[i].z;
        align_cloud->points[i].intensity = 0;
    }

    cloud_flag = 1;
    return;
}

void publishOdom (ros::Publisher imu_odom)
{
    msgs_Point imuOdom;
    pcl::toROSMsg (*IMU_odom, imuOdom);
    imuOdom.header.frame_id = "base_link";
    imuOdom.header.stamp = ros::Time::now();

    imu_odom.publish(imuOdom);
}

void publishCloud (ros::Publisher cloud_pub)
{
    msgs_Point cloudPub;
    pcl::toROSMsg(*align_cloud,cloudPub);
    cloudPub.header.frame_id = "base_link";
    cloudPub.header.stamp = ros::Time::now();

    cloud_pub.publish(cloudPub); 
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "imu_dr");
    ros::NodeHandle nh;
    ros::NodeHandle ph("~");

    std::string imu_msg, cloud_msg;
    float imu_roll, imu_pitch, imu_yaw;

    ph.param<std::string>("imu_msg",imu_msg,"os1_node/imu");
    ph.param<std::string>("cloud_msg",cloud_msg,"os1_node/points");
    ph.param("imu_roll",imu_roll, 0.0f);
    ph.param("imu_pitch",imu_pitch, 0.0f);
    ph.param("imu_yaw",imu_yaw, 0.0f);
    
    ros::Publisher imu_odom = nh.advertise<msgs_Point>("odom",50);
    ros::Publisher cloud_pub = nh.advertise<msgs_Point>("map_cloud",1000);
    
    ros::Subscriber imu_sub = nh.subscribe(imu_msg,1000,imuCallback);
    ros::Subscriber cloud_sub = nh.subscribe(cloud_msg,1000,cloudCallback);

    std::cout << "roll : " << imu_roll << "\n"
              << "pitch : " << imu_pitch << "\n"
              << "yaw : " << imu_yaw << "\n" << std::endl;

    //initial time 
    imu_current_time = ros::Time::now();
    imu_last_time = ros::Time::now();

    initRotationMatrix(imu_roll, imu_pitch, imu_yaw);
    ROS_INFO ("Matrix Initilize!!");

    ros::Rate rate(10);
    while(ros::ok())
    {
        //ros::spinOnce();

        TimeUpdate();
        estimateOdomFromImu();

        publishOdom(imu_odom);
        if(cloud_flag == 1)
        {
            publishCloud(cloud_pub);
            cloud_flag == 0;
        }

        ros::spinOnce();
        rate.sleep();
    }
    
}*/
