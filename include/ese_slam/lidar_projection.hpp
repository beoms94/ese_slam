#include "slam_util.h"

class LidarProjection
{
    private:
        ros::NodeHandle nh;

        ros::Subscriber odomMsgs;
        ros::Subscriber lidarMsgs;

        ros::Publisher pubMap;
        ros::Publisher pubLidarOdom;

        msgs_Point map_data_;
        msgs_Odom projected_odom_data_;

        Vector3f Pose_;
        Vector3f Velocity_;
        Vector3f RPY_;
        Vector4f Quaternion_;
        Matrix4f Transform_matrix_;

        ros::Time odom_recent_time, odom_last_time, cloud_recent_time;
        double odom_dt;

        Pointcloud::Ptr sub_Cloud;
        Pointcloud::Ptr transformed_Cloud;

    public:
        LidarProjection():
            nh("~")
            {
                odomMsgs = nh.subscribe("/odom",10, &LidarProjection::odomMsgsCallBack,this);
                lidarMsgs = nh.subscribe("/velodyne_points",10, &LidarProjection::lidarMsgsCallBack,this);

                pubMap = nh.advertise<msgs_Point>("projected_cloud",10);
                pubLidarOdom = nh.advertise<msgs_Odom>("projected_cloud_odom",10);

                initializeValue();
            }

        void initializeValue()
        {
            Pose_ << 0,0,0;
            Velocity_ << 0,0,0;
            Quaternion_ << 0,0,0,0;
            Transform_matrix_ << 0,0,0,0,
                                 0,0,0,0,
                                 0,0,0,0,
                                 0,0,0,0;

            sub_Cloud.reset(new Pointcloud);
            transformed_Cloud.reset(new Pointcloud);

            odom_recent_time = ros::Time::now();
            odom_last_time = ros::Time::now();
            cloud_recent_time = ros::Time::now();
            odom_dt = 0;
        }

        void odomTimeUpdate()
        {
            odom_last_time = odom_recent_time;
            odom_recent_time = ros::Time::now();

            odom_dt = (odom_recent_time - odom_last_time).toSec();
        }

        void transformMatrixUpdate()
        {
            RPY_(0) = atan2(2*((Quaternion_(0)*Quaternion_(1)) + (Quaternion_(2)*Quaternion_(3))),
                            1 - 2*((Quaternion_(1)*Quaternion_(1)) + (Quaternion_(2)*Quaternion_(2))));
            RPY_(1) = asin(2*((Quaternion_(0)*Quaternion_(2)) - (Quaternion_(3)*Quaternion_(1))));
            RPY_(2) = atan2(2*((Quaternion_(0)*Quaternion_(3)) + (Quaternion_(1)*Quaternion_(2))),
                            1 - 2*((Quaternion_(2)*Quaternion_(2)) + (Quaternion_(3)*Quaternion_(3))));
            
            //RPY_ = RPY_ * -1*PI/180;

            std::cout << "radian x :" << RPY_(0) << "\n"
                      << "radian y :" << RPY_(1) << "\n"
                      << "radian z :" << RPY_(2) << "\n"
                      << "degree x :" << RPY_(0)*180/PI << "\n"
                      << "degree y :" << RPY_(1)*180/PI << "\n"
                      << "degree z :" << RPY_(2)*180/PI << "\n" << std::endl;

            Transform_matrix_ << cos(RPY_(0)), -1*sin(RPY_(0)), 0 , Pose_(0),
                                 sin(RPY_(0)), cos(RPY_(0))   , 0 , Pose_(1),
                                 0           , 0              , 1 , Pose_(2),
                                 0           , 0              , 0 , 0;
        }

        void publishTransformedCloud()
        {
            ros::Time current_time = ros::Time::now();

            pcl::toROSMsg (*transformed_Cloud, map_data_);
            map_data_.header.frame_id = "odom";
            map_data_.header.stamp = current_time;
            pubMap.publish(map_data_);

            projected_odom_data_.header.frame_id = "odom";
            projected_odom_data_.header.stamp = current_time;

            projected_odom_data_.pose.pose.position.x = Pose_(0);
            projected_odom_data_.pose.pose.position.y = Pose_(1);
            projected_odom_data_.pose.pose.position.z = Pose_(2);
            projected_odom_data_.pose.pose.orientation.x = Quaternion_(0);
            projected_odom_data_.pose.pose.orientation.y = Quaternion_(1);
            projected_odom_data_.pose.pose.orientation.z = Quaternion_(2);
            projected_odom_data_.pose.pose.orientation.w = Quaternion_(3);
            projected_odom_data_.twist.twist.linear.x = Velocity_(0);
            projected_odom_data_.twist.twist.linear.y = Velocity_(1);
            projected_odom_data_.twist.twist.linear.z = Velocity_(2);

            pubLidarOdom.publish(projected_odom_data_);

            return;
        }

        void odomMsgsCallBack(const msgs_Odom::ConstPtr& odom_msg)
        {
            odomTimeUpdate();

            Pose_(0) = odom_msg->pose.pose.position.x;
            Pose_(1) = odom_msg->pose.pose.position.y;
            Pose_(2) = odom_msg->pose.pose.position.z;

            Quaternion_(0) = odom_msg->pose.pose.orientation.x;
            Quaternion_(1) = odom_msg->pose.pose.orientation.y;
            Quaternion_(2) = odom_msg->pose.pose.orientation.z;
            Quaternion_(3) = odom_msg->pose.pose.orientation.w;

            Velocity_(0) = odom_msg->twist.twist.linear.x;
            Velocity_(1) = odom_msg->twist.twist.linear.y;
            Velocity_(2) = odom_msg->twist.twist.linear.z;

            return;
        }

        void lidarMsgsCallBack(const msgs_Point::ConstPtr& point_msg)
        {
            pcl::fromROSMsg(*point_msg,*sub_Cloud);

            transformed_Cloud->points.resize(sub_Cloud->points.size());

            transformMatrixUpdate();

            pcl::transformPointCloud(*sub_Cloud, *transformed_Cloud, Transform_matrix_);

            publishTransformedCloud();
            return;
        }

        void runLidarProjection()
        {

        }
};