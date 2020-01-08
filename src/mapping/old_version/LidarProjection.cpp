#include <ese_slam/slam_util_ver2.h>

class LidarProjection
{
private:
    ros::NodeHandle nh;

    ros::Subscriber sub_odom;
    ros::Subscriber sub_lidar;

    ros::Publisher pub_ground;
    ros::Publisher pub_laneMarker;
    ros::Publisher pub_projected;

    Vector3f Pose_;
    Vector3f RPY_;
    Vector4f Quaternion_;
    Matrix4f Transform_matrix_;

    PointI nanPoint;

    Pointcloud::Ptr input_cloud_;           //subscribed pointcloud msgs
    //Pointcloud::Ptr transformed_cloud_;     //transformed pointcloud
    Pointcloud::Ptr full_cloud_;

    Pointcloud::Ptr projected_cloud_;
    Pointcloud::Ptr ground_cloud_;
    Pointcloud::Ptr laneMarker_cloud_;

    cv::Mat rangeMat;
    cv::Mat labelMat;
    cv::Mat groundMat;

    std_msgs::Header cloudHeader;

public:
    LidarProjection():
    nh("~")
    {
        sub_odom = nh.subscribe("/odom/inu",10,&LidarProjection::odomMsgsCallBack,this);
        sub_lidar = nh.subscribe("/velodyne_points",10, &LidarProjection::lidarMsgsCallBack,this);

        pub_ground = nh.advertise<msgs_Point>("ground_cloud",1);
        pub_laneMarker = nh.advertise<msgs_Point>("laneMarker_cloud",1);
        pub_projected = nh.advertise<msgs_Point>("projected_cloud",1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        initializeValue();
        resetParameter();
    }
    ~LidarProjection(){}

    void initializeValue()
    {
        Pose_ << 0,0,0;
        Quaternion_ << 0,0,0,0;
        Transform_matrix_ << 0,0,0,0,
                             0,0,0,0,
                             0,0,0,0,
                             0,0,0,0;
    }

    void resetParameter()
    {
        input_cloud_.reset(new Pointcloud);
        //transformed_cloud_.reset(new Pointcloud);
        full_cloud_.reset(new Pointcloud);
        full_cloud_->points.resize(N_SCAN*Horizon_SCAN);

        projected_cloud_.reset(new Pointcloud);
        projected_cloud_->points.resize(N_SCAN*Horizon_SCAN);
        ground_cloud_.reset(new Pointcloud);
        laneMarker_cloud_.reset(new Pointcloud);

        input_cloud_->clear();
        //transformed_cloud_->clear();
        full_cloud_->clear();
        projected_cloud_->clear();
        ground_cloud_->clear();
        laneMarker_cloud_->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));

        std::fill(full_cloud_->points.begin(), full_cloud_->points.end(), nanPoint);
        std::fill(projected_cloud_->points.begin(), projected_cloud_->points.end(), nanPoint);

        return;
    }
    
    void copyPointCloud(const msgs_Point::ConstPtr& point_msg)
    {
        cloudHeader = point_msg->header;
        pcl::fromROSMsg(*point_msg,*input_cloud_);
    }

    void transformMatrixUpdate()
    {
        RPY_(0) = atan2(2*((Quaternion_(0)*Quaternion_(1)) + (Quaternion_(2)*Quaternion_(3))),
                        1 - 2*((Quaternion_(1)*Quaternion_(1)) + (Quaternion_(2)*Quaternion_(2))));
        RPY_(1) = asin(2*((Quaternion_(0)*Quaternion_(2)) - (Quaternion_(3)*Quaternion_(1))));
        RPY_(2) = atan2(2*((Quaternion_(0)*Quaternion_(3)) + (Quaternion_(1)*Quaternion_(2))),
                        1 - 2*((Quaternion_(2)*Quaternion_(2)) + (Quaternion_(3)*Quaternion_(3))));
        
        RPY_(2) += PI;

        Transform_matrix_ << cos(RPY_(0))*cos(RPY_(1)), 
                                (-1*sin(RPY_(0))*cos(RPY_(2)))+(cos(RPY_(0))*sin(RPY_(1))*sin(RPY_(2))), 
                                (sin(RPY_(0))*sin(RPY_(2)))+(cos(RPY_(0))*sin(RPY_(1))*cos(RPY_(2))), 
                                Pose_(0),
                                sin(RPY_(0))*cos(RPY_(1)), 
                                (cos(RPY_(0))*cos(RPY_(2)))+(sin(RPY_(0))*sin(RPY_(1))*cos(RPY_(2))), 
                                (-1*cos(RPY_(0))*sin(RPY_(2)))+(sin(RPY_(0))*sin(RPY_(1))*cos(RPY_(2))), 
                                Pose_(1),
                                -1*sin(RPY_(1)),
                                cos(RPY_(1))*sin(RPY_(2)), 
                                cos(RPY_(1))*cos(RPY_(2)), 
                                Pose_(2),
                                0, 0, 0, 0;
    }

    void transformPointCloud()
    {
        //transformed_cloud_->points.resize(input_cloud_->points.size());
        pcl::transformPointCloud(*projected_cloud_, *projected_cloud_, Transform_matrix_);
        pcl::transformPointCloud(*ground_cloud_, *ground_cloud_, Transform_matrix_);

        return;
    }

    void projectPointCloud()
    {
        float vertical_angle, horizon_angle, range;
        size_t row_idx, column_idx, idx, cloudSize;
        PointI thisPoint;

        cloudSize = input_cloud_->points.size();

        for(size_t i=0; i<cloudSize; ++i)
        {
            thisPoint.x = input_cloud_->points[i].x;
            thisPoint.y = input_cloud_->points[i].y;
            thisPoint.z = input_cloud_->points[i].z;
            thisPoint.intensity = input_cloud_->points[i].intensity;

            vertical_angle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / PI;
            row_idx = (vertical_angle + ang_bottom) / ang_res_y;
            if(row_idx < 0 || row_idx >= N_SCAN)
                continue;
            
            horizon_angle = atan2(thisPoint.x, thisPoint.y) * 180 / PI;

            if (horizon_angle <= -90)
                column_idx = -int(horizon_angle / ang_res_x) - 450; 
            else if (horizon_angle >= 0)
                column_idx = -int(horizon_angle / ang_res_x) + 1350;
            else
                column_idx = 1350 - int(horizon_angle / ang_res_x);
            
            idx = column_idx + row_idx * Horizon_SCAN;
            projected_cloud_->points[idx] = thisPoint;
            
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            rangeMat.at<float>(row_idx, column_idx) = range;
            thisPoint.intensity = (float)row_idx + (float)column_idx / 10000.0;
            
            full_cloud_->points[idx] = thisPoint;
           
        }
    }

    void findGround()
    {
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;

        for(size_t j=0; j<Horizon_SCAN; ++j)
        {
            for(size_t i=0; i<groundScanInd; ++i)
            {
                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                if (full_cloud_->points[lowerInd].intensity == -1 ||
                    full_cloud_->points[upperInd].intensity == -1){
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                diffX = full_cloud_->points[upperInd].x - full_cloud_->points[lowerInd].x;
                diffY = full_cloud_->points[upperInd].y - full_cloud_->points[lowerInd].y;
                diffZ = full_cloud_->points[upperInd].z - full_cloud_->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 5){
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }

        if (pub_ground.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        ground_cloud_->push_back(projected_cloud_->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void odomMsgsCallBack(const msgs_Odom::ConstPtr& odom_msg)
    {
        Pose_(0) = odom_msg->pose.pose.position.x;
        Pose_(1) = odom_msg->pose.pose.position.y;
        Pose_(2) = odom_msg->pose.pose.position.z;

        Quaternion_(0) = odom_msg->pose.pose.orientation.x;
        Quaternion_(1) = odom_msg->pose.pose.orientation.y;
        Quaternion_(2) = odom_msg->pose.pose.orientation.z;
        Quaternion_(3) = odom_msg->pose.pose.orientation.w;

        return;
    }

    void lidarMsgsCallBack(const msgs_Point::ConstPtr& point_msg)
    {
        //1. copy pointcloud
        //2. update transformation matrix
        //3. project pointcloud
        //4. find ground from projected pointcloud
        //5. find lane marker from ground pointcloud
        //6. transformation copied pointcloud to odom
        //7. publish processed pointclouds

        copyPointCloud(point_msg);
        //transformMatrixUpdate();
        projectPointCloud();
        findGround();
        //5. find lane marker from ground pointcloud
        //transformPointCloud();
        publishPoint();
        resetParameter();

        return;
    }

    void publishPoint()
    {
        msgs_Point temp_cloud;
        ros::Time current_time = ros::Time::now();

        pcl::toROSMsg(*projected_cloud_, temp_cloud);
        temp_cloud.header.stamp = current_time;
        temp_cloud.header.frame_id = "odom";
        pub_projected.publish(temp_cloud);

        pcl::toROSMsg(*ground_cloud_, temp_cloud);
        temp_cloud.header.stamp = current_time;
        temp_cloud.header.frame_id = "odom";
        pub_ground.publish(temp_cloud);

        //publish lane marker cloud 

        return;
    }
};

int main (int argc, char** argv)
{
    ros::init(argc, argv, "LidarProjection");

    LidarProjection LP;

    ROS_INFO("\033[1;32m---->\033[0m Lidar Projection Started.");

    ros::spin();
    return 0;
}