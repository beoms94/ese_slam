#include "slam_util.h"

#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>              //이걸 과연 쓰게 될까 이게 뭔지도 모르는데 이것까지 알아봐야 하는거야?

using namespace gtsam;
using namespace std;

class MapOptimizer
{
    private:
        NonlinearFactorGraph gtsamGraph;
        Values initialEstimate;
        Values optimizedEstimate;

        noiseModel::Diagonal::shared_ptr priorNoise;
        noiseModel::Diagonal::shared_ptr odometryNoise;

        ros::NodeHandle nh;

        ros::Subscriber projectedLidarMsgs;
        ros::Subscriber projectedLidarOdomMsgs;

        ros::Publisher pubGlobalMap;
        ros::Publisher pubSampleMap;
        ros::Publisher pubTrajectory;
        ros::Publisher pubReMap;

        msgs_Point GlobalMap_;
        msgs_Point SampleMap_;
        msgs_Point Trajectory_;
        msgs_Point ReMap_;

        ros::Time last_sub_time;
        ros::Time last_processing_time;
        double mapping_processing_interval;

        PointMap::Ptr temp_cloud;
        PointMap::Ptr GlobalMap;

        Pointcloud globalMap[2000];

        Pointcloud::Ptr SampleMap;
        Pointcloud::Ptr voxeled_SampleMap;
        Pointcloud::Ptr next_SampleMap;
        Pointcloud::Ptr projected_cloud;
        Pointcloud::Ptr temp_global;

        Pointcloud::Ptr TargetCloud;
        Pointcloud::Ptr InputCloud;

        size_t TargetCloudMapIndex;
        size_t InputCloudMapIndex;

        Pointcloud::Ptr cloudKeyPoses2D;

        Vector3f projectedPose;
        Vector3f projectedVelocity;
        Vector3f projectedRPY;
        Vector4f projectedQuaternion;

        Vector3f current_pose;
        Vector3f last_pose;

        unsigned int current_index;
        unsigned int next_index;

        unsigned int Map_index;

        bool loopClosureEnableFlag;

        pcl::ApproximateVoxelGrid<PointI> voxel;

    public:
        MapOptimizer():
            nh("~")
            {
                projectedLidarMsgs = nh.subscribe("/lidar_projection/projected_cloud",10,&MapOptimizer::lidarCallBack,this);
                projectedLidarOdomMsgs = nh.subscribe("/lidar_projection/projected_cloud_odom",10,&MapOptimizer::odomCallBack,this);

                pubGlobalMap = nh.advertise<msgs_Point>("Global_map",10);
                pubSampleMap = nh.advertise<msgs_Point>("Sample_map",10);
                pubTrajectory = nh.advertise<msgs_Point>("Trajectory",10);
                pubReMap = nh.advertise<msgs_Point>("Remap",10);

                initializeValue();
            }
            
        void initializeValue()
        {
            priorNoise = noiseModel::Diagonal::Sigmas(Vector3(1e-6,1e-6,1e-8));
            odometryNoise = noiseModel::Diagonal::Sigmas(Vector3(1e-6,1e-6,1e-7));

            last_sub_time = ros::Time::now();
            last_processing_time = last_sub_time;
            mapping_processing_interval = 0.01;

            temp_cloud.reset(new PointMap);
            GlobalMap.reset(new PointMap);

            voxeled_SampleMap.reset(new Pointcloud);
            SampleMap.reset(new Pointcloud);
            next_SampleMap.reset(new Pointcloud);
            projected_cloud.reset(new Pointcloud);
            temp_global.reset(new Pointcloud);

            cloudKeyPoses2D.reset(new Pointcloud);

            TargetCloud.reset(new Pointcloud);
            InputCloud.reset(new Pointcloud);

            TargetCloudMapIndex = 0;
            InputCloudMapIndex = 0;

            projectedPose << 0,0,0;
            projectedVelocity << 0,0,0;
            projectedRPY << 0,0,0;
            projectedQuaternion << 0,0,0,0;

            current_pose << 0,0,0;
            last_pose << 0,0,0;

            current_index = 0;
            next_index = 0;

            Map_index = 1;

            loopClosureEnableFlag = false;

            voxel.setLeafSize (0.25,0.25,0.25);
        }

        void getRPYfromQuaternion()
        {
            double  yaw_;
            yaw_ = atan2(2*((projectedQuaternion(0)*projectedQuaternion(1)) + (projectedQuaternion(2)*projectedQuaternion(3))),
                            1 - 2*((projectedQuaternion(1)*projectedQuaternion(1)) + (projectedQuaternion(2)*projectedQuaternion(2))));
           projectedRPY(2) = yaw_;

           current_pose << projectedPose(0), projectedPose(1), yaw_;

           return;
        }

        void voxelfiltering()
        {
            voxeled_SampleMap->clear();
            voxel.setInputCloud(SampleMap);
            voxel.filter(*voxeled_SampleMap);
        }

        void updateGlobalMap()
        {
            if(!(SampleMap->points.empty()))
            {
                voxelfiltering();

                globalMap[Map_index].points.resize(voxeled_SampleMap->points.size());
                pcl::copyPointCloud(*voxeled_SampleMap, globalMap[Map_index]);
                
                publishSampleMap();
                //publishGlobalMap();
                publishTrajectory();
            }
        }

        void updateSampleMap()
        {
            if(cloudKeyPoses2D->points.empty() && current_index < 10)
            {
                if(SampleMap->points.empty())
                    *SampleMap = *projected_cloud;
                else
                    *SampleMap += *projected_cloud;
                current_index++;
            }
            else
            {
                *SampleMap += *projected_cloud;
                *next_SampleMap += *projected_cloud;
                current_index++;
                next_index++;

                if(cloudKeyPoses2D->points.empty() && current_index == 10)
                {
                    cloudKeyPoses2D->points.resize(1);
                    cloudKeyPoses2D->points[0].x = current_pose(0);
                    cloudKeyPoses2D->points[0].y = current_pose(1);
                    cloudKeyPoses2D->points[0].z = current_pose(2);
                    cloudKeyPoses2D->points[0].intensity = 0;
                }

                if(current_index == 20 && next_index == 10)
                {
                    //save sample map
                    Pointcloud::Ptr temp_cloud (new Pointcloud);
                    temp_cloud->points.resize(1);
                    temp_cloud->points[0].x = current_pose(0);
                    temp_cloud->points[0].y = current_pose(1);
                    temp_cloud->points[0].z = current_pose(2);
                    temp_cloud->points[0].intensity = cloudKeyPoses2D->points.size();
                    *cloudKeyPoses2D += *temp_cloud;

                    //downsampling the samplemap
                    voxelfiltering();

                    //update Map data
                    updateGlobalMap();
                    factorGraphLoop();

                    //save sample Map
                    std::string file_adrs = "/home/mbek/Desktop/bag_file/saved_samplemap/";
                    std::stringstream ss;
                    ss << file_adrs << Map_index << ".pcd";

                    voxeled_SampleMap->height = 1;
                    voxeled_SampleMap->width = voxeled_SampleMap->points.size();
                    pcl::io::savePCDFileASCII (ss.str(), *voxeled_SampleMap);
                    ROS_INFO("write pcd file.");
                    
                    //next->current
                    SampleMap->clear();
                    *SampleMap = *next_SampleMap;
                    next_SampleMap->clear();

                    current_index = 10;
                    next_index = 0;

                    Map_index++;
                }
            }
        }

        void factorGraphLoop ()
        {
            if(gtsamGraph.empty())
            {
                Pose2 priorMean(0.0, 0.0, 0.0);
                gtsamGraph.add(PriorFactor<Pose2>(0,priorMean,priorNoise));
                initialEstimate.insert(0,priorMean);
            }
            else
            {
                int map_index = cloudKeyPoses2D->points.size();
                Pose2 odometryFrom(cloudKeyPoses2D->points[map_index-2].x, cloudKeyPoses2D->points[map_index-2].y, cloudKeyPoses2D->points[map_index-2].z);
                Pose2 odometryTo(cloudKeyPoses2D->points[map_index-1].x, cloudKeyPoses2D->points[map_index-1].y, cloudKeyPoses2D->points[map_index-1].z);
                gtsamGraph.add(BetweenFactor<Pose2>(map_index-2,map_index-1,odometryFrom.between(odometryTo),odometryNoise));
                initialEstimate.insert(map_index-1,Pose2(cloudKeyPoses2D->points[map_index-1].x, cloudKeyPoses2D->points[map_index-1].y, cloudKeyPoses2D->points[map_index-1].z));
            }

            loopClosureEnableFlag = detectLoopClosure();
        }

        bool detectLoopClosure()
        {

        }

        void lidarCallBack(const msgs_Point::ConstPtr& cloud_msg)
        {
            last_sub_time = ros::Time::now();

            pcl::fromROSMsg(*cloud_msg,*projected_cloud);

            updateSampleMap();

            return;
        }

        void odomCallBack(const msgs_Odom::ConstPtr& odom_msg)
        {
            last_pose = current_pose;

            projectedPose(0) = odom_msg->pose.pose.position.x;
            projectedPose(1) = odom_msg->pose.pose.position.y;
            projectedPose(2) = odom_msg->pose.pose.position.z;

            projectedQuaternion(0) = odom_msg->pose.pose.orientation.x;
            projectedQuaternion(1) = odom_msg->pose.pose.orientation.y;
            projectedQuaternion(2) = odom_msg->pose.pose.orientation.z;
            projectedQuaternion(3) = odom_msg->pose.pose.orientation.w;

            getRPYfromQuaternion();

            return;
        }

        void loopClosureTread()
        {
            if(loopClosureEnableFlag == false)
                return;
            
            ros::Rate rate_loopclosure(1);
            while(ros::ok())
            {
                rate_loopclosure.sleep();
                performLoopClosure();
            }
        }

        void publishGlobalMapThread()
        {
            ros::Rate rate_globalMap(1);
            while(ros::ok())
            {
                rate_globalMap.sleep();
                publishGlobalMap();
            }
        }

        void publishTrajectory()
        {
            ros::Time pub_time = ros::Time::now();

            pcl::toROSMsg(*cloudKeyPoses2D, Trajectory_);
            Trajectory_.header.frame_id = "odom";
            Trajectory_.header.stamp = pub_time;

            pubTrajectory.publish(Trajectory_);
        }

        void publishSampleMap()
        {
            ros::Time pub_time = ros::Time::now();

            pcl::toROSMsg(*voxeled_SampleMap,SampleMap_);
            SampleMap_.header.frame_id = "odom";
            SampleMap_.header.stamp = pub_time;

            pubSampleMap.publish(SampleMap_);
        }

        void publishGlobalMap()
        {            
            ros::Time pub_time = ros::Time::now();

            *temp_global += globalMap[Map_index-1];

            pcl::toROSMsg(*temp_global, GlobalMap_);
            GlobalMap_.header.frame_id = "odom";
            GlobalMap_.header.stamp = pub_time;

            pubGlobalMap.publish(GlobalMap_);
            
            return;
        }

        void performLoopClosure()
        {
            return;
        }

        void runMapOptimizer()
        {
            if((last_sub_time - last_processing_time).toSec() > mapping_processing_interval)
            {
                last_processing_time = last_sub_time;
            }
        }
};