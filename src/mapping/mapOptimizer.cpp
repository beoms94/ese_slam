#include <ese_slam/slam_util.h>

#include "ese_slam/save_sample_map.h"
#include "ese_slam/save_global_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class MapOptimizer{
    private:
        // -------   GTSAM   ------- //
        NonlinearFactorGraph graph;

        Values init_value;
        Values optimized_value;

        noiseModel::Diagonal::shared_ptr priorNoise;
        noiseModel::Diagonal::shared_ptr odometryNoise;
        noiseModel::Diagonal::shared_ptr ndtNoise;

        size_t graph_idx;
        size_t crnt_idx;
        bool loopclosure_flag;
        bool isLoop;
        double search_size;
        double threshold_score;
        double ndt_align_score;

        size_t nearest_node_idx;
        Vector3 nearest_node_point;
        Vector3 nearest_node_rpy;
        Matrix4f init_tf_matrix;
        Matrix4f ndt_tf_matrix;
        Eigen::Affine3f ndt_tf_affine;
        Vector3 ndt_tf_position;
        Vector3 ndt_tf_rpy;

        ros::Publisher pub_optimized_trajectory;

        // ------- ROS basic ------- //
        ros::NodeHandle nh;

        ros::Subscriber sub_laneMarker;
        ros::Subscriber sub_object;
        ros::Subscriber sub_odometry;
        ros::Subscriber sub_fix;

        //ros::Publisher pub_transformed;
        ros::Publisher pub_sample_map;
        ros::Publisher pub_trajectory;
        ros::Publisher pub_originMap;
        ros::Publisher pub_target_map;
        ros::Publisher pub_input_map;
        ros::Publisher pub_optimized_odom;

        // ------- for debuging ------- //
        ros::Publisher pub_origin_trajectory;
        // ---------------------------- //

        ros::ServiceServer sample_service;
        ros::ServiceServer global_service;

        tf::TransformBroadcaster odom_broadcaster;

        Pointcloud::Ptr input_lane_cloud_;
        Pointcloud::Ptr transformed_lane_cloud_;
        Pointcloud::Ptr input_objt_cloud_;
        Pointcloud::Ptr transformed_objt_cloud_;

        Pointcloud::Ptr last_lane_sample_map;
        Pointcloud::Ptr crnt_lane_sample_map;
        Pointcloud::Ptr last_objt_sample_map;
        Pointcloud::Ptr crnt_objt_sample_map;
        Pointcloud::Ptr global_map;

        Pointcloud::Ptr original_map_position;
        Pointcloud::Ptr original_map_rpy;
        Pointcloud::Ptr tf_map_position;
        Pointcloud::Ptr tf_map_rpy;
        Pointcloud::Ptr gps_data;
        size_t sample_map_idx;

        Pointcloud::Ptr based_sample_map[5000];    //원점을 중심으로 하는 sample map
        PointMap::Ptr map_data; 

        Vector3f original_position_;
        Vector3f position_;
        
        Vector4f quaternion_;
        Vector3f original_rpy_;
        Vector3f rpy_;

        Vector2f gps_;
        
        Eigen::Affine3f odom_tf_affine;
        Eigen::Affine3f original_tf_affine;
        Eigen::Affine3f last_tf_affine;

        double mapSize;
        double objt_leafSize, lane_leafSize;
        double Epsilon, stepSize, resolution;
        int iteration;
        bool saveMapFlag;

        size_t last_optimized_node_idx;
        size_t last_crnt_idx;

    public:
        MapOptimizer():
        nh("~")
        {
            // ------- ROS basic ------- //
            sample_service = nh.advertiseService("save_sample_map",&MapOptimizer::save_sample_service,this);
            global_service = nh.advertiseService("save_global_map",&MapOptimizer::save_global_service,this);

            sub_laneMarker = nh.subscribe<msgs_Point>("/ese_lidar_projection/laneMarker_cloud",
                                                      10,&MapOptimizer::laneCallBack,this);
            sub_object = nh.subscribe<msgs_Point>("/ese_lidar_projection/object_cloud",
                                                      10,&MapOptimizer::objtCallBack,this);
            sub_odometry = nh.subscribe<msgs_Odom>("/odom",10,&MapOptimizer::odomCallBack,this);
            sub_fix = nh.subscribe<msgs_Nav>("/gps",10,&MapOptimizer::gpsCallBack,this);

            //pub_transformed = nh.advertise<msgs_Point>("test_cloud",1);
            pub_sample_map = nh.advertise<msgs_Point>("sample_map",1);
            pub_trajectory = nh.advertise<msgs_Point>("trajectory",1);
            pub_originMap  = nh.advertise<msgs_Point>("origin_map",1);

            // ------- for debuging ------- //
            pub_origin_trajectory = nh.advertise<msgs_Point>("original_trajectory",1);
            // ---------------------------- //

            pub_optimized_trajectory = nh.advertise<msgs_Point>("optimized_trajectory",1);
            pub_input_map = nh.advertise<msgs_Point>("input_map",1);
            pub_target_map = nh.advertise<msgs_Point>("target_map",1);

            pub_optimized_odom = nh.advertise<msgs_Odom>("final_odom",1);

            initializeValue();
        }
        ~MapOptimizer(){}

        void initializeValue()
        {
            nh.param("mapSize",mapSize,5.0);
            nh.param("search_size",search_size,10.0);
            nh.param("threshold_score",threshold_score,1.0);
            nh.param("objt_leafSize",objt_leafSize,1.0);
            nh.param("lane_leafSize",lane_leafSize,0.1);
            nh.param("Epsilon",Epsilon,0.01);
            nh.param("stepSize",stepSize,0.1);
            nh.param("resolution",resolution,1.0);
            nh.param("iteration",iteration,200);
            
            input_lane_cloud_.reset(new Pointcloud);
            transformed_lane_cloud_.reset(new Pointcloud);
            input_objt_cloud_.reset(new Pointcloud);
            transformed_objt_cloud_.reset(new Pointcloud);

            last_lane_sample_map.reset(new Pointcloud);
            crnt_lane_sample_map.reset(new Pointcloud);
            last_objt_sample_map.reset(new Pointcloud);
            crnt_objt_sample_map.reset(new Pointcloud);
            global_map.reset(new Pointcloud);

            original_map_position.reset(new Pointcloud);
            original_map_position->clear();
            original_map_rpy.reset(new Pointcloud);
            original_map_rpy->clear();
            
            tf_map_position.reset(new Pointcloud);
            tf_map_position->clear();
            tf_map_rpy.reset(new Pointcloud);
            tf_map_rpy->clear();

            gps_data.reset(new Pointcloud);
            gps_data->clear();

            map_data.reset(new PointMap);
            map_data->clear();

            Pointcloud::Ptr temp (new Pointcloud);
            temp->points.resize(1);
            temp->points[0].x = 0.0;
            temp->points[0].y = 0.0;
            temp->points[0].z = 0.0;
            temp->points[0].intensity = 0.0;
            *tf_map_position += *temp;
            *tf_map_rpy += *temp;
            *original_map_position += *temp;
            *original_map_rpy += *temp;

            sample_map_idx = 0;
            
            original_position_ << 0,0,0;
            original_rpy_ << 0,0,0;
            position_ << 0,0,0;
            quaternion_ << 0,0,0,0;
            rpy_ << 0,0,0;
            gps_ << 0,0;

            saveMapFlag = false;

            // -------   GTSAM   ------- //
            gtsam::Vector Vector6(6);
            Vector6 << 25*1e-2,25*1e-2,25*1e-4,25*1e-4,25*1e-4,25*1e-3;
            priorNoise = noiseModel::Diagonal::Sigmas(Vector6);
            odometryNoise = noiseModel::Diagonal::Sigmas(Vector6);

            if(graph.empty())
            {
                graph_idx = 0;

                Pose3 priorMean = Pose3(Rot3::ypr(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
                graph.add(PriorFactor<Pose3>(graph_idx, priorMean, priorNoise));
                init_value.insert(graph_idx, priorMean);
            }

            loopclosure_flag = true;
            isLoop = false;

            init_tf_matrix << 1,0,0,0,
                              0,1,0,0,
                              0,0,1,0,
                              0,0,0,0;

            ndt_tf_matrix  << 1,0,0,0,
                              0,1,0,0,
                              0,0,1,0,
                              0,0,0,1;

            ndt_tf_position << 0,0,0;
            ndt_tf_rpy << 0,0,0;

            last_optimized_node_idx = 0;
            last_crnt_idx = 0;

            last_tf_affine = pcl::getTransformation(0.0,0.0,0.0,0.0,0.0,0.0);
        }

        void copyLanePointCloud(const msgs_Point::ConstPtr& point_msg)
        {
            input_lane_cloud_.reset(new Pointcloud);
            input_lane_cloud_->clear();

            pcl::fromROSMsg(*point_msg, *input_lane_cloud_);
            return;
        }

        void copyObjtPointCloud(const msgs_Point::ConstPtr& point_msg)
        {
            input_objt_cloud_.reset(new Pointcloud);
            input_objt_cloud_->clear();

            pcl::fromROSMsg(*point_msg, *input_objt_cloud_);
            return;
        }

        void LanetransformMatrixUpdate()
        {
            original_tf_affine = pcl::getTransformation(original_position_(0),original_position_(1),original_position_(2),
                                                        original_rpy_(2),original_rpy_(1),original_rpy_(0));
        }

        void LanetransformPointCloud()
        {
            transformed_lane_cloud_.reset(new Pointcloud);
            transformed_lane_cloud_->points.resize(input_lane_cloud_->points.size());

            pcl::transformPointCloud(*input_lane_cloud_, *transformed_lane_cloud_, original_tf_affine);
        }

        void ObjttransformPointCloud()
        {
            transformed_objt_cloud_.reset(new Pointcloud);
            transformed_objt_cloud_->points.resize(input_objt_cloud_->points.size());

            pcl::transformPointCloud(*input_objt_cloud_, *transformed_objt_cloud_, original_tf_affine);
        }

        void saveLanePointCloud()
        {
            *crnt_lane_sample_map += *transformed_lane_cloud_;
            *last_lane_sample_map += *transformed_lane_cloud_;
        }

        void saveObjtPointCloud()
        {
            *crnt_objt_sample_map += *transformed_objt_cloud_;
            *last_objt_sample_map += *transformed_objt_cloud_;
        }

        void laneCallBack (const msgs_Point::ConstPtr& point_msg)
        {
            //1. copy input msgs
            //2. update transformation matrix
            //3. transform input cloud to transformed cloud
            //4. save transformed cloud to sample map 

            copyLanePointCloud(point_msg);
            LanetransformMatrixUpdate();
            LanetransformPointCloud();
            saveLanePointCloud();
        }

        void objtCallBack (const msgs_Point::ConstPtr& point_msg)
        {
            //1. copy input msgs
            //2. update transformation matrix
            //3. transform input cloud to transformed cloud
            //4. save transformed cloud to sample map 

            copyObjtPointCloud(point_msg);
            LanetransformMatrixUpdate();
            ObjttransformPointCloud();
            saveObjtPointCloud();
        }

        void gpsCallBack (const msgs_Nav::ConstPtr& gps_msg)
        {
            if(sample_map_idx == 0)
            {
                Pointcloud::Ptr temp_ (new Pointcloud);
                temp_->points.resize(1);
                temp_->points[0].x = gps_msg->latitude;
                temp_->points[0].y = gps_msg->longitude;
                temp_->points[0].z = 0.0;
                temp_->points[0].intensity = sample_map_idx;

                *gps_data += *temp_;
            }
            else
            {
                gps_(0) = gps_msg->latitude;
                gps_(1) = gps_msg->longitude;
            }

            return;
        }

        void odomCallBack (const msgs_Odom::ConstPtr& odom_msg)
        {
            //1. save odometry data
            //2. calculate distance from last node 
            //3. if distance bigger than save the sample map and update global map

            saveOdometry(odom_msg);
            tfPositionRpyUpdate();
            saveMapFlag = calculateDistance();
            if(saveMapFlag)
            {
                //tfPositionRpyUpdate();
                updateSampleMap();
                publishMap();
            }
        }

        void saveOdometry (const msgs_Odom::ConstPtr& odom_msg)
        {
            original_position_(0) = odom_msg->pose.pose.position.x;
            original_position_(1) = odom_msg->pose.pose.position.y;
            original_position_(2) = odom_msg->pose.pose.position.z;

            double roll_, pitch_, yaw_;
            geometry_msgs::Quaternion geoQuat = odom_msg->pose.pose.orientation;
            tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll_, pitch_, yaw_);

            original_rpy_(0) = yaw_;
            original_rpy_(1) = pitch_;
            original_rpy_(2) = roll_;


            return;
        }

        void publishOdom()
        {
            tf::Matrix3x3 odom_mat;
            tf::Quaternion odom_quat;
            odom_mat.setEulerYPR(rpy_(0),rpy_(1),rpy_(2));
            odom_mat.getRotation(odom_quat);

            geometry_msgs::TransformStamped odom_trans;
            odom_trans.header.stamp = ros::Time::now();
            odom_trans.header.frame_id = "odom";
            odom_trans.child_frame_id = "base_footprint";

            odom_trans.transform.translation.x = position_(0);
            odom_trans.transform.translation.y = position_(1);
            odom_trans.transform.translation.z = position_(2);
            odom_trans.transform.rotation.x = odom_quat.getX();
            odom_trans.transform.rotation.y = odom_quat.getY();
            odom_trans.transform.rotation.z = odom_quat.getZ();
            odom_trans.transform.rotation.w = odom_quat.getW();

            odom_broadcaster.sendTransform(odom_trans);

            msgs_Odom odom_msg;
            odom_msg.header.frame_id = "odom";
            odom_msg.header.stamp = ros::Time::now();

            odom_msg.pose.pose.position.x = position_(0);
            odom_msg.pose.pose.position.y = position_(1);
            odom_msg.pose.pose.position.z = position_(2);
            odom_msg.pose.pose.orientation.x = odom_quat.getX();
            odom_msg.pose.pose.orientation.y = odom_quat.getY();
            odom_msg.pose.pose.orientation.z = odom_quat.getZ();
            odom_msg.pose.pose.orientation.w = odom_quat.getW();

            pub_optimized_odom.publish(odom_msg);

            return;
        }

        void tfPositionRpyUpdate()
        {
            position_(0) = tf_map_position->points[last_optimized_node_idx].x + (original_position_(0) - original_map_position->points[last_optimized_node_idx].x);
            position_(1) = tf_map_position->points[last_optimized_node_idx].y + (original_position_(1) - original_map_position->points[last_optimized_node_idx].y);
            position_(2) = tf_map_position->points[last_optimized_node_idx].z + (original_position_(2) - original_map_position->points[last_optimized_node_idx].z);

            rpy_(0) = tf_map_rpy->points[last_optimized_node_idx].x + (original_rpy_(0) - original_map_rpy->points[last_optimized_node_idx].x);
            rpy_(1) = tf_map_rpy->points[last_optimized_node_idx].y + (original_rpy_(1) - original_map_rpy->points[last_optimized_node_idx].y);
            rpy_(2) = tf_map_rpy->points[last_optimized_node_idx].z + (original_rpy_(2) - original_map_rpy->points[last_optimized_node_idx].z);

            publishOdom();
        }

        bool calculateDistance()
        {
            float diff_x, diff_y, diff_z, dist;
            diff_x = original_position_(0) - original_map_position->points[sample_map_idx].x;
            diff_y = original_position_(1) - original_map_position->points[sample_map_idx].y;
            diff_z = original_position_(2) - original_map_position->points[sample_map_idx].z;

            dist = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);

            if(dist >= mapSize)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        void voxelfiltering ()
        {
            Pointcloud::Ptr temp (new Pointcloud);

            pcl::VoxelGrid<PointI> voxel;
            voxel.setLeafSize (objt_leafSize, objt_leafSize, objt_leafSize);
            voxel.setInputCloud (last_objt_sample_map);
            voxel.filter (*temp);

            last_objt_sample_map.reset(new Pointcloud);
            *last_objt_sample_map = *temp;

            voxel.setInputCloud (last_lane_sample_map);
            voxel.setLeafSize (lane_leafSize, lane_leafSize, lane_leafSize);
            voxel.filter (*temp);

            last_lane_sample_map.reset(new Pointcloud);
            *last_lane_sample_map = *temp;

        }

        Eigen::Affine3f getTFMatrixToOrigin()
        {
            //Matrix4f mat_to_origin;
            Eigen::Affine3f mat_to_origin;
            double roll_, pitch_ , yaw_;
            yaw_ = -1*original_map_rpy->points[sample_map_idx].x;
            pitch_ = -1*original_map_rpy->points[sample_map_idx].y;
            roll_ = -1*original_map_rpy->points[sample_map_idx].z;

            mat_to_origin = pcl::getTransformation(0.0, 0.0, 0.0, roll_, pitch_, yaw_);

            return mat_to_origin;
        }

        void updateSampleMap()
        {
            //update original map position and rpy ----- odom base
            Pointcloud::Ptr temp (new Pointcloud);
            temp->points.resize(1);
            temp->points[0].x = original_position_(0);
            temp->points[0].y = original_position_(1);
            temp->points[0].z = original_position_(2);
            temp->points[0].intensity = sample_map_idx;
            *original_map_position += *temp;

            temp->points[0].x = original_rpy_(0);
            temp->points[0].y = original_rpy_(1);
            temp->points[0].z = original_rpy_(2);
            *original_map_rpy += *temp;

            // update sample map poistion and rpy
            temp->points[0].x = position_(0); //x__;
            temp->points[0].y = position_(1); //y__;
            temp->points[0].z = position_(2); //z__;
            *tf_map_position += *temp;

            //yaw pitch roll
            temp->points[0].x = rpy_(0); //yaw_;
            temp->points[0].y = rpy_(1); //pitch_;
            temp->points[0].z = rpy_(2); //roll_;
            *tf_map_rpy += *temp;

            //gps data
            temp->points[0].x = gps_(0);
            temp->points[0].y = gps_(1);
            temp->points[0].z = 0.0;
            *gps_data += *temp;

            //update sample map
            voxelfiltering();

            //based_sample_map update
            Matrix4f mat_to_origin;
            Pointcloud::Ptr temp_to_origin (new Pointcloud);
            //temp_to_origin->points.resize(last_lane_sample_map->points.size() + last_objt_sample_map->points.size());
            temp_to_origin->clear();
            *temp_to_origin += *last_lane_sample_map;
            *temp_to_origin += *last_objt_sample_map;

            double x_, y_, z_;
            x_ = -1*original_map_position->points[sample_map_idx].x;
            y_ = -1*original_map_position->points[sample_map_idx].y;
            z_ = -1*original_map_position->points[sample_map_idx].z;
            mat_to_origin << 1,0,0,x_,
                             0,1,0,y_,
                             0,0,1,z_,
                             0,0,0,1;
            pcl::transformPointCloud(*temp_to_origin,*temp_to_origin,mat_to_origin);
            Eigen::Affine3f mat_to_origin_2 = getTFMatrixToOrigin();
            pcl::transformPointCloud(*temp_to_origin,*temp_to_origin,mat_to_origin_2);

            based_sample_map[sample_map_idx].reset(new Pointcloud);
            based_sample_map[sample_map_idx]->clear();
            based_sample_map[sample_map_idx]->points.resize(last_lane_sample_map->points.size() + last_objt_sample_map->points.size());
            pcl::copyPointCloud(*temp_to_origin,*based_sample_map[sample_map_idx]);
            sample_map_idx++;

            last_lane_sample_map.reset(new Pointcloud);
            last_lane_sample_map->clear();
            *last_lane_sample_map = *crnt_lane_sample_map;
            crnt_lane_sample_map.reset(new Pointcloud);
            crnt_lane_sample_map->clear();

            last_objt_sample_map.reset(new Pointcloud);
            last_objt_sample_map->clear();
            *last_objt_sample_map = *crnt_objt_sample_map;
            crnt_objt_sample_map.reset(new Pointcloud);
            crnt_objt_sample_map->clear();

            saveMapFlag = false;

            // -------- GTSAM -------- //
            graph_idx++;
            gtsam::Pose3 poseFrom = Pose3(Rot3::ypr(tf_map_rpy->points[sample_map_idx-1].x,-1*tf_map_rpy->points[sample_map_idx-1].y,-1*tf_map_rpy->points[sample_map_idx-1].z),
                                            Point3(tf_map_position->points[sample_map_idx-1].x,tf_map_position->points[sample_map_idx-1].y,tf_map_position->points[sample_map_idx-1].z));
            gtsam::Pose3 poseTo   = Pose3(Rot3::ypr(tf_map_rpy->points[sample_map_idx].x,-1*tf_map_rpy->points[sample_map_idx].y,-1*tf_map_rpy->points[sample_map_idx].z),
                                            Point3(tf_map_position->points[sample_map_idx].x,tf_map_position->points[sample_map_idx].y,tf_map_position->points[sample_map_idx].z));
            graph.add(BetweenFactor<Pose3>(graph_idx-1,graph_idx,poseFrom.between(poseTo),odometryNoise));
            init_value.insert(graph_idx, poseTo);

            return;
        }

        void publishMap()
        {
            msgs_Point temp_cloud;
            ros::Time time_ = ros::Time::now();

            pcl::toROSMsg(*tf_map_position, temp_cloud);
            temp_cloud.header.frame_id = "odom";
            temp_cloud.header.stamp = time_;
            pub_trajectory.publish(temp_cloud);

            pcl::toROSMsg(*original_map_position, temp_cloud);
            temp_cloud.header.frame_id = "odom";
            temp_cloud.header.stamp = time_;
            pub_origin_trajectory.publish(temp_cloud);

            if(sample_map_idx != 0)
            {
                Eigen::Affine3f mat_;
                mat_ = pcl::getTransformation(tf_map_position->points[sample_map_idx-1].x, tf_map_position->points[sample_map_idx-1].y, tf_map_position->points[sample_map_idx-1].z,
                                              tf_map_rpy->points[sample_map_idx-1].z, tf_map_rpy->points[sample_map_idx-1].y, tf_map_rpy->points[sample_map_idx-1].x);
                Pointcloud::Ptr temp_ (new Pointcloud);
                temp_->points.resize(based_sample_map[sample_map_idx-1]->points.size());

                pcl::transformPointCloud(*based_sample_map[sample_map_idx-1], *temp_, mat_);

                pcl::toROSMsg(*temp_,temp_cloud);
                temp_cloud.header.frame_id = "odom";
                temp_cloud.header.stamp = time_;
                pub_sample_map.publish(temp_cloud);

                pcl::toROSMsg(*based_sample_map[sample_map_idx-1],temp_cloud);
                temp_cloud.header.frame_id = "odom";
                temp_cloud.header.stamp = time_;
                pub_originMap.publish(temp_cloud);
            }

            if(isLoop)
            {
                Eigen::Affine3f mat_;
                mat_ = pcl::getTransformation(tf_map_position->points[crnt_idx].x, tf_map_position->points[crnt_idx].y, tf_map_position->points[crnt_idx].z,
                                              tf_map_rpy->points[crnt_idx].z, tf_map_rpy->points[crnt_idx].y, tf_map_rpy->points[crnt_idx].x);
                Pointcloud::Ptr temp_ (new Pointcloud);
                temp_->points.resize(based_sample_map[crnt_idx]->points.size());

                pcl::transformPointCloud(*based_sample_map[crnt_idx], *temp_, mat_);

                pcl::toROSMsg(*temp_,temp_cloud);
                temp_cloud.header.frame_id = "odom";
                temp_cloud.header.stamp = time_;
                pub_input_map.publish(temp_cloud);

                mat_ = pcl::getTransformation(tf_map_position->points[nearest_node_idx].x, tf_map_position->points[nearest_node_idx].y, tf_map_position->points[nearest_node_idx].z,
                                              tf_map_rpy->points[nearest_node_idx].z, tf_map_rpy->points[nearest_node_idx].y, tf_map_rpy->points[nearest_node_idx].x);
                temp_->points.resize(based_sample_map[nearest_node_idx]->points.size());
                pcl::transformPointCloud(*based_sample_map[nearest_node_idx], *temp_, mat_);

                pcl::toROSMsg(*temp_,temp_cloud);
                temp_cloud.header.frame_id = "odom";
                temp_cloud.header.stamp = time_;
                pub_target_map.publish(temp_cloud);

                isLoop = false;
            }


        }

        void loopClosureThread()
        {
            if(loopclosure_flag == false)
                return;
            
            ros::Rate rate_loop(1);
            while(ros::ok())
            {
                if(findNearestNode())
                {
                    if(calculateNDTScore())
                    {
                        DoLoopClosure();
                    }
                }

                rate_loop.sleep();
            }
        }

        bool findNearestNode()
        {
            //the number of sample map is smaller than 5, don't doing loop closure
            if(sample_map_idx < 5)
                return false;

            //if the number of sample map is bigger than 5 find nearest node 
            double last_dist = search_size + 1;
            crnt_idx = sample_map_idx - 1;

            pcl::KdTreeFLANN<PointI> kdtree;
            kdtree.setInputCloud(tf_map_position);
            
            PointI input_node;
            input_node.x = tf_map_position->points[crnt_idx].x;
            input_node.y = tf_map_position->points[crnt_idx].y;
            input_node.z = tf_map_position->points[crnt_idx].z;
            input_node.intensity = tf_map_position->points[crnt_idx].intensity;

            std::vector<int> node_idx_radius_search;
            std::vector<float> node_radius_sqared_dist;

            kdtree.radiusSearch(input_node, search_size, node_idx_radius_search, node_radius_sqared_dist);
            for(size_t i=0; i<node_idx_radius_search.size(); i++)
            {
                double dist = sqrt(node_radius_sqared_dist[i]);
                if(last_dist > dist && node_idx_radius_search[i] < (crnt_idx-4))
                {
                    last_dist = dist;
                    nearest_node_idx = node_idx_radius_search[i];
                }
            }

            if(nearest_node_idx == crnt_idx -1 || nearest_node_idx == crnt_idx -2 || 
               nearest_node_idx == crnt_idx -3 || nearest_node_idx == crnt_idx -4 ||
               last_dist > search_size)
                return false;
            else
            {
                std::cout << "current_index : " << crnt_idx << "\nnearest_node_index : " << nearest_node_idx << std::endl;
                return true;
            }
        }

        bool calculateNDTScore()
        {
            Pointcloud::Ptr input_ (new Pointcloud);
            Pointcloud::Ptr target_ (new Pointcloud);
            Pointcloud::Ptr aligned_ (new Pointcloud);

            Eigen::Affine3f input_mat_,target_mat_;
            target_mat_ = pcl::getTransformation(tf_map_position->points[nearest_node_idx].x, tf_map_position->points[nearest_node_idx].y, tf_map_position->points[nearest_node_idx].z,
                                                 tf_map_rpy->points[nearest_node_idx].z, tf_map_rpy->points[nearest_node_idx].y, tf_map_rpy->points[nearest_node_idx].x);
            input_mat_ = pcl::getTransformation(tf_map_position->points[crnt_idx].x, tf_map_position->points[crnt_idx].y, tf_map_position->points[crnt_idx].z,
                                                tf_map_rpy->points[crnt_idx].z, tf_map_rpy->points[crnt_idx].y, tf_map_rpy->points[crnt_idx].x);
            
            input_->points.resize(based_sample_map[crnt_idx]->points.size());
            target_->points.resize(based_sample_map[nearest_node_idx]->points.size());

            pcl::transformPointCloud(*based_sample_map[nearest_node_idx], *target_, target_mat_);
            pcl::transformPointCloud(*based_sample_map[crnt_idx], *input_, input_mat_);
            
            pcl::NormalDistributionsTransform<PointI, PointI> ndt;

            ndt.setTransformationEpsilon(Epsilon);
            ndt.setStepSize (stepSize);
            ndt.setResolution (resolution);
            ndt.setMaximumIterations (iteration);
            
            ndt.setInputSource (input_);
            ndt.setInputTarget (target_);
            ndt.align(*aligned_, init_tf_matrix);

            ndt_align_score = ndt.getFitnessScore();
            
            if(ndt_align_score < threshold_score)
            {
                ndt_tf_matrix = ndt.getFinalTransformation();
                ndt_tf_affine = ndt.getFinalTransformation();

                // ------- for test ------- //
                std::cout << crnt_idx << " " << nearest_node_idx << " "
                          << ndt_align_score << std::endl;
                //std::cout << ndt_tf_matrix << std::endl;
                // ------------------------ //

                return true;
            }
            else if(ndt_align_score > threshold_score || ndt.hasConverged() == false)
                return false;
        }

        //이놈을 어떻게 할것이냐...... 
        void DoLoopClosure()
        {
            //ndt_tf_matrix에서 rpy xyz 추출 --> 왜냐하면 1.을 위하여 
            //1. loopclosure 발생한 node의 맞는 위치 찾기
            //2. 변환한 node와 기준 node에 대한 factor추가
            //3. init_value 업데이트
            //4. tf_map_position, tf_map_rpy 업데이트
            //5. sample_map_ 업데이트

            last_crnt_idx = crnt_idx;

            float x_, y_, z_, roll_, pitch_, yaw_;
            pcl::getTranslationAndEulerAngles(ndt_tf_affine, x_, y_, z_, roll_, pitch_, yaw_);
            Vector4f temp_correct_node;
            Vector3f temp_correct_node_rpy;
            Vector4f temp_wrong_node;
            temp_wrong_node << tf_map_position->points[last_crnt_idx].x, 
                               tf_map_position->points[last_crnt_idx].y,
                               tf_map_position->points[last_crnt_idx].z,
                               1;
            temp_correct_node = ndt_tf_matrix * temp_wrong_node;
            temp_correct_node_rpy << tf_map_rpy->points[last_crnt_idx].x + yaw_,
                                     tf_map_rpy->points[last_crnt_idx].y + pitch_,
                                     tf_map_rpy->points[last_crnt_idx].z + roll_;

            gtsam::Pose3 pose_From = Pose3(Rot3::ypr(temp_correct_node_rpy(0),temp_correct_node_rpy(1),temp_correct_node_rpy(2)),
                                            Point3(temp_correct_node(0),temp_correct_node(1),temp_correct_node(2)));
            gtsam::Pose3 pose_To = Pose3(Rot3::ypr(tf_map_rpy->points[nearest_node_idx].x,tf_map_rpy->points[nearest_node_idx].y,tf_map_rpy->points[nearest_node_idx].z),
                                            Point3(tf_map_position->points[nearest_node_idx].x,tf_map_position->points[nearest_node_idx].y,tf_map_position->points[nearest_node_idx].z));

            float ndt_noise  = (float)ndt_align_score;
            gtsam::Vector Vector6(6);
            Vector6 << ndt_noise, ndt_noise, ndt_noise, ndt_noise, ndt_noise, ndt_noise; 
            graph.add(BetweenFactor<Pose3>(crnt_idx,nearest_node_idx,pose_From.between(pose_To),ndtNoise));

            LevenbergMarquardtOptimizer optimizer(graph, init_value);
            optimized_value = optimizer.optimize();

            //init_value.update(optimized_value);

            size_t opt_cnt = optimized_value.size();
            last_tf_affine = ndt_tf_affine*last_tf_affine;

            optimizingSampleMap();

            return;
        }

        void optimizingSampleMap()
        {            
            for(size_t i=0; i<optimized_value.size(); i++)
            {
                tf_map_position->points[i].x = optimized_value.at<Pose3>(i).translation().x();
                tf_map_position->points[i].y = optimized_value.at<Pose3>(i).translation().y();
                tf_map_position->points[i].z = optimized_value.at<Pose3>(i).translation().z();
                tf_map_rpy->points[i].x = optimized_value.at<Pose3>(i).rotation().yaw();
                tf_map_rpy->points[i].y = optimized_value.at<Pose3>(i).rotation().pitch();
                tf_map_rpy->points[i].x = optimized_value.at<Pose3>(i).rotation().roll();
            }
            last_optimized_node_idx = optimized_value.size() - 1;

            if(sample_map_idx > last_optimized_node_idx)
            {
                for(size_t i=last_optimized_node_idx+1;i<sample_map_idx+1;i++)
                {
                    tf_map_position->points[i].x = tf_map_position->points[last_optimized_node_idx].x + (original_map_position->points[i].x - original_map_position->points[last_optimized_node_idx].x);
                    tf_map_position->points[i].y = tf_map_position->points[last_optimized_node_idx].y + (original_map_position->points[i].y - original_map_position->points[last_optimized_node_idx].y);
                    tf_map_position->points[i].z = tf_map_position->points[last_optimized_node_idx].z + (original_map_position->points[i].z - original_map_position->points[last_optimized_node_idx].z);
                    tf_map_rpy->points[i].x = tf_map_rpy->points[last_optimized_node_idx].x + (original_map_rpy->points[i].x - original_map_rpy->points[last_optimized_node_idx].x);
                    tf_map_rpy->points[i].y = tf_map_rpy->points[last_optimized_node_idx].y + (original_map_rpy->points[i].y - original_map_rpy->points[last_optimized_node_idx].y);
                    tf_map_rpy->points[i].z = tf_map_rpy->points[last_optimized_node_idx].z + (original_map_rpy->points[i].z - original_map_rpy->points[last_optimized_node_idx].z);
                }
            }
            isLoop = true;
        }

        bool save_sample_service (ese_slam::save_sample_mapRequest  &req,
                                  ese_slam::save_sample_mapResponse &res)
        {
            for(size_t i=0;i<sample_map_idx;i++)
            {
                savePCD(i);
            }

            return true;
        }

        bool save_global_service (ese_slam::save_global_mapRequest  &req,
                                  ese_slam::save_global_mapResponse &res)
        {
            std::string file_adrs = "/home/mbek/Desktop/bag_file/global_map.pcd";
			std::stringstream ss;
			ss << file_adrs;

            size_t total_size = 0;

            Pointcloud::Ptr temp_ (new Pointcloud);
            Pointcloud::Ptr global_ (new Pointcloud);
            global_->clear();

            for(size_t i=0; i<last_optimized_node_idx; i++)
            {
                temp_->resize(based_sample_map[i]->points.size());

                Eigen::Affine3f mat_;
                double x_, y_, z_, roll_, pitch_ , yaw_;
                x_ = tf_map_position->points[i].x;
                y_ = tf_map_position->points[i].y;
                z_ = tf_map_position->points[i].z;
                yaw_ = tf_map_rpy->points[i].x;
                pitch_ = tf_map_rpy->points[i].y;
                roll_ = tf_map_rpy->points[i].z;

                mat_ = pcl::getTransformation(x_, y_, z_, roll_, pitch_, yaw_);

                pcl::transformPointCloud(*based_sample_map[i], *temp_, mat_);

                //temp_->height = 1;
                total_size += based_sample_map[i]->points.size();
                *global_ += *temp_;
                pcl::io::savePCDFileASCII (ss.str(), *temp_);
                std::cout << "Add " << i << " Sample Map!! " << std::endl;
            }

            global_->height = 1;
            global_->width = total_size;

            pcl::io::savePCDFileASCII (ss.str(), *global_);
            std::cout << "Save Global Map!! " << std::endl;

            return true;
        }

        // --------- for test --------- //
        void savePCD(size_t i)
        {
            std::string file_adrs = "/home/mbek/Desktop/bag_file/saved_samplemap/";
			std::stringstream ss;
			ss << file_adrs << i << ".pcd";

            Pointcloud::Ptr temp_ (new Pointcloud);
            temp_->resize(based_sample_map[i]->points.size());

            Eigen::Affine3f mat_;
            double x_, y_, z_, roll_, pitch_ , yaw_;
            x_ = original_map_position->points[i].x;
            y_ = original_map_position->points[i].y;
            z_ = original_map_position->points[i].z;
            yaw_ = original_map_rpy->points[i].x;
            pitch_ = original_map_rpy->points[i].y;
            roll_ = original_map_rpy->points[i].z;

            mat_ = pcl::getTransformation(x_, y_, z_, roll_, pitch_, yaw_);

            pcl::transformPointCloud(*based_sample_map[i], *temp_, mat_);

            temp_->height = 1;
            temp_->width = based_sample_map[i]->points.size();
            pcl::io::savePCDFileASCII (ss.str(), *temp_);
            std::cout << "Save " << i << ".pcd!! " << std::endl;

            return;
        }
        // ----------------------------- //
};

int main (int argc, char** argv)
{
    ros::init(argc, argv, "mapOptimizer");

    MapOptimizer MO;

    std::thread loopClosureThread(&MapOptimizer::loopClosureThread, &MO);

    ROS_INFO("\033[1;32m---->\033[0m ESE Map Optimizer Started.");

    ros::spin();

    return 0;
}