#include <ese_slam/slam_util.h>
#include <ese_slam/UTM.h>

#define LANE 1
#define OBJT 2

class EseLocalization
{
    private:
        // ------- ROS Basic ------ //
        ros::NodeHandle nh;

        ros::Subscriber sub_laneMarker;
        ros::Subscriber sub_object;
        ros::Subscriber sub_odometry;
        ros::Subscriber sub_fix;

        ros::Publisher pub_mapTrajectory;
        ros::Publisher pub_mapOrigin;
        ros::Publisher pub_mapPoint;

        ros::Publisher pub_test;

        // ------ Map Data ------- //
        PointMap::Ptr map_trajectory;
        Pointcloud::Ptr map_position;
        Pointcloud::Ptr map_point;

        // ------ LiDAR Data ----- //
        Pointcloud::Ptr input_lane_cloud;
        Pointcloud::Ptr input_objt_cloud;
        Pointcloud::Ptr transformed_lane_cloud;
        Pointcloud::Ptr transformed_objt_cloud;

        Pointcloud::Ptr crnt_lane_sample_map;
        Pointcloud::Ptr last_lane_sample_map;
        Pointcloud::Ptr crnt_objt_sample_map;
        Pointcloud::Ptr last_objt_sample_map; 

        Pointcloud::Ptr crnt_map_cloud;
        Pointcloud::Ptr last_map_cloud;
        Pointcloud::Ptr crnt_based_map_cloud;
        Pointcloud::Ptr last_based_map_cloud;

        // ------ Odom Data ------ //
        Vector3f original_position_;
        Vector3f original_rpy_;
        Vector3f tf_position_;
        Vector3f tf_rpy_;

        Vector3f last_position_;
        Vector3f last_rpy_;
        Vector3f crnt_position_;
        Vector3f crnt_rpy_;

        bool saveMapFlag;

        // ------ Gps Data ------- //
        bool isGps;

        double f_utm_x, f_utm_y, f_utm_z;

        Vector3f gps_;
        Vector3f last_gps_;
        Vector3f crnt_gps_;

        // ------ TF Matrix ------ //
        Eigen::Affine3f original_tf_affine;

        // ------ ROS Param ------ //
        double mapSize, targetMapSize;
        double objt_leafSize, lane_leafSize;
        double search_size;

        // ------ Loop Closure --- //
        bool loopclosure_flag;
        size_t nearest_node_idx;
        double Epsilon, stepSize, iteration, resolution;
        Matrix4f init_tf_matrix;

        Pointcloud::Ptr temp_target;

    public:
        EseLocalization ():
        nh("~")
        {
            sub_laneMarker = nh.subscribe<msgs_Point>("/ese_lidar_projection/laneMarker_cloud",
                                                        10,&EseLocalization::laneCallBack,this);
            sub_object = nh.subscribe<msgs_Point>("/ese_lidar_projection/object_cloud",
                                                        10,&EseLocalization::objtCallBack,this);
            sub_odometry = nh.subscribe<msgs_Odom>("/odom",10,&EseLocalization::odomCallBack,this);
            sub_fix = nh.subscribe<msgs_Nav>("/fix",10,&EseLocalization::gpsCallBack,this);

            pub_mapTrajectory = nh.advertise<msgs_Point>("map_trajectory",1);
            pub_mapOrigin = nh.advertise<msgs_Point>("map_origin",1);
            pub_mapPoint = nh.advertise<msgs_Point>("sample_map",1);
            pub_test = nh.advertise<msgs_Point>("last_crnt",1);

            initializeValue();
            loadMapData();
        }
        ~EseLocalization(){}

        void initializeValue ()
        {
            nh.param("mapSize", mapSize, 15.0);
            nh.param("targetMapSize",targetMapSize,65.0);
            nh.param("search_size",search_size,10.0);
            nh.param("objt_leafSize", objt_leafSize, 1.0);
            nh.param("lane_leafSize", lane_leafSize, 0.1);
            nh.param("Epsilon",Epsilon,0.01);
            nh.param("stepSize",stepSize, 2.0);
            nh.param("resolution",resolution, 1.0);
            nh.param("iteration", iteration, 50.0);

            map_trajectory.reset(new PointMap);
            map_trajectory->clear();
            map_position.reset(new Pointcloud);
            map_position->clear();
            map_point.reset(new Pointcloud);
            map_point->clear();

            input_lane_cloud.reset(new Pointcloud);
            input_lane_cloud->clear();
            input_objt_cloud.reset(new Pointcloud);
            input_objt_cloud->clear();

            transformed_lane_cloud.reset(new Pointcloud);
            transformed_lane_cloud->clear();
            transformed_objt_cloud.reset(new Pointcloud);
            transformed_objt_cloud->clear();

            crnt_lane_sample_map.reset(new Pointcloud);
            crnt_lane_sample_map->clear();
            last_lane_sample_map.reset(new Pointcloud);
            last_lane_sample_map->clear();
            crnt_objt_sample_map.reset(new Pointcloud);
            crnt_objt_sample_map->clear();
            last_objt_sample_map.reset(new Pointcloud);
            last_objt_sample_map->clear();

            crnt_map_cloud.reset(new Pointcloud);
            crnt_map_cloud->clear();
            last_map_cloud.reset(new Pointcloud);
            last_map_cloud->clear();
            crnt_based_map_cloud.reset(new Pointcloud);
            crnt_based_map_cloud->clear();
            last_based_map_cloud.reset(new Pointcloud);
            last_based_map_cloud->clear();

            original_position_ << 0,0,0;
            original_rpy_ << 0,0,0;
            tf_position_ << 0,0,0;
            tf_rpy_ << 0,0,0;
            last_position_ << 0,0,0;
            last_rpy_ << 0,0,0;
            crnt_position_ << 0,0,0;
            crnt_rpy_ << 0,0,0;

            saveMapFlag = false;
            loopclosure_flag = true;

            nearest_node_idx = 0;

            init_tf_matrix << 1,0,0,0,
                              0,1,0,0,
                              0,0,1,0,
                              0,0,0,1;

            return;
        }

        void loadMapData ()
        {
            loadTrajectory();
            loadPointCloud();
        }

        void loadTrajectory ()
        {
            std::string file_adrs = "/home/beoms/Desktop/bag_file/map_trajectory.pcd";
            std::stringstream ss;
            ss << file_adrs;

            if(pcl::io::loadPCDFile<MapInfo> (ss.str(), *map_trajectory) == -1)
            {
                PCL_ERROR("Could not load file of map_trajectory.pcd!");
            }
            else
            {
                ROS_INFO("\033[1;32m---->\033[0m Success to Load Map Trajectory!");
            }

            // ------- for debuging ------- //
            Pointcloud::Ptr temp_cloud (new Pointcloud);
            ros::Time time_ = ros::Time::now();

            temp_cloud->points.resize(map_trajectory->points.size());
            map_position->points.resize(map_trajectory->points.size());
            for(size_t i=0;i<temp_cloud->points.size(); i++)
            {
                temp_cloud->points[i].x = map_trajectory->points[i].x;
                temp_cloud->points[i].y = map_trajectory->points[i].y;
                temp_cloud->points[i].z = map_trajectory->points[i].z;
                temp_cloud->points[i].intensity = i;

                map_position->points[i].x = map_trajectory->points[i].x;
                map_position->points[i].y = map_trajectory->points[i].y;
                map_position->points[i].z = map_trajectory->points[i].z;
                map_position->points[i].intensity = map_trajectory->points[i].intensity;
            }

            msgs_Point temp_;
            pcl::toROSMsg(*temp_cloud, temp_);
            temp_.header.frame_id = "odom";
            temp_.header.stamp = time_;
            pub_mapTrajectory.publish(temp_);
        }

        void loadPointCloud ()
        {
            std::string file_adrs = "/home/beoms/Desktop/bag_file/global_map.pcd";
            std::stringstream ss;
            ss << file_adrs;
            
            if(pcl::io::loadPCDFile<PointI> (ss.str(), *map_point) == -1)
            {
                PCL_ERROR("Could not load file of map_trajectory.pcd!");
            } 
            else
            {
                ROS_INFO("\033[1;32m---->\033[0m Success to Load Map Pointcloud!");
            }
        }

        void laneCallBack (const msgs_Point::ConstPtr& lane_msgs)
        {
            copyPointCloud(lane_msgs, LANE);
            transformMatrixUpdate();
            transformPointCloud(LANE);
            savePointCloud(LANE);
        }

        void objtCallBack (const msgs_Point::ConstPtr& objt_msgs)
        {
            copyPointCloud(objt_msgs, OBJT);
            transformMatrixUpdate();
            transformPointCloud(OBJT);
            savePointCloud(OBJT);
        }

        void copyPointCloud(const msgs_Point::ConstPtr& point_msg, int i)
        {
            if(i == LANE)
            {
                input_lane_cloud.reset(new Pointcloud);
                input_lane_cloud->clear();

                pcl::fromROSMsg(*point_msg, *input_lane_cloud);
            }
            else if(i == OBJT)
            {
                input_objt_cloud.reset(new Pointcloud);
                input_objt_cloud->clear();

                pcl::fromROSMsg(*point_msg, *input_objt_cloud);
            }

            return;
        }

        void transformMatrixUpdate ()
        {
            original_tf_affine = pcl::getTransformation(original_position_(0),original_position_(1),original_position_(2),
                                                        original_rpy_(2),original_rpy_(1),original_rpy_(0));
        }
        void transformPointCloud (int i)
        {
            if(i == LANE)
            {
                transformed_lane_cloud.reset(new Pointcloud);
                transformed_lane_cloud->clear();

                pcl::transformPointCloud(*input_lane_cloud, *transformed_lane_cloud, original_tf_affine);
            }
            else if(i == OBJT)
            {
                transformed_objt_cloud.reset(new Pointcloud);
                transformed_objt_cloud->clear();

                pcl::transformPointCloud(*input_objt_cloud, *transformed_objt_cloud, original_tf_affine);
            }

            return;
        }

        void savePointCloud (int i)
        {
            if(i == LANE)
            {
                *crnt_lane_sample_map += *transformed_lane_cloud;
                *last_lane_sample_map += *transformed_lane_cloud;
            }
            else if(i == OBJT)
            {
                *crnt_objt_sample_map += *transformed_objt_cloud;
                *last_objt_sample_map += *transformed_objt_cloud;
            }

            return;
        }

        void odomCallBack (const msgs_Odom::ConstPtr& odom_msg)
        {
            saveOdometry(odom_msg);
            // 추후 업데이트
            //tfPositionRpyUpdate();
            saveMapFlag = calculateDistance();
            if(saveMapFlag)
            {
                updateSampleMap();
            }
        }

        void saveOdometry (const msgs_Odom::ConstPtr& odom_msg)
        {
            original_position_(0) = odom_msg->pose.pose.position.x;
            original_position_(1) = odom_msg->pose.pose.position.y;
            original_position_(2) = odom_msg->pose.pose.position.z;

            double r_,p_,y_;
            geometry_msgs::Quaternion geoQuat = odom_msg->pose.pose.orientation;
            tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(r_, p_, y_);

            original_rpy_(0) = y_;
            original_rpy_(1) = p_;
            original_rpy_(2) = r_;

            return;
        }

        void tfPositionRpyUpdate ()
        {
        }

        bool calculateDistance ()
        {
            double d_x, d_y, d_z, dist;
            d_x = original_position_(0) - crnt_position_(0);
            d_y = original_position_(1) - crnt_position_(1);
            d_z = original_position_(2) - crnt_position_(2);

            dist = sqrt(d_x*d_x + d_y*d_y + d_z*d_z);

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

        Eigen::Affine3f getTFMatrixToOrigin ()
        {
            Eigen::Affine3f mat_to_origin;
            double r_, p_ , y_;
            y_ = -1 * last_rpy_(0);
            p_ = -1 * last_rpy_(1);
            r_ = -1 * last_rpy_(2);

            mat_to_origin = pcl::getTransformation(0.0, 0.0, 0.0, r_, p_, y_);

            return mat_to_origin;
        }

        void updateSampleMap ()
        {
            last_position_(0) = crnt_position_(0);
            last_position_(1) = crnt_position_(1);
            last_position_(2) = crnt_position_(2);
            last_rpy_(0) = crnt_rpy_(0);
            last_rpy_(1) = crnt_rpy_(1);
            last_rpy_(2) = crnt_rpy_(2);
            last_gps_(0) = crnt_gps_(0);
            last_gps_(1) = crnt_gps_(1);
            last_gps_(2) = crnt_gps_(2);

            crnt_position_(0) = original_position_(0);
            crnt_position_(1) = original_position_(1);
            crnt_position_(2) = original_position_(2);
            crnt_rpy_(0) = original_rpy_(0);
            crnt_rpy_(1) = original_rpy_(1);
            crnt_rpy_(2) = original_rpy_(2);
            crnt_gps_(0) = gps_(0);
            crnt_gps_(1) = gps_(1);
            crnt_gps_(2) = gps_(2);

            last_based_map_cloud.reset(new Pointcloud);
            last_based_map_cloud->clear();
            *last_based_map_cloud = *crnt_based_map_cloud;

            voxelfiltering();

            Matrix4f mat_to_origin;
            Pointcloud::Ptr temp_cloud (new Pointcloud);
            temp_cloud->clear();
            *temp_cloud += *last_lane_sample_map;
            *temp_cloud += *last_objt_sample_map;

            double x_, y_, z_;
            x_ = -1 * last_position_(0);
            y_ = -1 * last_position_(1);
            z_ = -1 * last_position_(2);
            mat_to_origin << 1,0,0,x_,
                             0,1,0,y_,
                             0,0,1,z_,
                             0,0,0,1;
            pcl::transformPointCloud(*temp_cloud,*temp_cloud,mat_to_origin);
            Eigen::Affine3f mat_to_origin_2 = getTFMatrixToOrigin();
            pcl::transformPointCloud(*temp_cloud,*temp_cloud,mat_to_origin_2);

            crnt_based_map_cloud.reset(new Pointcloud);
            crnt_based_map_cloud->clear();
            crnt_based_map_cloud->points.resize(temp_cloud->points.size());
            pcl::copyPointCloud(*temp_cloud, *crnt_based_map_cloud);

            publishMap();

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
        }

        void publishMap ()
        {
            msgs_Point temp_cloud;
            ros::Time time_ = ros::Time::now();

            // ------ map trajectory ------ //
            Pointcloud::Ptr temp_ (new Pointcloud);
            temp_->clear();

            temp_->points.resize(map_trajectory->points.size());
            for(size_t i=0;i<temp_->points.size(); i++)
            {
                temp_->points[i].x = map_trajectory->points[i].x;
                temp_->points[i].y = map_trajectory->points[i].y;
                temp_->points[i].z = map_trajectory->points[i].z;
                temp_->points[i].intensity = i;
            }

            pcl::toROSMsg(*temp_, temp_cloud);
            temp_cloud.header.frame_id = "odom";
            temp_cloud.header.stamp = time_;
            pub_mapTrajectory.publish(temp_cloud);

            // ------ sample point cloud ------ //
            Eigen::Affine3f mat_;
            mat_ = pcl::getTransformation(last_position_(0),last_position_(1),last_position_(2),
                                          last_rpy_(2),last_rpy_(1),last_rpy_(0));
            temp_->points.resize(crnt_based_map_cloud->points.size());
            pcl::transformPointCloud(*crnt_based_map_cloud,*temp_,mat_);

            pcl::toROSMsg(*temp_, temp_cloud);
            temp_cloud.header.frame_id = "odom";
            temp_cloud.header.stamp = time_;
            pub_mapPoint.publish(temp_cloud);

            // ------ origin based point cloud ------ //
            pcl::toROSMsg(*crnt_based_map_cloud, temp_cloud);
            temp_cloud.header.frame_id = "odom";
            temp_cloud.header.stamp = time_;
            pub_mapOrigin.publish(temp_cloud);

            // ------------------------------------- //
            Pointcloud::Ptr test_ (new Pointcloud);
            test_->points.resize(2);
            test_->points[0].x = last_position_(0);
            test_->points[0].y = last_position_(1);
            test_->points[0].z = last_position_(2);
            test_->points[0].intensity = 1;
            test_->points[1].x = crnt_position_(0);
            test_->points[1].y = crnt_position_(1);
            test_->points[1].z = crnt_position_(2);
            test_->points[1].intensity = 100;

            pcl::toROSMsg(*test_, temp_cloud);
            temp_cloud.header.frame_id = "odom";
            temp_cloud.header.stamp = time_;
            pub_test.publish(temp_cloud);

        }

        void gpsCallBack (const msgs_Nav::ConstPtr& gps_msg)
        {
            double utm_x, utm_y, utm_z;
            LatLonToUTMXY(gps_msg->latitude, gps_msg->longitude, 52, utm_y, utm_x);

            utm_y = -utm_y;
            utm_z = gps_msg->altitude;

            if(!isGps)
            {
                if (!isnan(utm_x) && !isnan(utm_y) && !isnan(utm_z)) {
                    f_utm_x = utm_x;
                    f_utm_y = utm_y;
                    f_utm_z = utm_z;
                    isGps = true;
                }
            }
            else {
            //for debugging
                utm_x -= f_utm_x;
                utm_y -= f_utm_y;
                utm_z -= f_utm_z;
            }

            gps_(0) = utm_x;
            gps_(1) = utm_y;
            gps_(2) = utm_z;

            /*std::cout << utm_x << "\n"
                      << utm_y << "\n"
                      << utm_z << "\n"
                      << f_utm_x << "\n"
                      << f_utm_y << "\n"
                      << f_utm_z << "\n" << std::endl;*/

            return;
        }

        void loopClosureThread ()
        {
            if(loopclosure_flag == false)
                return;

            ros::Rate rate_loop(100);
            while(ros::ok())
            {
                if(findNearestNode())
                {
                    if(calculateNDTScore())
                    {
                        DoLoopClosure();
                    }
                }
            }
        }

        bool findNearestNode ()
        {
            /*if(crnt_map_cloud->points.empty())
                return false;*/
            
            double last_dist = search_size + 1;

            PointI input_position;
            input_position.x = last_position_(0);
            input_position.y = last_position_(1);
            input_position.z = last_position_(2);
            input_position.intensity = 0.0;

            pcl::KdTreeFLANN<PointI> kdtree;
            kdtree.setInputCloud(map_position);

            std::vector<int> node_idx_radius_search;
            std::vector<float> node_radius_sqared_dist;

            kdtree.radiusSearch(input_position, search_size, node_idx_radius_search, node_radius_sqared_dist);
            
            for(size_t i=0; i<node_idx_radius_search.size(); i++)
            {
                double dist = sqrt(node_radius_sqared_dist[i]);
                if(last_dist > dist)
                {
                    last_dist = dist;
                    nearest_node_idx = node_idx_radius_search[i];
                }
            }

            if (last_dist < search_size + 1)
            {
                std::cout << nearest_node_idx << std::endl;
                return true;
            }
            else
            {
                return false;
            }
            
        }

        bool calculateNDTScore ()
        {
            Pointcloud::Ptr input_ (new Pointcloud);
            Pointcloud::Ptr target_ (new Pointcloud);
            Pointcloud::Ptr aligned_ (new Pointcloud);
 
            extractNodePointCloud();

            Eigen::Affine3f input_mat_;
            input_mat_ = pcl::getTransformation(last_position_(0),last_position_(1),last_position_(2),
                                                last_rpy_(2),last_rpy_(1),last_rpy_(0));
            input_->points.resize(crnt_based_map_cloud->points.size());
            pcl::transformPointCloud(*crnt_based_map_cloud, *input_, input_mat_);

            pcl::NormalDistributionsTransform<PointI, PointI> ndt;

            ndt.setTransformationEpsilon(Epsilon);
            ndt.setStepSize (stepSize);
            ndt.setResolution (resolution);
            ndt.setMaximumIterations (iteration);
            
            ndt.setInputSource (input_);
            ndt.setInputTarget (target_);
            ndt.align(*aligned_, init_tf_matrix);
        }

        void extractNodePointCloud ()
        {
            temp_target.reset(new Pointcloud);
            temp_target->clear();

            PointI input_position;
            input_position.x = map_position->points[nearest_node_idx].x;
            input_position.y = map_position->points[nearest_node_idx].y;
            input_position.z = map_position->points[nearest_node_idx].z;
            input_position.intensity = 0.0;

            pcl::KdTreeFLANN<PointI> kdtree;
            kdtree.setInputCloud(map_point);

            std::vector<int> node_idx_radius_search;
            std::vector<float> node_radius_sqared_dist;

            kdtree.radiusSearch(input_position, targetMapSize, node_idx_radius_search, node_radius_sqared_dist);

            pcl::PointIndices::Ptr temp_idx (new pcl::PointIndices);
            for(size_t i=0; i<node_idx_radius_search.size(); i++)
            {
                temp_idx->indices.push_back(node_idx_radius_search[i]);
            }

            pcl::ExtractIndices<PointI> extract;
            extract.setInputCloud (map_point);
            extract.setIndices (temp_idx);
            extract.setNegative (false);
            extract.filter (*temp_target);

            return;
        }

        void DoLoopClosure ()
        {

        }
};

int main (int argc, char** argv)
{
    ros::init(argc, argv, "ese_localization");

    EseLocalization EL;

    std::thread loopClosureThread(&EseLocalization::loopClosureThread, &EL);

    ROS_INFO("\033[1;32m---->\033[0m ESE Localization Started.");

    ros::spin();

    return 0;
}