/*
//                                   ESE SLAM ( MAP OPTIMIZER )

              개발 시작 일자 : 2018.12.27                    초기 개발 완료 일자 : 2019.01.18            
                                                                                                  
 이 소프트웨어는 인천대학교 임베디드시스템공학과 Active Control Lab에서 진행하였으며, LiDAR, GPS, Odometry를 기반   
 으로 SLAM을 수행함을 목적으로 한다. 이 프로그램은 Carnegie Mellon University에서 개발한 LeGO-LOAM을 참고하여 제
 작 되었다.                                                   

    개발 담당자   : 문범석                                                                                                   
    개발 참여자   : 문범석                
 
    개발 환경     : Ubuntu 16.04 LTS, ROS kinetic  
    사용 언어     : C++                                                             
    사용 Library : 1. PCL 
                  2. Eigen 
                  3. GTSAM 
                  4. Opencv(사용 예정)
    입력 데이터   : 
    -------------------------------------------------------------------------------------------------------------
    |        Type         |               Topic Name              |     Frame_id     |       Message Type       |
    |---------------------|---------------------------------------|------------------|--------------------------|
    |      Odometry       |                  odom                 |       odom       |    nav_msgs::Odometry    |
    |---------------------|---------------------------------------|------------------|--------------------------|
    |                     | ese_lidar_projection/leftMarker_cloud |  lidar_velodyne  | sensor_msgs::PointCloud2 |
    |     Pointcloud      |   ese_lidar_projection/object_cloud   |  lidar_velodyne  | sensor_msgs::PointCloud2 |
    |---------------------|---------------------------------------|------------------|--------------------------|
    |         GPS         |                   fix                 |        gps       |  sensor_msgs::NavSatFix  |
    -------------------------------------------------------------------------------------------------------------
    
    지원 서비스   : 1. save_sample_map.srv 
                  2. save_global_map.srv 
                  3. save_optimized_trajectory.srv  (제작 예정)

    출력 데이터   : 
    -------------------------------------------------------------------------------------------
    |        Type         |     Topic Name      |     Frame_id     |       Message Type       |
    |---------------------|---------------------|------------------|--------------------------|
    |                     | original_trajectory |       odom       | sensor_msgs::PointCloud2 |
    |      Odometry       |      trajectory     |       odom       | sensor_msgs::PointCloud2 |
    |                     |      final_odom     |       odom       |    nav_msgs::Odometry    |
    |---------------------|---------------------|------------------|--------------------------|
    |                     |      origin_map     |       odom       | sensor_msgs::PointCloud2 |
    |     Pointcloud      |      sample_map     |       odom       | sensor_msgs::PointCloud2 |
    |                     |       input_map     |       odom       | sensor_msgs::PointCloud2 |
    |                     |      target_map     |       odom       | sensor_msgs::PointCloud2 |
    -------------------------------------------------------------------------------------------
*/


/*
//                                       HEADER LINE

  ese_slam/slam_util.h  :  기본 포함 header 및 define, PointXYZMAP 에 대한 정의가 포함되어 있다. 
  ese_slam/UTM.h        :  GPS로 부터 획득한 WGS84 좌표 데이터를 UTM 좌표로 변환하는 함수들이 정의되어있다.

  ese_slam/save_sample_map.h  &  ese_slam/save_global_map.h  :  Service를 사용하기 위한 header 

  GTSAM 관련 Header  :  1. Rot3.h & Pose3.h               |  3차원적인 차량 pose를 표현하기 위한 header
                                                         |  2차원으로 바꾸려면 Rort2.h 와 Pose2 사용
                       ----------------------------------|---------------------------------------- 
                       2. PriorFactor.h                  |  초기 위치 설정을 위한 factor
                       ----------------------------------|----------------------------------------
                       3. BetweenFactor.h                |  Odometry측정값 간 차이를 담는 factor
                       ----------------------------------|----------------------------------------
                       4. NonlinearFactorGraph.h         |  입력 데이터가 비선형이기 떄문에 사용  
                       ----------------------------------|----------------------------------------        
                       5. LevenbergMarquardtOptimizer.h  |  loop closing시 정합을 위해 사용
                       ----------------------------------|----------------------------------------
                       6. Values.h                       |  BetweenFactor와 유사하나 데이터 원본이
                                                         |  저장되며 실제 정합된 값들이 저장됨 
*/
#include <ese_slam/slam_util.h>
#include <ese_slam/UTM.h>

#include "ese_slam/save_sample_map.h"
#include "ese_slam/save_global_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Values.h>

using namespace gtsam;

/*
//                                      CLASS PART

  SLAM에 필요한 변수 및 함수에 대한 모든 정의가 들어있음 
  모든 변수들은 private로 되어있어 외부에서 접근이 불가하며, 모든 함수들은 public으로 되어 있어 접근이 가능함
  아마 외부에서 접근할 일은 없을 것으로 생각됨
*/
class MapOptimizer{
    /*
    //                                         변수 선언부         
    
    // GTSAM part
          GTSAM은 모르는 사람이 너무 많으므로 변수에 대한 설명을 모두 해드림. 보고 참고하시면 됨 
          이거로 불충분 하면 참고 자료를 보시는걸 추천함!

          참고 자료 - 1) https://www.cc.gatech.edu/grads/j/jdong37/files/gtsam-tutorial.pdf
                    2) https://www.cc.gatech.edu/fac/Frank.Dellaert/FrankDellaert/Frank_Dellaert/Entries/2013/5/10_Factor_Graphs_Tutorial_files/gtsam.pdf

          1. graph               :  PriorFactor 와 BetweenFactor가 저장되며 정합전 pose 데이터가 저장된다.

          2. init_value          :  정합 전 Odometry를 통해 얻은 위치를 저장하기 위한 value
          3. optimized_value     :  정합 후 Odometry를 통해 얻은 위치를 저장하기 위한 value

          4. priorNoise          :  초기 위치에 대한 오차를 설정
          5. odometryNoise       :  Odometry data의 오차를 설정
          6. ndtNoise            :  ndt 결과 score를 오차로 설정

          7. graph_idx           :  현재 graph의 index
          8. crnt_idx            :  가장 최근에 생성된 graph의 index
          9. nearest_node_idx    :  가장 인접한 graph node의 index이며, GPS 데이터를 이용하여 찾는다.
          10. loop_closure_flag  :  True --> loop_closure 실행 O , False --> loop_closure 실행 X
          11. isLoop             :  loop_closing이 발생했는지 여부를 알려줌
          12. search_size        :  가장 인접한 node를 찾기위한 최대 범위
          13. threshold_score    :  loop_closing을 하기위한 최대 score
          14. ndt_align_score    :  최종 ndt 계산 결과 score

          15. nearest_node_point :  가장 인접한 node의 xyz
          16. nearest_node_rpy   :  가장 인접한 node의 yaw pitch roll
          17. init_tf_matrix     :  ndt계산을 위한 초기 정합 matrix     -->     계산을 위한 것이므로 건들지 마시오
          18. ndt_tf_matrix      :  ndt 계산을 통해 얻은 변환 Matrix
          19. ndt_tf_affine      :  ndt 계산을 통해 얻은 변환 Affine

    // ROS part (Parameter Only)
          launch file에서 사용되는 Parameter 위주로 설명 하겠습니다. 그외 변수들은 쫒아가다보면 이해할 수 있을 거라 생각 됨

           1. mapSize          :  Sample map 생성을 위한 거리 주기 (ex. 5.0 --> 5m 간격으로 sample map 생성)
           2. search_size      :  GTSAM part에서 설명
           3. threshold_score  :  GTSAM part에서 설명
           4. right_leafSize    :  Sample map 생성 시 Object pointcloud voxel filter 크기
           5. left_leafSize    :  Sample map 생성 시 left pointcloud voxel filter 크기
           6. Epsilon          :  ndt 계산이 멈추기 위한 최저 score 변화량
           7. stepSize         :  초기 ndt 변환시 최대 변화 거리 및 각도(라디안) 
           8. resolution       :  ndt 계산을 위한 Voxel 크기
           9. iteration        :  ndt 게산 시 최대 반복 횟수 
    */
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
        size_t nearest_node_idx;
        bool loopclosure_flag;
        bool isLoop;
        double search_size;
        double threshold_score;
        double ndt_align_score;

        Vector3 nearest_node_point;
        Vector3 nearest_node_rpy;
        Matrix4f init_tf_matrix;
        Matrix4f ndt_tf_matrix;
        Eigen::Affine3f ndt_tf_affine;

        // ------- ROS basic ------- //
        ros::NodeHandle nh;

        ros::Subscriber sub_left;
        ros::Subscriber sub_right;
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
        ros::Publisher pub_optimized_trajectory;
        // ---------------------------- //

        ros::ServiceServer sample_service;
        ros::ServiceServer global_service;

        tf::TransformBroadcaster odom_broadcaster;

        Pointcloud::Ptr input_left_cloud_;
        Pointcloud::Ptr transformed_left_cloud_;
        Pointcloud::Ptr input_right_cloud_;
        Pointcloud::Ptr transformed_right_cloud_;

        Pointcloud::Ptr last_left_sample_map;
        Pointcloud::Ptr crnt_left_sample_map;
        Pointcloud::Ptr last_right_sample_map;
        Pointcloud::Ptr crnt_right_sample_map;

        Pointcloud::Ptr original_map_position;
        Pointcloud::Ptr original_map_rpy;
        Pointcloud::Ptr tf_map_position;
        Pointcloud::Ptr tf_map_rpy;
        Pointcloud::Ptr gps_data;
        size_t sample_map_idx;

        Pointcloud::Ptr based_sample_map[5000];    //원점을 중심으로 하는 sample map
        Pointcloud::Ptr global_map;
        PointMap::Ptr map_data; 

        Vector3f original_position_;
        Vector3f position_;
        
        Vector4f quaternion_;
        Vector3f original_rpy_;
        Vector3f rpy_;

        Vector3f gps_;
        
        Eigen::Affine3f odom_tf_affine;
        Eigen::Affine3f original_tf_affine;
        Eigen::Affine3f last_tf_affine;

        double mapSize;
        double right_leafSize, left_leafSize;
        double Epsilon, stepSize, resolution;
        int iteration;
        bool saveMapFlag;

        size_t last_optimized_node_idx;
        size_t last_crnt_idx;
        size_t loop_input_idx, loop_target_idx;

        double f_utm_x, f_utm_y, f_utm_z;
        bool gps_first;

    /*
    //                                         함수 선언부

             SECTION 1.  Initializing Variables
             SECTION 2.  PointCloud Data Processing
             SECTION 3.  GPS Data Processing
             SECTION 4.  Odometry Data Processing
             SECTION 5.  Loop Closure Thread Processing
             SECTION 6.  Service Processing
    */
    public:
        /*
        //                          SECTION 1.  Initializing Variables

            MapOptimizer()
                '--initializeValue()
        */
        MapOptimizer():
        nh("~")
        {
            // ------- ROS basic ------- //
            sample_service = nh.advertiseService("save_sample_map",&MapOptimizer::save_sample_service,this);
            global_service = nh.advertiseService("save_global_map",&MapOptimizer::save_global_service,this);

            sub_left = nh.subscribe<msgs_Point>("/ese_left_projection/points",
                                                      10,&MapOptimizer::leftCallBack,this);
            sub_right = nh.subscribe<msgs_Point>("/ese_right_projection/points",
                                                      10,&MapOptimizer::rightCallBack,this);
            sub_odometry = nh.subscribe<msgs_Odom>("/odom",10,&MapOptimizer::odomCallBack,this);
            sub_fix = nh.subscribe<msgs_Nav>("/fix",10,&MapOptimizer::gpsCallBack,this);

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
            nh.param("right_leafSize",right_leafSize,1.0);
            nh.param("left_leafSize",left_leafSize,1.0);
            nh.param("Epsilon",Epsilon,0.01);
            nh.param("stepSize",stepSize,0.1);
            nh.param("resolution",resolution,1.0);
            nh.param("iteration",iteration,400);
            
            input_left_cloud_.reset(new Pointcloud);
            transformed_left_cloud_.reset(new Pointcloud);
            input_right_cloud_.reset(new Pointcloud);
            transformed_right_cloud_.reset(new Pointcloud);

            last_left_sample_map.reset(new Pointcloud);
            crnt_left_sample_map.reset(new Pointcloud);
            last_right_sample_map.reset(new Pointcloud);
            crnt_right_sample_map.reset(new Pointcloud);
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
            gps_ << 0,0,0;

            gps_first = false;

            saveMapFlag = false;

            // -------   GTSAM   ------- //
            gtsam::Vector Vector6(6);
            Vector6 << 25*1e-2,25*1e-2,25*1e-5,25*1e-5,25*1e-5,25*1e-3;
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

            last_optimized_node_idx = 0;
            last_crnt_idx = 0;

            ndt_tf_affine = pcl::getTransformation(0.0,0.0,0.0,0.0,0.0,0.0);
            last_tf_affine = pcl::getTransformation(0.0,0.0,0.0,0.0,0.0,0.0);
        }

        /*
        //                         SECTION 2.  PointCloud Data Processing

            PointCloud Data Processing
                   |
                   |-- leftCallBack()
                   |      |
                   |      |-- copyleftPointCloud()
                   |      |-- transformMatrixUpdate()
                   |      |-- lefttransformPointCloud()
                   |      |-- saveleftPointCloud()
                   |
                   '-- rightCallBack()
                          |
                          |-- copyrightPointCloud()
                          |-- transformMatrixUpdate()
                          |-- righttransformPointCloud()
                          |-- saverightPointCloud()
        */
        void copyLeftPointCloud(const msgs_Point::ConstPtr& point_msg)
        {
            input_left_cloud_.reset(new Pointcloud);
            input_left_cloud_->clear();

            pcl::fromROSMsg(*point_msg, *input_left_cloud_);
            return;
        }

        void copyRightPointCloud(const msgs_Point::ConstPtr& point_msg)
        {
            input_right_cloud_.reset(new Pointcloud);
            input_right_cloud_->clear();

            pcl::fromROSMsg(*point_msg, *input_right_cloud_);
            return;
        }

        void transformMatrixUpdate()
        {
            original_tf_affine = pcl::getTransformation(original_position_(0),original_position_(1),original_position_(2),
                                                        original_rpy_(2),original_rpy_(1),original_rpy_(0));
        }

        void leftTransformPointCloud()
        {
            Pointcloud::Ptr temp_ (new Pointcloud);

            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*input_left_cloud_, *temp_, indices);

            transformed_left_cloud_.reset(new Pointcloud);
            transformed_left_cloud_->points.resize(input_left_cloud_->points.size());

            pcl::transformPointCloud(*temp_, *transformed_left_cloud_, original_tf_affine);
        }

        void rightTransformPointCloud()
        {
            Pointcloud::Ptr temp_ (new Pointcloud);

            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*input_right_cloud_, *temp_, indices);

            transformed_right_cloud_.reset(new Pointcloud);
            transformed_right_cloud_->points.resize(input_right_cloud_->points.size());

            pcl::transformPointCloud(*temp_, *transformed_right_cloud_, original_tf_affine);
        }

        void saveLeftPointCloud()
        {
            *crnt_left_sample_map += *transformed_left_cloud_;
            *last_left_sample_map += *transformed_left_cloud_;
        }

        void saveRightPointCloud()
        {
            *crnt_right_sample_map += *transformed_right_cloud_;
            *last_right_sample_map += *transformed_right_cloud_;
        }

        void leftCallBack (const msgs_Point::ConstPtr& point_msg)
        {
            //1. copy input msgs
            //2. update transformation matrix
            //3. transform input cloud to transformed cloud
            //4. save transformed cloud to sample map 

            copyLeftPointCloud(point_msg);
            transformMatrixUpdate();
            leftTransformPointCloud();
            saveLeftPointCloud();
        }

        void rightCallBack (const msgs_Point::ConstPtr& point_msg)
        {
            //1. copy input msgs
            //2. update transformation matrix
            //3. transform input cloud to transformed cloud
            //4. save transformed cloud to sample map 

            copyRightPointCloud(point_msg);
            transformMatrixUpdate();
            rightTransformPointCloud();
            saveRightPointCloud();
        }

        void gpsCallBack (const msgs_Nav::ConstPtr& gps_msg)
        {
            double utm_x, utm_y, utm_z;
            LatLonToUTMXY(gps_msg->latitude, gps_msg->longitude, 52, utm_y, utm_x);

            //utm_y = -utm_y;
            utm_z = gps_msg->altitude; 

            if(sample_map_idx == 0)
            {
                gps_data.reset(new Pointcloud);
                gps_data->clear();

                Pointcloud::Ptr temp_ (new Pointcloud);
                temp_->points.resize(1);
                temp_->points[0].x = utm_y;
                temp_->points[0].y = utm_x;
                temp_->points[0].z = 0.0;   //utm_z;
                temp_->points[0].intensity = sample_map_idx;

                *gps_data += *temp_;
            }
            else
            {
                gps_(0) = utm_y;
                gps_(1) = utm_x;
                gps_(2) = utm_z;
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
            double x_, y_, z_, roll_, pitch_, yaw_;
            x_ = original_position_(0) - original_map_position->points[last_optimized_node_idx].x;
            y_ = original_position_(1) - original_map_position->points[last_optimized_node_idx].y;
            z_ = original_position_(2) - original_map_position->points[last_optimized_node_idx].z;
            roll_ = original_rpy_(2) - original_map_rpy->points[last_optimized_node_idx].z;
            pitch_ = original_rpy_(1) - original_map_rpy->points[last_optimized_node_idx].y;
            yaw_ = original_rpy_(0) - original_map_rpy->points[last_optimized_node_idx].x;


            double d_x, d_y, d_z, d_roll, d_pitch, d_yaw;
            d_x = 0.0; d_y = 0.0; d_z = 0.0;
            d_roll = tf_map_rpy->points[last_optimized_node_idx].z - original_map_rpy->points[last_optimized_node_idx].z;
            d_pitch = tf_map_rpy->points[last_optimized_node_idx].y - original_map_rpy->points[last_optimized_node_idx].y;
            d_yaw = tf_map_rpy->points[last_optimized_node_idx].x - original_map_rpy->points[last_optimized_node_idx].x;
            
            Eigen::Affine3f last_;
            last_ = pcl::getTransformation(d_x,d_y,d_z,d_roll,d_pitch,d_yaw);

            Vector4f c_tf, tf_; 
            c_tf << x_, y_, z_, 1;
            tf_ = last_.matrix() * c_tf;            

            position_(0) = tf_map_position->points[last_optimized_node_idx].x + tf_(0);
            position_(1) = tf_map_position->points[last_optimized_node_idx].y + tf_(1);
            position_(2) = tf_map_position->points[last_optimized_node_idx].z + tf_(2);
            rpy_(0) = tf_map_rpy->points[last_optimized_node_idx].x + yaw_;
            rpy_(1) = tf_map_rpy->points[last_optimized_node_idx].y + pitch_;
            rpy_(2) = tf_map_rpy->points[last_optimized_node_idx].z + roll_;

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
            voxel.setLeafSize (right_leafSize, right_leafSize, right_leafSize);
            voxel.setInputCloud (last_right_sample_map);
            voxel.filter (*temp);

            last_right_sample_map.reset(new Pointcloud);
            *last_right_sample_map = *temp;

            voxel.setInputCloud (last_left_sample_map);
            voxel.setLeafSize (left_leafSize, left_leafSize, left_leafSize);
            voxel.filter (*temp);

            last_left_sample_map.reset(new Pointcloud);
            *last_left_sample_map = *temp;

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
            temp->points[0].x = gps_(0); //utm_x
            temp->points[0].y = gps_(1); //utm_y
            temp->points[0].z = gps_(2); //utm_z
            *gps_data += *temp;

            //update sample map
            voxelfiltering();

            //based_sample_map update
            Matrix4f mat_to_origin;
            Pointcloud::Ptr temp_to_origin (new Pointcloud);
            //temp_to_origin->points.resize(last_left_sample_map->points.size() + last_right_sample_map->points.size());
            temp_to_origin->clear();
            *temp_to_origin += *last_left_sample_map;
            *temp_to_origin += *last_right_sample_map;

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
            based_sample_map[sample_map_idx]->points.resize(last_left_sample_map->points.size() + last_right_sample_map->points.size());
            pcl::copyPointCloud(*temp_to_origin,*based_sample_map[sample_map_idx]);
            sample_map_idx++;

            last_left_sample_map.reset(new Pointcloud);
            last_left_sample_map->clear();
            *last_left_sample_map = *crnt_left_sample_map;
            crnt_left_sample_map.reset(new Pointcloud);
            crnt_left_sample_map->clear();

            last_right_sample_map.reset(new Pointcloud);
            last_right_sample_map->clear();
            *last_right_sample_map = *crnt_right_sample_map;
            crnt_right_sample_map.reset(new Pointcloud);
            crnt_right_sample_map->clear();

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
                mat_ = pcl::getTransformation(tf_map_position->points[loop_input_idx].x, tf_map_position->points[loop_input_idx].y, tf_map_position->points[loop_input_idx].z,
                                              tf_map_rpy->points[loop_input_idx].z, tf_map_rpy->points[loop_input_idx].y, tf_map_rpy->points[loop_input_idx].x);
                Pointcloud::Ptr temp_ (new Pointcloud);
                temp_->points.resize(based_sample_map[loop_input_idx]->points.size());

                pcl::transformPointCloud(*based_sample_map[loop_input_idx], *temp_, mat_);

                pcl::toROSMsg(*temp_,temp_cloud);
                temp_cloud.header.frame_id = "odom";
                temp_cloud.header.stamp = time_;
                pub_input_map.publish(temp_cloud);

                mat_ = pcl::getTransformation(tf_map_position->points[loop_target_idx].x, tf_map_position->points[loop_target_idx].y, tf_map_position->points[loop_target_idx].z,
                                              tf_map_rpy->points[loop_target_idx].z, tf_map_rpy->points[loop_target_idx].y, tf_map_rpy->points[loop_target_idx].x);
                temp_->points.resize(based_sample_map[loop_target_idx]->points.size());
                pcl::transformPointCloud(*based_sample_map[loop_target_idx], *temp_, mat_);

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
            //the number of sample map is smaller than 5, don't loop closure
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

            /*for(size_t i=0; i<node_idx_radius_search.size(); i++)
            {
                double dist = sqrt(node_radius_sqared_dist[i]);
                if(last_dist > dist && node_idx_radius_search[i] < (crnt_idx-4))
                {
                    last_dist = dist;
                    nearest_node_idx = node_idx_radius_search[i];
                }
            }*/

            // ----------- for test gps ----------- //
            double gps_last_dist = search_size + 1;
            double crnt_gps_x, crnt_gps_y, crnt_gps_z;
            crnt_gps_x = gps_data->points[crnt_idx].x;
            crnt_gps_y = gps_data->points[crnt_idx].y;
            crnt_gps_z = gps_data->points[crnt_idx].z;

            for(size_t i=0; i<node_idx_radius_search.size();i++)
            {
                double diff_x, diff_y, diff_z, dist;
                diff_x = crnt_gps_x - gps_data->points[node_idx_radius_search[i]].x;
                diff_y = crnt_gps_y - gps_data->points[node_idx_radius_search[i]].y;
                diff_z = crnt_gps_z - gps_data->points[node_idx_radius_search[i]].z;
                dist = sqrt((diff_x*diff_x)+(diff_y*diff_y)); //+(diff_z*diff_z));

                if(gps_last_dist > dist && node_idx_radius_search[i] < crnt_idx-30)
                {
                    gps_last_dist = dist;
                    nearest_node_idx = node_idx_radius_search[i];
                } 

            }
            // ------------------------------------ //

            if( gps_last_dist > search_size)
                return false;
            else
            {
                std::cout << "current_index : " << crnt_idx 
                          << "\nnearest_node_index : " << nearest_node_idx << std::endl;
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

            double d_x,d_y,d_z;
            d_x = gps_data->points[nearest_node_idx].x - gps_data->points[crnt_idx].x;
            d_y = gps_data->points[nearest_node_idx].y - gps_data->points[crnt_idx].y;
            d_z = gps_data->points[nearest_node_idx].z - gps_data->points[crnt_idx].z;

            Eigen::Affine3f mat_;
            mat_ = pcl::getTransformation(d_x,d_y,d_x,0.0,0.0,0.0);
            //init_tf_matrix = mat_.matrix();

            pcl::NormalDistributionsTransform<PointI, PointI> ndt;

            ndt.setTransformationEpsilon(Epsilon);
            ndt.setStepSize (stepSize);
            ndt.setResolution (resolution);
            ndt.setMaximumIterations (iteration);

            ndt.setInputSource (input_);
            ndt.setInputTarget (target_);
            ndt.align(*aligned_, init_tf_matrix);

            ndt_align_score = ndt.getFitnessScore();

            // ------- for test ------- //
            std::cout << crnt_idx << " " << nearest_node_idx << " "
                      << ndt_align_score << std::endl;
            // ------------------------ //
            
            if(ndt_align_score < threshold_score)
            {
                ndt_tf_matrix = ndt.getFinalTransformation();
                ndt_tf_affine = ndt.getFinalTransformation();

                return true;
            }
            else if(ndt_align_score > threshold_score || ndt.hasConverged() == false)
                return false;
        }

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
            Vector6 << ndt_noise*1e-4, ndt_noise*1e-4, ndt_noise*1e-4, ndt_noise*1e-4, ndt_noise*1e-4 , ndt_noise*1e-4; 
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
                double d_x, d_y, d_z, d_roll, d_pitch, d_yaw;
                d_x = 0.0; d_y = 0.0; d_z = 0.0;
                d_roll = tf_map_rpy->points[last_optimized_node_idx].z - original_map_rpy->points[last_optimized_node_idx].z;
                d_pitch = tf_map_rpy->points[last_optimized_node_idx].y - original_map_rpy->points[last_optimized_node_idx].y;
                d_yaw = tf_map_rpy->points[last_optimized_node_idx].x - original_map_rpy->points[last_optimized_node_idx].x;
                
                Eigen::Affine3f last_;
                last_ = pcl::getTransformation(d_x,d_y,d_z,d_roll,d_pitch,d_yaw);

                for(size_t i=last_optimized_node_idx+1;i<sample_map_idx+1;i++)
                {
                    /*double x_, y_, z_, roll_, pitch_, yaw_;
                    x_ = original_map_position->points[i].x - original_map_position->points[last_optimized_node_idx].x;
                    y_ = original_map_position->points[i].y - original_map_position->points[last_optimized_node_idx].y;
                    z_ = original_map_position->points[i].z - original_map_position->points[last_optimized_node_idx].z;
                    roll_ = original_map_rpy->points[i].z - original_map_rpy->points[last_optimized_node_idx].z;
                    pitch_ = original_map_rpy->points[i].y - original_map_rpy->points[last_optimized_node_idx].y;
                    yaw_ = original_map_rpy->points[i].x - original_map_rpy->points[last_optimized_node_idx].x;  

                    Vector4f c_tf, tf_; 
                    c_tf << x_, y_, z_, 1;
                    tf_ = last_.matrix() * c_tf;            

                    tf_map_position->points[i].x = tf_map_position->points[last_optimized_node_idx].x + tf_(0);
                    tf_map_position->points[i].y = tf_map_position->points[last_optimized_node_idx].y + tf_(1);
                    tf_map_position->points[i].z = tf_map_position->points[last_optimized_node_idx].z + tf_(2);
                    tf_map_rpy->points[i].z = tf_map_rpy->points[last_optimized_node_idx].x + yaw_;
                    tf_map_rpy->points[i].y = tf_map_rpy->points[last_optimized_node_idx].y + pitch_;
                    tf_map_rpy->points[i].x = tf_map_rpy->points[last_optimized_node_idx].z + roll_;*/

                    
                    tf_map_position->points[i].x = tf_map_position->points[last_optimized_node_idx].x + (original_map_position->points[i].x - original_map_position->points[last_optimized_node_idx].x);
                    tf_map_position->points[i].y = tf_map_position->points[last_optimized_node_idx].y + (original_map_position->points[i].y - original_map_position->points[last_optimized_node_idx].y);
                    tf_map_position->points[i].z = tf_map_position->points[last_optimized_node_idx].z + (original_map_position->points[i].z - original_map_position->points[last_optimized_node_idx].z);
                    tf_map_rpy->points[i].x = tf_map_rpy->points[last_optimized_node_idx].x + (original_map_rpy->points[i].x - original_map_rpy->points[last_optimized_node_idx].x);
                    tf_map_rpy->points[i].y = tf_map_rpy->points[last_optimized_node_idx].y + (original_map_rpy->points[i].y - original_map_rpy->points[last_optimized_node_idx].y);
                    tf_map_rpy->points[i].z = tf_map_rpy->points[last_optimized_node_idx].z + (original_map_rpy->points[i].z - original_map_rpy->points[last_optimized_node_idx].z);
                    
                }
            }

            loop_input_idx = crnt_idx;
            loop_target_idx = nearest_node_idx;
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
            // Save Map Pointcloud 
            std::string file_adrs = "/home/beoms/Desktop/bag_file/global_map.pcd";
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
            
            // Save Map Trajectory
            std::string trajectory_adrs = "/home/beoms/Desktop/bag_file/map_trajectory.pcd";    
            std::stringstream sss;
            sss << trajectory_adrs;

            map_data.reset(new PointMap);
            map_data->clear();
            map_data->points.resize(last_optimized_node_idx + 1);

            for(size_t i=0; i<last_optimized_node_idx; i++)
            {
                map_data->points[i].x = tf_map_position->points[i].x;
                map_data->points[i].y = tf_map_position->points[i].y;
                map_data->points[i].z = tf_map_position->points[i].z;
                map_data->points[i].utm_x = gps_data->points[i].x;
                map_data->points[i].utm_y = gps_data->points[i].y;
                map_data->points[i].utm_z = gps_data->points[i].z;
            }

            map_data->height = 1;
            map_data->width = last_optimized_node_idx + 1;

            pcl::io::savePCDFileASCII (sss.str(), *map_data);
            std::cout << "Save Map Trajectory!! " << std::endl;
            
            return true;
        }

        void savePCD(size_t i)
        {
            std::string file_adrs = "/home/beoms/Desktop/bag_file/saved_samplemap/";
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
};

/*
//                                     MAIN PART

  loopClosure thread + MapOptimizer loop 가 동작, 총 2개의 thread가 동작한다. 
  건드릴 것 하나도 없음 그냥 만지지 마세요
*/
int main (int argc, char** argv)
{
    ros::init(argc, argv, "mapOptimizer");

    MapOptimizer MO;

    std::thread loopClosureThread(&MapOptimizer::loopClosureThread, &MO);

    ROS_INFO("\033[1;32m---->\033[0m ESE Map Optimizer Started.");

    ros::spin();

    return 0;
}