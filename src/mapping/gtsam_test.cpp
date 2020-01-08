#include <ese_slam/slam_util.h>

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

int main (int argc, char** argv)
{
    ros::init(argc, argv, "gtsam_test");
    ros::NodeHandle nh;

    ros::Publisher pub_init_point=nh.advertise<msgs_Point>("init_pose",10);
    ros::Publisher pub_optimized_point = nh.advertise<msgs_Point>("optimized_pose",10);

    Pointcloud::Ptr init_point (new Pointcloud);
    Pointcloud::Ptr optimized_point (new Pointcloud);
    Pointcloud::Ptr init_rpy (new Pointcloud);
    Pointcloud::Ptr optimized_rpy (new Pointcloud);

    init_point->points.resize(24);
    optimized_point->points.resize(24);
    init_rpy->points.resize(24);
    optimized_rpy->points.resize(24);

    NonlinearFactorGraph graph;

    Values init_value;
    Values optimized_value;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odomNoise;

    init_point->points[0].x=0.0; init_point->points[0].y=0.0; init_point->points[0].z=0.0;
    init_point->points[1].x=5.0; init_point->points[1].y=0.0; init_point->points[1].z=0.0;
    init_point->points[2].x=10.0; init_point->points[2].y=0.0; init_point->points[2].z=0.0;
    init_point->points[3].x=15.0; init_point->points[3].y=0.0; init_point->points[3].z=0.0;
    init_point->points[4].x=15.0; init_point->points[4].y=5.0; init_point->points[4].z=0.0;
    init_point->points[5].x=15.0; init_point->points[5].y=10.0; init_point->points[5].z=0.0;
    init_point->points[6].x=15.0; init_point->points[6].y=15.0; init_point->points[6].z=0.0;
    init_point->points[7].x=10.0; init_point->points[7].y=15.0; init_point->points[7].z=0.0;
    init_point->points[8].x=5.0; init_point->points[8].y=15.0; init_point->points[8].z=0.0;
    init_point->points[9].x=0.0; init_point->points[9].y=15.0; init_point->points[9].z=0.0;
    init_point->points[10].x=0.0; init_point->points[10].y=10.0; init_point->points[10].z=0.0;
    init_point->points[11].x=0.0; init_point->points[11].y=5.0; init_point->points[11].z=0.0;
    init_point->points[12].x=0.0; init_point->points[12].y=0.0; init_point->points[12].z=0.0;
    init_point->points[13].x=0.0; init_point->points[13].y=-5.0; init_point->points[13].z=0.0;
    init_point->points[14].x=0.0; init_point->points[14].y=-10.0; init_point->points[14].z=0.0;
    init_point->points[15].x=0.0; init_point->points[15].y=-15.0; init_point->points[15].z=0.0;
    init_point->points[16].x=-5.0; init_point->points[16].y=-15.0; init_point->points[16].z=0.0;
    init_point->points[17].x=-10.0; init_point->points[17].y=-15.0; init_point->points[17].z=0.0;
    init_point->points[18].x=-15.0; init_point->points[18].y=-15.0; init_point->points[18].z=0.0;
    init_point->points[19].x=-15.0; init_point->points[19].y=-10.0; init_point->points[19].z=0.0;
    init_point->points[20].x=-15.0; init_point->points[20].y=-5.0; init_point->points[20].z=0.0;
    init_point->points[21].x=-15.0; init_point->points[21].y=0.0; init_point->points[21].z=0.0;
    init_point->points[22].x=-10.0; init_point->points[22].y=0.0; init_point->points[22].z=0.0;
    init_point->points[23].x=-5.0; init_point->points[23].y=0.0; init_point->points[23].z=0.0;

    for(int i=0;i<init_rpy->points.size(); i++)
    {
        init_rpy->points[i].x = 0.0;
        init_rpy->points[i].y = 0.0;
    }

    init_rpy->points[0].z = 0.0; init_rpy->points[1].z = 0.0; init_rpy->points[2].z = 0.0;
    init_rpy->points[3].z = PI/2; init_rpy->points[4].z = PI/2; init_rpy->points[5].z = PI/2;
    init_rpy->points[6].z = -1*PI; init_rpy->points[7].z = -1*PI; init_rpy->points[8].z = -1*PI;
    init_rpy->points[9].z = -1*PI/2; init_rpy->points[10].z = -1*PI/2; init_rpy->points[11].z = -1*PI/2;
    init_rpy->points[12].z = -1*PI/2; init_rpy->points[13].z = -1*PI/2; init_rpy->points[14].z = -1*PI/2;
    init_rpy->points[15].z = -1*PI; init_rpy->points[16].z = -1*PI; init_rpy->points[17].z = -1*PI;
    init_rpy->points[18].z = PI; init_rpy->points[19].z = PI; init_rpy->points[20].z = PI;
    init_rpy->points[21].z = 0.0; init_rpy->points[22].z = 0.0; init_rpy->points[23].z = 0.0;

    for(int i=0; i<init_point->points.size();i++)
    {
        init_point->points[i].x += (((rand()%25+1)-12.5)/10);
        init_point->points[i].y += (((rand()%25+1)-12.5)/10);
        init_point->points[i].z += (((rand()%25+1)-12.5)/10);
        init_point->points[i].intensity = i;
        init_rpy->points[i].x += (((rand()%5+1)-2.5)/10);
        init_rpy->points[i].y += (((rand()%5+1)-2.5)/10);
        init_rpy->points[i].z += (((rand()%5+1)-2.5)/10);
    }

    gtsam::Vector Vector6(6);
    Vector6 << 0.25, 0.25, 0.25, 0.025, 0.025, 0.025;
    priorNoise = noiseModel::Diagonal::Sigmas(Vector6);
    odomNoise = noiseModel::Diagonal::Sigmas(Vector6);

    Pose3 priorMean = Pose3(Rot3::ypr(init_rpy->points[0].z, init_rpy->points[0].y, init_rpy->points[0].x),
                             Point3(init_point->points[0].x, init_point->points[0].y, init_point->points[0].z));
    graph.add(PriorFactor<Pose3>(0, priorMean, priorNoise));
    init_value.insert(0, priorMean);

    for(int i=1;i<24;i++)
    {
        Pose3 poseFrom = Pose3(Rot3::ypr(init_rpy->points[i-1].z, init_rpy->points[i-1].y, init_rpy->points[i-1].x),
                             Point3(init_point->points[i-1].x, init_point->points[i-1].y, init_point->points[i-1].z));
        Pose3 poseTo = Pose3(Rot3::ypr(init_rpy->points[i].z, init_rpy->points[i].y, init_rpy->points[i].x),
                             Point3(init_point->points[i].x, init_point->points[i].y, init_point->points[i].z));

         graph.add(BetweenFactor<Pose3>(i-1,i,poseFrom.between(poseTo),odomNoise));
         init_value.insert(i, poseTo);
    }
    graph.add(BetweenFactor<Pose3>(12,0,Pose3(Rot3::ypr(-1*PI/2, 0.0, 0.0),Point3(0.0, 0.0, 0.0)),odomNoise));
    graph.add(BetweenFactor<Pose3>(23,0,Pose3(Rot3::ypr(0.0, 0.0, 0.0),Point3(0.0, 0.0, 0.0)),odomNoise));

    LevenbergMarquardtOptimizer optimizer(graph, init_value);
    optimized_value = optimizer.optimize();

    for(int i =0; i<24; i++)
    {
        optimized_point->points[i].x = optimized_value.at<Pose3>(i).translation().x();
        optimized_point->points[i].y = optimized_value.at<Pose3>(i).translation().y();
        optimized_point->points[i].z = optimized_value.at<Pose3>(i).translation().z();
    }    

    ros::Rate rate_(10);
    while(1)
    {
        //ros::spinOnce();

        msgs_Point temp_cloud;
        ros::Time time_ = ros::Time::now();

        pcl::toROSMsg(*init_point, temp_cloud);
        temp_cloud.header.frame_id = "odom";
        temp_cloud.header.stamp = time_;
        pub_init_point.publish(temp_cloud);

        pcl::toROSMsg(*optimized_point, temp_cloud);
        temp_cloud.header.frame_id = "odom";
        temp_cloud.header.stamp = time_;
        pub_optimized_point.publish(temp_cloud);
        
        ROS_INFO("pub \n");
        sleep(1);
        //rate_.sleep();
    }

    return 0;
}