cmake_minimum_required(VERSION 2.8.3)
project(ese_slam)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport

  pcl_ros
  pcl_conversions

  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

find_package(GTSAM REQUIRED QUIET)
find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

add_service_files(FILES
  save_sample_map.srv
  save_global_map.srv
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES ese_slam
#  CATKIN_DEPENDS nav_msgs pcl_conversions pcl_ros roscpp rospy sensor_msgs tf
  DEPENDS system_lib PCL
)

include_directories(
 include
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
)

link_directories(
	include
  ${OpenCV_LIBRARY_DIRS}
	${PCL_LIBRARY_DIRS}
	${GTSAM_LIBRARY_DIRS}
)

##-----------------------------------------------------------------
#add_executable(imu_dr src/mapping/imu_dr.cpp)
#target_link_libraries(imu_dr ${catkin_LIBRARIES})

##-----------------------------------------------------------------
#add_executable(lidar_projection src/mapping/lidar_projection.cpp)
#target_link_libraries(lidar_projection ${catkin_LIBRARIES})
#add_dependencies(lidar_projection ${catkin_EXPORTED_TARGETS})

##-----------------------------------------------------------------
add_executable(mapOptimizer src/mapping/mapOptimizer.cpp)
target_link_libraries(mapOptimizer ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)

add_executable(gtsam_test src/mapping/gtsam_test.cpp)
target_link_libraries(gtsam_test ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)

##-----------------------------------------------------------------
#add_executable(LidarProjection src/mapping/LidarProjection.cpp)
#target_link_libraries(LidarProjection ${catkin_LIBRARIES})
#add_dependencies(LidarProjection ${catkin_EXPORTED_TARGETS})
