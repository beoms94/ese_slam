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
add_executable(mapOptimizer src/mapping/mapOptimizer.cpp)
target_link_libraries(mapOptimizer ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)

add_executable(ese_localization src/mapping/ese_localization.cpp)
target_link_libraries(ese_localization ${catkin_LIBRARIES} ${PCL_LIBRARIES} gtsam)
