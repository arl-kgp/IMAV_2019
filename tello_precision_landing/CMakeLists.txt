cmake_minimum_required(VERSION 2.8.3)
project(bebop_precision_landing)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)


set(CMAKE_CXX_STANDARD 11)
find_package(Protobuf REQUIRED)

find_package(PkgConfig)
## use pkg-config to get hints for 0mq locations
pkg_check_modules(PC_ZeroMQ QUIET zmq)
find_package (Threads)

## use the hint from above to find where 'zmq.hpp' is located
find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )

## use the hint from about to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )
        
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

add_executable(ibvs_test src/detect.cpp src/helpers.cpp src/point_detect.cpp src/track.cpp src/ibvs.cpp src/point_detect.cpp src/ros.cpp src/image.pb.cc)
#add_executable(ibvs_point src/detect.cpp src/helpers.cpp src/point_detect.cpp src/track.cpp src/ibvs.cpp src/point_detect.cpp src/point.cpp)
target_link_libraries(ibvs_test ${ZeroMQ_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${Protobuf_LIBRARIES})

target_link_libraries(ibvs_test ${OpenCV_LIBRARIES})
target_link_libraries(ibvs_test ${Eigen3_LIBRARIES})
#target_link_libraries(ibvs_point ${catkin_LIBRARIES})
#target_link_libraries(ibvs_point ${OpenCV_LIBRARIES})
#target_link_libraries(ibvs_point ${Eigen3_LIBRARIES})

#configure_file(src/tello_land.py tello_land.py COPYONLY)
#configure_file(src/image_pb2.py image_pb2.py COPYONLY)
