#ifndef ROS_H
#define ROS_H

#include "helpers.h"
#include "image.pb.h"

#include <zmq.hpp>

extern Mat buffer_frame;
extern uint64_t fc;

void rqt_plot(float);
void publish_vel(MatrixXf vel);
void publish_vel_forward();
Mat get_new_frame();
#endif