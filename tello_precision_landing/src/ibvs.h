#ifndef IBVS_H
#define IBVS_H

#include <math.h>
#include "helpers.h"

using namespace cv;
using namespace std;
using namespace Eigen;

extern double cx;
extern double cy;
extern float fx;
extern float fy;

extern float depth;

extern vector<Point2d> reference;

extern void rqt_plot(float);

MatrixXf get_velocity_ibvs(vector<Point2f> points);

#endif