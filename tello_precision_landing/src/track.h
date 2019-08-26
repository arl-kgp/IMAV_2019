#ifndef TRACK_H
#define TRACK_H
#include "helpers.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking/tracker.hpp>

using namespace std;
using namespace cv;

bool track(Mat initframe, Rect &r, bool);


#endif