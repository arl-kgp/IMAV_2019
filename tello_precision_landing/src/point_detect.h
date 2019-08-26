#ifndef POINT_DETECT_H
#define POINT_DETECT_H

#include "helpers.h"
#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <string>

vector<Point2f> get_ordered_points(Mat croppedFrame, Point2f centroid);
#endif