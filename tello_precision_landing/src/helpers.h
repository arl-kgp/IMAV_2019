#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <queue>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking/tracker.hpp>

#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/QR>    


using namespace std;
using namespace cv;
using namespace Eigen;


extern int thres;
extern int thresh2;
extern int erosion_size;
typedef Point3_<uint8_t> Pixel;
typedef Point_<uint8_t> Pixeli;

extern int maxTrackbar;
extern int maxCorners;
extern RNG rng;

const double PI = 3.1415;

struct quad
{
	Point2f p[4];
};

double distance(Point2f p1, Point2f p2);

bool isValid(int i, int j);

extern string source_window;

double tri_area(Point2f p1, Point2f p2, Point2f p3);

double quad_area(quad q);

int min(int x, int y);
int max(int x, int y);

vector<vector<int> > getCombinations(vector<int> indices, int n, int r);
void combinationUtil(vector<vector<int> >& out, int data[], int start, int end, int index, int r, vector<int> indices);


void Smoothen(Mat im, Mat& smoothed);
#endif