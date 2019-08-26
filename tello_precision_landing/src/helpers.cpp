#include "helpers.h"

int thres = 120;
int thresh2 = 150;
int erosion_size = 3;

int maxTrackbar = 100;
int maxCorners = 12;
RNG rng(2135);

string source_window = "ShiTomashiCorner";

double distance(Point2f p1, Point2f p2)
{
	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

double tri_area(Point2f p1, Point2f p2, Point2f p3)
{
	double a = distance(p1, p2), b = distance(p2, p3), c = distance(p3, p1);
	double s = (a + b + c) / 2;
	return sqrt(s * (s - a) * (s - b) * (s - c));
}

double quad_area(quad q)
{
	auto ABC = (q.p[0].y - q.p[1].y) * q.p[2].x + (q.p[1].x - q.p[0].x) * q.p[2].y + (q.p[0].x * q.p[1].y - q.p[1].x * q.p[0].y);
	auto ABD = (q.p[0].y - q.p[1].y) * q.p[3].x + (q.p[1].x - q.p[0].x) * q.p[3].y + (q.p[0].x * q.p[1].y - q.p[1].x * q.p[0].y);
	auto BCD = (q.p[1].y - q.p[2].y) * q.p[3].x + (q.p[2].x - q.p[1].x) * q.p[3].y + (q.p[1].x * q.p[2].y - q.p[2].x * q.p[1].y);
	auto CAD = (q.p[2].y - q.p[0].y) * q.p[3].x + (q.p[0].x - q.p[2].x) * q.p[3].y + (q.p[2].x * q.p[0].y - q.p[0].x * q.p[2].y);
	if(ABC == 0 || BCD == 0 || CAD == 0 || ABD == 0) return -1e10;
	if (ABC < 0)
	{
		ABC *= -1, ABD *= -1, BCD *= -1, CAD *= -1;
	}
	if (ABD > 0 && BCD > 0 && CAD < 0)
	{
		return tri_area(q.p[0], q.p[1], q.p[2]) + tri_area(q.p[0], q.p[3], q.p[2]);
	}
	else if (ABD > 0 && BCD < 0 && CAD > 0)
	{
		return tri_area(q.p[0], q.p[1], q.p[3]) + tri_area(q.p[0], q.p[3], q.p[2]);
	}
	else if (ABD < 0 && BCD > 0 && CAD > 0)
	{
		return tri_area(q.p[0], q.p[3], q.p[1]) + tri_area(q.p[0], q.p[1], q.p[2]);
	}
	return -1e10;
}

int min(int x, int y)
{
	return x > y ? y : x;
}
int max(int x, int y)
{
	return x < y ? y : x;
}

vector<vector<int>> getCombinations(vector<int> indices, int n, int r)
{
	int *data = new int[r];
	vector<vector<int>> out;
	combinationUtil(out, data, 0, n - 1, 0, r, indices);
	delete data;
	return out;
}

void combinationUtil(vector<vector<int>> &out, int data[], int start, int end, int index, int r, vector<int> indices)
{
	if (index == r)
	{
		vector<int> t;
		for (int j = 0; j < r; j++)
			t.push_back(data[j]);
		out.push_back(t);
		return;
	}
	for (int i = start; i <= end && end - i + 1 >= r - index; i++)
	{
		data[index] = indices[i];
		combinationUtil(out, data, i + 1, end, index + 1, r, indices);
	}
}


void Smoothen(Mat im, Mat& smoothed)
{

    Mat cont = ~im;
    Mat original = Mat::zeros(im.rows, im.cols, CV_8UC1);
    smoothed = Mat(im.rows, im.cols, CV_8UC1, Scalar(255));

    // contour smoothing parameters for gaussian filter
    int filterRadius = 5;
    int filterSize = 2 * filterRadius + 1;
    double sigma = 10;

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(cont, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0));
    for(size_t j = 0; j < contours.size(); j++)
	{
        size_t len = contours[j].size() + 2 * filterRadius;
        size_t idx = (contours[j].size() - filterRadius);
        vector<float> x, y;
        for (size_t i = 0; i < len; i++)
        {
            x.push_back(contours[j][(idx + i) % contours[j].size()].x);
            y.push_back(contours[j][(idx + i) % contours[j].size()].y);
        }
        vector<float> xFilt, yFilt;
        GaussianBlur(x, xFilt, Size(filterSize, filterSize), sigma, sigma);
        GaussianBlur(y, yFilt, Size(filterSize, filterSize), sigma, sigma);
        vector<vector<Point> > smoothContours;
        vector<Point> smooth;
        for (size_t i = filterRadius; i < contours[j].size() + filterRadius; i++)
        {
            smooth.push_back(Point(xFilt[i], yFilt[i]));
        }
        smoothContours.push_back(smooth);

        Scalar color;

        if(hierarchy[j][3] < 0 )
        {
            color = Scalar(0);
        }
        else
        {
            color = Scalar(255);
        }
        drawContours(smoothed, smoothContours, 0, color, -1);
    }
}