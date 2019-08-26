#include "track.h"
#include "helpers.h"
#include "detect.h"
#include "ros.h"
#include "ibvs.h"
#include "point_detect.h"

bool track(Mat initframe, Rect &r, bool servo)
{
	Ptr<Tracker> tracker = TrackerMedianFlow::create();
	Rect2d roi = r;
	if (roi.width == 0 || roi.height == 0)
		return 0;
	tracker->init(initframe, roi);
	Mat frame = initframe;
	Mat crop, crop2;
	Mat stats, centroids;
	bool firstTime = 1;
	quad prev_quad_types[3];
	quad quad_types[3];
	printf("Start the tracking process, press ESC to quit.\n");

	for (;;)
	{
		//cap >> frame;
		if (frame.rows == 0 || frame.cols == 0)
			break;
		if (!tracker->update(frame, roi))
			return 0;

		double x = max(0, roi.x);
		double y = max(0, roi.y);
		Point2f initial(x, y);
		int width = min(frame.size().width - x, roi.width);
		int height = min(frame.size().height - y, roi.height);
		Rect bbox(x, y, width, height);

		frame(bbox).copyTo(crop);
		cvtColor(crop, crop, CV_BGR2GRAY);
		Mat element = getStructuringElement(MORPH_ELLIPSE,
											Size(2 * erosion_size + 1, 2 * erosion_size + 1),
											Point(erosion_size, erosion_size));

		cout << crop.rows << " " << crop.cols << endl;
		//dilate(res, res, element);
		equalizeHist(crop, crop);
		threshold(crop, crop, thresh2, 255, 0);
		erode(crop, crop, element);

		//cout << crop.channels();
		int ct = connectedComponentsWithStats(crop, crop2, stats, centroids, 4, CV_32S);
		int maxa = 0;
		int maxl = 1;

		for (int k = 1; k < ct; k++)
			if (maxa < stats.at<int>(k, CC_STAT_AREA))
				maxl = k, maxa = stats.at<int>(k, CC_STAT_AREA);

		crop2.forEach<Pixeli>([&crop, crop2, maxl](Pixeli &p, const int *position) -> void {
			if (crop2.at<int>(position[0], position[1]) != maxl)
				crop.at<uchar>(position[0], position[1]) = 0;
			//cout << (int)position[0] << " " << (int)position[1] <<"\n";
		});

		//erode(crop, crop, element);
		dilate(crop, crop, element);
		connectedComponentsWithStats(crop, crop2, stats, centroids, 4, CV_32S);

		double area = stats.at<int>(1, CC_STAT_AREA) / (double)stats.at<int>(0, CC_STAT_AREA);

		if (!(area > 0.1 && area < 0.45))
		{
			cout << area << " Area out of range in track box restarting!\n";
			//imshow("crop", crop);
			//waitKey(0);
			return 0;
		}

		Point2f centroid2(centroids.at<double>(1, 0) / crop.cols * 228.0, centroids.at<double>(1, 1) / crop.rows * 228.0);
		resize(crop, crop, Size(228, 228), 0, 0);
		/*Mat blur1, blur2;
		blur(crop,blur1,Size(3,3));
		GaussianBlur(crop, blur2,Size(3,3),0);
		absdiff(blur2,blur1,crop);
		equalizeHist(crop, crop);
    	Canny( crop, crop, 25, 150, 3 );
		GaussianBlur(crop, crop,Size(5,5),0);*/
		Mat fout;
		Smoothen(crop, fout);
		imshow("crop", crop);

		auto points = get_ordered_points(fout, centroid2);
		if (points.size() != 12)
		{
			cout << "Insufficient points\n";
			frame = get_new_frame();
			continue;
		}
		int radius = 4;
		int font = FONT_HERSHEY_SCRIPT_SIMPLEX;

		for (int i = 0; i < points.size(); i++)
		{
			std::ostringstream ss;
			ss << i;
			std::string s(ss.str());
			points[i].x = points[i].x * width / 228.0;
			points[i].y = points[i].y * height / 228.0;
			points[i] += initial;
			//cout << i << " " << (int) points[i].x<< " " << (int)points[i].y << "\n";
			putText(frame, s.c_str(), points[i], font, 1, (255, 255, 255), 2);

			circle(frame, points[i], radius * 48 / (i + 24), Scalar(255, 0, 0), FILLED);
		}

		Point2d centroid(bbox.x + centroids.at<double>(maxl, 0), bbox.y + centroids.at<double>(maxl, 1));

		cout << centroid.x << " " << centroid.y << "\n";
		circle(frame, centroid, 3, Scalar(0, 255, 0), -1, 8, 0);

		rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
		if (servo)
		{
			MatrixXf v = get_velocity_ibvs(points);
			if (v.norm() < 0.01)
			{
				cout << "IBVS FINISHED\n";
				break;
			}
			publish_vel(v);
		}
		else
		{
			cout << "\nCurrent points are: \n";
			for (int i = 0; i < points.size(); i++)
			{
				cout << i << " "<< points[i].x << " " << points[i].y << "\n";
			}
			cout << "\n";
		}
		imshow("tracker", frame);
		if (waitKey(1) == 27)
			break;
		//std::this_thread::sleep_for(std::chrono::milliseconds(1000));

		frame = get_new_frame();
	}
	return 1;
}