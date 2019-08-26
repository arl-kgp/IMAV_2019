#include "detect.h"
#include "track.h"
#include "ros.h"
#include "helpers.h"

int startDetect(bool servo)
{

	namedWindow("HoughCircles", WINDOW_NORMAL);
	namedWindow("2", WINDOW_NORMAL);
	createTrackbar("thres", "HoughCircles", &thres, 300);
	createTrackbar("thres2", "HoughCircles", &thresh2, 300);
	createTrackbar("erosion", "HoughCircles", &erosion_size, 300);
	//cout<<"here"<<endl;
	while (1)
	{
		Mat img = get_new_frame();
		cout << "!@#$";
		Mat img2 = img.clone();

		Mat gray;
		cvtColor(img, gray, CV_BGR2GRAY);
		equalizeHist(gray, gray);
		adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_MEAN_C,
						  THRESH_BINARY, 51, 2);
		GaussianBlur(gray, gray, Size(9, 9), 2, 2);

		vector<Vec3f> circles;
		HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, gray.rows / 8, 200, thres, 0, 0);

		for (size_t i = 0; i < circles.size(); i++)
		{
			Mat1b mask(img.size(), uchar(0));
			Vec3f circ = circles[i];
			//circ[2] *= 0.9;

			circle(mask, Point(circ[0], circ[1]), circ[2], Scalar(255), CV_FILLED);

			int x = max(0, circ[0] - circ[2]);
			int y = max(0, circ[1] - circ[2]);
			int width = min(img.size().width - x, 2 * circ[2]);
			int height = min(img.size().height - y, 2 * circ[2]);
			Rect bbox(x, y, width, height);
			// Create a black image
			Mat res;
			Mat res2;
			Mat stats, centroids;

			publish_vel_forward();

			// Copy only the image under the white circle to black image
			//gray.copyTo(res, mask);
			res = img(bbox);
			cvtColor(res, res, CV_BGR2GRAY);
			Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));

			if (res.size().empty())
				continue;
			equalizeHist(res, res);
			erode(res, res, element);
			//dilate(res, res, element);
			threshold(res, res, thresh2, 255, 0);
			int ct = connectedComponentsWithStats(res, res2, stats, centroids, 4, CV_32S);
			int maxa = 0;
			int maxl = 1;

			imshow("crop", res);

			for (int k = 1; k < ct; k++)
				if (maxa < stats.at<int>(k, CC_STAT_AREA))
					maxl = k, maxa = stats.at<int>(k, CC_STAT_AREA);
			//cout << maxl << " " << << "\n";

			res2.forEach<Pixeli>([&res2, maxl](Pixeli &p, const int *position) -> void {
				if (res2.at<int>(position[0], position[1]) != maxl)
					res2.at<int>(position[0], position[1]) = 0;
				//cout << (int)position[0] << " " << (int)position[1] <<"\n";
			});

			Point centroid(circ[0] + centroids.at<double>(maxl, 0) - circ[2], circ[1] + centroids.at<double>(maxl, 1) - circ[2]);
			Point diff = Point(circ[0], circ[1]) - centroid;
			double dx = abs((double)diff.x / circ[2]);
			double dy = abs((double)diff.y / circ[2]);
			double r3 = sqrt(dx * dx + dy * dy);
			if (r3 > 0.085)
			{
				cout << "Ignoring candidate because of insuffcient symmetry of largest component\n";
				continue;
			}
			//cout << r3 << "\n"; //<< diff.x << diff.y << " " << centroid.x << " " << circ[0] << "\n";
			normalize(res2, res2, 0, 255, NORM_MINMAX, CV_32S, Mat());

			convertScaleAbs(res2, res2);
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);

			int maxCorners = 12;
			vector<Point2f> corners;
			double qualityLevel = 0.01;
			double minDistance = 10;
			int blockSize = 3, gradientSize = 3;
			bool useHarrisDetector = true;
			double k = 0.04;

			/// Copy the source image

			/// Apply corner detection
			goodFeaturesToTrack(res2,
								corners,
								maxCorners,
								qualityLevel,
								minDistance,
								Mat(),
								blockSize,
								gradientSize,
								useHarrisDetector,
								k);

			Point minxp(10000, 10000);
			Point minyp(10000, 10000);
			Point maxxp(-10000, -10000);
			Point maxyp(-10000, -10000);

			for (size_t i = 0; i < corners.size(); i++)
			{
				if (corners[i].x < minxp.x)
					minxp = corners[i];

				if (corners[i].x > maxxp.x)
					maxxp = corners[i];

				if (corners[i].y > maxyp.y)
					maxyp = corners[i];

				if (corners[i].y < minyp.y)
					minyp = corners[i];
			}

			Point c = (((minxp + maxxp) / 2) + ((minyp + maxyp) / 2)) / 2 - Point(circ[2], circ[2]);
			dx = abs((double)c.x / circ[2]);
			dy = abs((double)c.y / circ[2]);
			double r4 = sqrt(dx * dx + dy * dy);
			//cout << r4 << " " << minxp.x << " " << maxxp.x << " " << minyp.y << " " << maxyp.y << " "<< circ[2]	 << "\n";

			if (r4 > 0.085)
			{
				cout << "Ignoring candidate as bounding box diagnoals do not intersect at center\n";
				continue;
			}

			dilate(res2, res2, element);
			Mat res3;
			connectedComponentsWithStats(res2, res3, stats, centroids, 4, CV_32S);

			if (stats.at<int>(0, CC_STAT_AREA) == 0)
				continue;
			double area = stats.at<int>(1, CC_STAT_AREA) / (double)stats.at<int>(0, CC_STAT_AREA);

			if (!(area > 0.1 && area < 0.45))
			{
				cout << "Ignoring candidate as area does not lie in range\n";
				continue;
			}
			cout << area << "\n";
			imshow("", res);
			imshow("2", res2);

			cout << "Helipad detected! Starting track \n";

			if (!track(img, bbox, servo))
				cout << "Tracking lost! Restarting detection process. \n\n\n\n";
		}

		imshow("Hough Circle Transform Demo", img);
		waitKey(100);
	}
}