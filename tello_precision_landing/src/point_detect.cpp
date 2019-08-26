#include "point_detect.h"
#include <cmath>

vector<Point2f> get_ordered_points(Mat croppedFrame, Point2f centroid)
{
    maxCorners = MAX(maxCorners, 1);
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 11, gradientSize = 5;
    bool useHarrisDetector = false;
    double k = 0.04;
    quad quad_types[3];

    goodFeaturesToTrack(croppedFrame,
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        Mat(),
                        blockSize,
                        gradientSize,
                        useHarrisDetector,
                        k);
    vector<Point2f> ret;
    //cout << "** Number of corners detected: " << corners.size() << endl;
    if (corners.size() != 12)
    {
        cout << "insufficient corners" << endl;
        return ret;
    }
    vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    for (int i = 0; i < 3; i++)
    {
        double maxAreaQuad = 0;
        for (auto j : getCombinations(indices, 12 - 4 * i, 4))
        {
            quad qd = {corners[j[0]], corners[j[1]], corners[j[2]], corners[j[3]]};
            double areaNew = quad_area(qd);
            if (maxAreaQuad < areaNew)
            {
                for (int n = 0; n < 4; n++)
                    quad_types[i].p[n] = qd.p[n];
                maxAreaQuad = areaNew;
            }
        }

        cout << "max quad area " << i << " " << maxAreaQuad << endl;

        for (int j = 0; j < indices.size(); j++)
        {
            for (int k = 0; k < 4; k++)
            {
                if (quad_types[i].p[k] == corners[indices[j]])
                {
                    indices.erase(indices.begin() + j);
                    j--;
                    break;
                }
            }
        }
        cout << "\n";
        Point2f a, b, c, d;
        vector<tuple<double, int>> agp;
        for (int k = 0; k < 4; k++)
        {
            quad_types[i].p[k] -= centroid;
            double ag = atan2(quad_types[i].p[k].y, quad_types[i].p[k].x);
            agp.push_back(make_tuple(ag, k));
        }
        sort(agp.begin(), agp.end());
        a = quad_types[i].p[get<1>(agp[0])];
        b = quad_types[i].p[get<1>(agp[1])];
        c = quad_types[i].p[get<1>(agp[2])];
        d = quad_types[i].p[get<1>(agp[3])];

        if((distance(a,b) < distance(a,d) && i<2) || (distance(a,b) > distance(a,d) && i==2))
        {
            ret.push_back(a + centroid);
            ret.push_back(b + centroid);
            ret.push_back(c + centroid);
            ret.push_back(d + centroid);
        }
        else
        {
            ret.push_back(b + centroid);
            ret.push_back(c + centroid);
            ret.push_back(d + centroid);
            ret.push_back(a + centroid);
        }

        /*ret.push_back(quad_types[i].p[0]);
        ret.push_back(quad_types[i].p[1]);
        ret.push_back(quad_types[i].p[2]);
        ret.push_back(quad_types[i].p[3]);*/
    }
    return ret;
}