#include <stdio.h>
#include <iostream>
#include "helpers.h"
#include "ibvs.h"
#include <eigen3/Eigen/SVD>

#include <eigen3/Eigen/src/QR/CompleteOrthogonalDecomposition.h>
//using Eigen::completeOrthogonalDecomposition
//sift parameters for detection(f2d)
double cx = 3.681653710406367850e+02;
double cy = 2.497677007139825491e+02;
float fx = 7.092159469231584126e+02;
float fy = 7.102890453175559742e+02;

float error_threshold = 0.1;
float depth = 1;
int lambda = 1; //for setting the extra speed of camera

vector<Point2d> reference;
void rqt_plot(double);
MatrixXf get_velocity_ibvs(vector<Point2f> points)
{

    MatrixXf velocity(4, 1);
    MatrixXf L1(1, 4);
    L1.resize(2 * reference.size(), 4);
    MatrixXf LInv(4, 1);
    LInv.resize(4, 2 * reference.size());

    MatrixXf s_ref(1, 1);
    s_ref.resize(2 * reference.size(), 1);
    MatrixXf s(1, 1);
    s.resize(2 * reference.size(), 1);
    int i = 0;
    for (int i = 0; i < reference.size(); i++)
    {
        s_ref(2*i, 0) = (reference[i].x - cx) / fx;
        s_ref(2*i + 1, 0) = (reference[i].y - cy) / fy;
    }

    auto pt = points;
    auto d = depth;

    for (int i = 0; i < reference.size(); i++)
    {
        s(2*i, 0) = (pt[i].x - cx) / fx;
        s(2*i + 1, 0) = (pt[i].y - cy) / fy;
    }

    for (int i = 0; i < reference.size(); i++)
    {
        float x = (pt[i].x - cx) / fx;
        float y = (pt[i].y - cy) / fy;
        L1(2*i, 0) = -1 / d;
        L1(2*i, 1) = 0;
        L1(2*i, 2) = x / d;
        L1(2*i, 3) = y;

        L1(2*i + 1, 0) = 0;
        L1(2*i + 1, 1) = -1 / d;
        L1(2*i + 1, 2) = y / d;
        L1(2*i + 1, 3) = -x;
    }
    Eigen::CompleteOrthogonalDecomposition<MatrixXf> c(L1);
    LInv = c.pseudoInverse();
    
    velocity = -LInv * (s - s_ref);
    rqt_plot((s-s_ref).norm());
    //velocity *= 0.1;
    
    cout << (s-s_ref).norm()<< " " << depth << "\n";
    cout << "ibvs out:" << velocity(0, 0) << " " << velocity(1, 0) << " " << velocity(2, 0) << " " << velocity(3, 0) << "\n";

    return velocity;
}
