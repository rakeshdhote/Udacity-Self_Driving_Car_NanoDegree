#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    /**
      * Calculate the RMSE .
    */
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size

    if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for (unsigned int i = 0; i < estimations.size(); ++i) {

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array() * residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse / estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
    /**
      * Calculate a Jacobian
    */
    MatrixXd Hj(3, 4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //check division by zero
    double den = px * px + py * py;
    double sqrtden = sqrt(den);

    if (abs(den) <= 1e-15) {
        cout << " Error Division by zero" << endl;
        Hj << 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;

    } else {
        Hj << px / sqrtden, py / sqrtden, 0, 0,
                -py / den, px / den, 0, 0,
                py * (vx * py - vy * px) / pow(den, 1.5), px * (vy * px - vx * py) / pow(den, 1.5), px / sqrtden, py / sqrtden;
    }

    return Hj;

}
