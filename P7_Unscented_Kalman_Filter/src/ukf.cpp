#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

#define cutoff (0.0001)

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_ = false;
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 0.5; //0.5; //30;

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.55; //0.5 30;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    // Initialize CTRV model parameters
    n_x_ = 5;     // px, py, v, phi, phidot
    n_aug_ = 7;   // px, py, v, phi, phidot, mu_acc, mu_yawdd
    lambda_ = 3 - n_aug_;
    n_sigma_ = (2 * n_aug_) + 1;

    // predicted sigma points
    Xsig_pred_ = MatrixXd(n_x_, n_sigma_); 
    Xsig_pred_.fill(0.0);

    weights_ = VectorXd(n_sigma_);
    weights_.segment(1, 2 * n_aug_).fill(0.5d / (n_aug_ + lambda_));
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    // Update the process noise
    Q_ = MatrixXd(2, 2);
    Q_ << std_a_ * std_a_, 0.0,
            0.0, std_yawdd_ * std_yawdd_;

    // The Laser measurement noise 
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << std_laspx_ * std_laspx_, 0.0,
            0.0, std_laspy_ * std_laspx_;

    // The radar measurement noise 
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_ * std_radr_, 0.0, 0.0,
            0.0, std_radphi_ * std_radphi_, 0.0,
            0.0, 0.0, std_radrd_ * std_radrd_;

    // The state vector X 
    x_ = VectorXd(n_x_);

    // The augmented sigma points 
    Xsig_ = MatrixXd(n_aug_, n_sigma_);

    // The state covariance matrix 
    P_ = MatrixXd(n_x_, n_x_);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /** Make sure you switch between lidar and radar
    measurements.
    */

    /* If this is the first measurement */
    if (!is_initialized_) {

        // Initialize the covariance 
        P_ << 1, 0, 0, 0, 0,
                0, 1, 0, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1;

        // Initialize the state 
        double px, py, v;
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            double r = meas_package.raw_measurements_[0];
            double phi = meas_package.raw_measurements_[1];
            double rdot = meas_package.raw_measurements_[2];

            px = r * cos(phi);
            py = r * sin(phi);
            v = sqrt(rdot * cos(phi) * rdot * cos(phi) + rdot * sin(phi) * rdot * sin(phi));

        } else {
            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
            v = 0;
        }

        /* If initial values are zero they will set to an initial guess
         * and the uncertainty will be increased.
         * Initial zeros would cause the algorithm to fail when using only Radar data. */
        if (fabs(px) < cutoff) {
            px = cutoff;
        }
        if (fabs(py) < cutoff) {
            py = cutoff;
        }

        // Initialize the state 
        x_ << px, py, v, 0, 0;

        // Set the time 
        time_us_ = meas_package.timestamp_;

        // Set the flag 
        is_initialized_ = true;
    } else {
        // Calculate the time difference between current time and previous measurement */
        double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
        time_us_ = meas_package.timestamp_;

        /* Predict the X and P after delta_t time
         * Note: We don't have to worry about delta_t = 0, and more importantly, we should not skip predictions if delta_t = 0 since
         * we need to regenerate the sigma points after each measurement update. Or the second measurement would use old predicted
         * sigma points
         */
        Prediction(delta_t);

        // Update the posterior with the measurement */
        if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            UpdateLidar(meas_package);
        } else {
            UpdateRadar(meas_package);
        }
    }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /** Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    // Create augmented mean state
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(n_x_) = x_;
    x_aug(n_x_) = 0.0;
    x_aug(n_x_ + 1) = 0.0;

    // Create augmented covariance matrix
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0f);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug.bottomRightCorner(Q_.rows(), Q_.cols()) = Q_;

    // Square root of P
    MatrixXd A = P_aug.llt().matrixL();

    // Create augmented sigma points
    Xsig_.colwise() = x_aug;
    MatrixXd offset = A * sqrt(lambda_ + n_aug_);

    Xsig_.block(0, 1, n_aug_, n_aug_) += offset;
    Xsig_.block(0, n_aug_ + 1, n_aug_, n_aug_) -= offset;

    // Predict sigma points
    for (int i = 0; i < n_sigma_; i++) {

        double px = Xsig_(0, i);
        double py = Xsig_(1, i);
        double v = Xsig_(2, i);
        double yaw = Xsig_(3, i);
        double yawd = Xsig_(4, i);
        double nu_a = Xsig_(5, i);
        double nu_yawdd = Xsig_(6, i);

        // predict
        double px_p, py_p;

        if (fabs(yawd) > cutoff) {
            px_p = px + ((v / yawd) * (sin(yaw + (yawd * delta_t)) - sin(yaw)));
            py_p = py + ((v / yawd) * (cos(yaw) - cos(yaw + (yawd * delta_t))));
        } else {
            px_p = px + (v * delta_t * cos(yaw));
            py_p = py + (v * delta_t * sin(yaw));
        }

        double v_p = v;
        double yaw_p = yaw + (yawd * delta_t);
        double yawd_p = yawd;

        // adding noise
        px_p = px_p + (0.5d * nu_a * delta_t * delta_t * cos(yaw));
        py_p = py_p + (0.5d * nu_a * delta_t * delta_t * sin(yaw));
        v_p = v_p + (nu_a * delta_t);
        yaw_p = yaw_p + (0.5d * nu_yawdd * delta_t * delta_t);
        yawd_p = yawd_p + (nu_yawdd * delta_t);

        // Write predicted sigma point into the correct column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;

        // normalize angle
        Xsig_pred_(3, i) = normangle(Xsig_pred_(3, i));

        Xsig_pred_(4, i) = yawd_p;
    }

    // Compute the predicted state's mean
    x_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        x_ = x_ + (weights_(i) * Xsig_pred_.col(i));
    }

    // Compute the predicted state's covariance
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {

    // find differnce in state
     VectorXd x_diff = Xsig_pred_.col(i) - x_;

     // normalize angle
     x_diff(3) = normangle(x_diff(3));

     // calculate co-variance
     P_ = P_ + (weights_(i) * x_diff * x_diff.transpose());
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
    */
    MatrixXd Zsig = MatrixXd(2, n_sigma_);
    VectorXd z_pred = VectorXd(2);

    // Just copy over the rows 
    for (int i = 0; i < 2; i++) {
        Zsig.row(i) = Xsig_pred_.row(i);
    }

    // Compute the difference 
    z_pred = Zsig * weights_;

    // Measurement covariance matrix S 
    MatrixXd S = MatrixXd(2, 2);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        VectorXd residual = Zsig.col(i) - z_pred;
        S = S + (weights_(i) * residual * residual.transpose());
    }

    // add measurement noise covariance matrix 
    S = S + R_laser_;

    // Create matrix for cross correlation Tc 
    MatrixXd Tc = MatrixXd(n_x_, 2);
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        VectorXd tx = Xsig_pred_.col(i) - x_;
        VectorXd tz = Zsig.col(i) - z_pred;
        Tc = Tc + weights_(i) * tx * tz.transpose();
    }

    // Kalman filter K
    MatrixXd K = Tc * S.inverse();

    // residual
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // Update the state and covariance
    x_ = x_ + (K * z_diff);
    P_ = P_ - (K * S * K.transpose());

    // Compute the NIS 
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.
    */
    MatrixXd Zsig = MatrixXd(3, n_sigma_);
    Zsig.fill(0.0);

    // transform sigma points into measurement space 
    for (int i = 0; i < n_sigma_; i++) {

        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw) * v;
        double v2 = sin(yaw) * v;
        double sqrtterm = sqrt((p_x * p_x) + (p_y * p_y));

        // Measurement model
        Zsig(0, i) = sqrtterm;
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = ((p_x * v1) + (p_y * v2)) / sqrtterm;
    }

    // mean predicted measurement
    VectorXd z_pred = VectorXd(3);
    z_pred.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        z_pred = z_pred + (weights_(i) * Zsig.col(i));
    }

    // Measurement covariance matrix S
    MatrixXd S = MatrixXd(3, 3);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        // Residual 
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // angle normalization
        z_diff(1) = normangle(z_diff(1));

        // Update the measurement covariance
        S = S + (weights_(i) * z_diff * z_diff.transpose());
    }

    // Add measurement noise covariance matrix
    S = S + R_radar_;

    // Create & calculate matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, 3);
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {
        // Residual */
        VectorXd z_diff = Zsig.col(i) - z_pred;

        // Angle normalization 
        z_diff(1) = normangle(z_diff(1));

        // State difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // Angle normalization 
        x_diff(3) = normangle(x_diff(3));

        // Compute the cross correlation 
        Tc = Tc + (weights_(i) * x_diff * z_diff.transpose());
    }

    // Kalman gain K 
    MatrixXd K = Tc * S.inverse();

    // Residual 
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

    // Angle normalization
    z_diff(1) = normangle(z_diff(1));

    // Update state mean and covariance matrix
    x_ = x_ + (K * z_diff);
    P_ = P_ - (K * S * K.transpose());

    // Compute the NIS 
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

}

double UKF::normangle(double angle) {
    while (angle > M_PI) angle -= 2. * M_PI;
    while (angle < -M_PI) angle += 2. * M_PI;
    return angle;
}
