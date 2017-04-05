#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    is_initialized_ = false;

    previous_timestamp_ = 0;

    //set the acceleration noise components
    noise_ax = 9;
    noise_ay = 9;

    // initializing matrices
    Hj_ = MatrixXd(3, 4);

    // Initialize the laser measurement noise matrix
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << 0.0225, 0,
            0, 0.0225;

    // Initialize the radar measurement noise matrix
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << 0.09, 0, 0,
            0, 0.0009, 0,
            0, 0, 0.09;

    // Laser measurement function
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0;

    //state covariance matrix P_
    ekf_.P_ = MatrixXd(4u, 4u);
    ekf_.P_ << 0.1f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.1f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.0f, 0.0f, 0.0f, 1.0f;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */

        // first measurement
        cout << "EKF: " << endl;
        ekf_.x_ = VectorXd(4);

        if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float r = measurement_pack.raw_measurements_[0];
            float phi = measurement_pack.raw_measurements_[1];
            float rdot = measurement_pack.raw_measurements_[2];

            ekf_.x_ << r * cos(phi), 
                       r * sin(phi), 
                       rdot * cos(phi), 
                       rdot * sin(phi);

        } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            ekf_.x_ << measurement_pack.raw_measurements_[0], 
                       measurement_pack.raw_measurements_[1], 
                       0, 
                       0;
        }

        previous_timestamp_ = measurement_pack.timestamp_;
        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    /*****************************************************************************
     *  Prediction
     ****************************************************************************/

    /**
       * Update the state transition matrix F according to the new elapsed time.
        - Time is measured in seconds.
       * Update the process noise covariance matrix.
       * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
     */

    //compute the time elapsed between the current and previous measurements
    float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
    previous_timestamp_ = measurement_pack.timestamp_;

  /* If the time differnce is too less, we need not predict again */
  if (fabs(dt) > 0.000001f)
  {
    // Define state transition matrix
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1.0, 0.0, dt, 0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0;

    //set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << pow(dt, 4) / 4.0 * noise_ax, 0.0, pow(dt, 3) / 2.0 * noise_ax, 0.0,
            0.0, pow(dt, 4) / 4.0 * noise_ay, 0.0, pow(dt, 3) / 2.0 * noise_ay,
            pow(dt, 3) / 2.0 * noise_ax, 0.0, pow(dt, 2) * noise_ax, 0.0,
            0.0, pow(dt, 3) / 2.0 * noise_ay, 0.0, pow(dt, 2) * noise_ay;

    ekf_.Predict();
  }
    /*****************************************************************************
     *  Update
     ****************************************************************************/

    /**
       * Use the sensor type to perform the update step.
       * Update the state and covariance matrices.
     */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
        ekf_.R_ = R_radar_;

        /* If the Jacobian is 0, for some reason, don't update */
        if (!ekf_.H_.isZero()) {
            ekf_.UpdateEKF(measurement_pack.raw_measurements_);
        }

    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

}
