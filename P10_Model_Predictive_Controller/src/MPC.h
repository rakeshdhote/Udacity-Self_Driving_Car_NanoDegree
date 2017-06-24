#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include <cppad/utility/vector.hpp>

using namespace std;

// Constants
const size_t N = 9; // number of timesteps
const double dt = 0.1; // timestep evaluation frequency
const double Lf = 2.67;

const double coeff_cte = 3.0;
const double coeff_epsi = 3.0;
const double coeff_v = 0.1;
const double coeff_delta = 160;
const double coeff_a = 0.2;
const double coeff_ddelta = 25.0;
const double coeff_da = 50.0;

static double mph2mps(double mph) {
  return mph * 0.44704;
}


class MPC {
public:
  MPC();

  virtual ~MPC();

  vector<double> x_pred;
  vector<double> y_pred;
  void clear_prediction();

  double steer;
  double throttle;

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

};

#endif /* MPC_H */
