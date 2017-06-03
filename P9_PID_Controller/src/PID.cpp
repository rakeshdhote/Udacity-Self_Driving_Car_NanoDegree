#include "PID.h"
#include "math.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  cte_previous = 0.0;
  cte_sum = 0.0;

  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;

}

double PID::UpdateError(double cte) {

  double steering = -Kp * cte - Kd * (cte_previous - cte) - Ki * cte_sum;
  double maxsteering = 1.0;
  double minsteering = -1.0;

    // if (steering > max_range)
    // {
    //     steering = max_range;
    // }
    // else if (steering < min_range)
    // {
    //     steering = min_range;
    // }

	// tanh
	steering = tanh(steering);

  // Update the cte history variables
  cte_previous = cte;
  cte_sum += cte;

  return steering;

}

double PID::TotalError() {
}

