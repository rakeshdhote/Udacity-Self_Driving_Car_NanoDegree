# Self-Driving Car Engineer Nanodegree 
# Model Predictive Controller (MPC)
- - - 

## 1. Project Overview   
The objective of this project is to control a self-driving car autonomously via model predictive controller (MPC) around a simulator lake course. 

## MPC - Model   
A kintematic model is used for MPC. The model is defined via the following state and actutators:  

```
state = [x, y, psi, v]
actuator = [delta, a]

x = X-position of the car
y = Y-position of the car
psi = car heading direction
v = velocity

delta = steering angle
a = accleration (1) or break/decleration (-1).
```

**State Update**  
```
x[t+1] = x[t] + v[t] * cos(psi[t]) * dt
y[t+1] = y[t] + v[t] * sin(psi[t]) * dt
psi[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
v[t+1] = v[t] + a[t] * dt

Lf = 2.67 m (given) = distance between the center of mass of the vehicle and the front wheels.  
```

The state is also augmented with cross track error (`cte`) and heading angle error (`epsi`). The `cte` defines offset from the centre of the road, while the `epsi` defines difference in the ideal and acutal heading directions.  

## 2. Hyperparameter Setting   

In order for the MPC to follow along the wayline, the cost function is defined as sum of weighted errors of cte, epsi, velocity, steering angle, accleration, change in steering angle, and change in throttle. The weights for the cost function components are defined as follows:  

```c++
const double coeff_cte = 3.0; 		// cte
const double coeff_epsi = 3.0;		// epsi
const double coeff_v = 0.1;		// velocity
const double coeff_delta = 160;		// steering angle
const double coeff_a = 0.2;		// accleration
const double coeff_ddelta = 25.0;	// change in steering angle
const double coeff_da = 50.0;		// change in accleration/throttle
```

Time step (`N`) and time steps (`dt`) are selected using trial and error approach for a reference velocity of 40 mph. Different values of `dt` are tested between 0.05 to 0.25 for `N` ranging from 6 to 20. For higher `N` and lower `dt`, the simulation was laggy because of the computaitonal overhead. Using visual inspection of how the car is moving, I setteled with the following values for `N` and `dt`:   

```c++
const size_t N = 9; 
const double dt = 0.1;
```

## 3. Polynomial Fitting and MPC Preprocessing   

A polynomial is fitted to waypoints.
If the student preprocesses waypoints, the vehicle state, and/or actuators prior to the MPC procedure it is described.

The waypoints for the simulator lake course are first transformed to vehicle coordinate system via transformation and rotation as  

```c++
double dx = ptsx[i] - delayedX;
double dy = ptsy[i] - delayedY;

waypoints_xs[i] = dx * cos(delayedPsi) + dy * sin(delayedPsi);
waypoints_ys[i] = dy * cos(delayedPsi) - dx * sin(delayedPsi);
```

## 4. Model Predictive Control with Latency  
The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency.  


## Basic Build Instructions 
1. Clone this repo. 
2. Make a build directory:  `mkdir build && cd build` 
3. Compile:  `cmake .. && make` 
4. Run it:  `./mpc`. 
