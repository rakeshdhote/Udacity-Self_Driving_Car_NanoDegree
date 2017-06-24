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

Lf = distance between the center of mass of the vehicle and the front wheels.  
```

The state is also augmented with cross track error (`cte`) and heading angle error (`epsi`). The `cte` defines offset from the centre of the road, while `epsi` defines difference in the ideal and acutal heading directions.  

## 2. Timestep Length and Elapsed Duration (N & dt)   





## 3. Polynomial Fitting and MPC Preprocessing   


## 4. Model Predictive Control with Latency  


## Basic Build Instructions 
1. Clone this repo. 
2. Make a build directory:  `mkdir build && cd build` 
3. Compile:  `cmake .. && make` 
4. Run it:  `./mpc`. 
