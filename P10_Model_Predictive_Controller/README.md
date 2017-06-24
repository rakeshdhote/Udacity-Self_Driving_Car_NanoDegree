# Self-Driving Car Engineer Nanodegree 
# Model Predictive Controller (MPC)
- - - 

## 1. Project Overview   
The objective of this project is to control a self-driving car autonomously via model predictive controller (MPC) around a simulator lake course. 

## 2. MPC  - Model   
A kintematic model is used for MPC. The following state, actutator and update equations are used:  

```
state = [x, y, $\psi$, v]
actuator = [$\delta$, a]
```


## 3. Setting up PID hyper parameters   



## Basic Build Instructions 
1. Clone this repo. 
2. Make a build directory:  `mkdir build && cd build` 
3. Compile:  `cmake .. && make` 
4. Run it:  `./mpc`. 
