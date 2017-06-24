#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
  int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
   uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {

          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"]; // waypoint x
          vector<double> ptsy = j[1]["ptsy"]; // waypoint y
          double px = j[1]["x"]; // car x
          double py = j[1]["y"]; // car y
          double psi = j[1]["psi"];
          double v = j[1]["speed"];
          double vmps = mph2mps(v);
          double steer_value = j[1]["steering_angle"];
          double throttle_value = j[1]["throttle"];

          // delayed state
          auto delayedX = px + vmps * cos(psi) * dt;
          auto delayedY = py + vmps * sin(psi) * dt;
          auto delayedPsi = psi + (vmps * steer_value / Lf) * dt;
          auto delayedV = vmps + throttle_value * dt;

          const int waypoints = ptsx.size();

          vector<double>  waypoints_xs(waypoints);
          vector<double>  waypoints_ys(waypoints);

          for (int i = 0; i < waypoints; ++i) {

            double dx = ptsx[i] - delayedX;
            double dy = ptsy[i] - delayedY;

            waypoints_xs[i] = dx * cos(delayedPsi) + dy * sin(delayedPsi);
            waypoints_ys[i] = dy * cos(delayedPsi) - dx * sin(delayedPsi);
          }

          // fit the polynomial
          auto coeffs = polyfit(Eigen::Map<Eigen::VectorXd>(&waypoints_xs[0], waypoints_xs.size()),
            Eigen::Map<Eigen::VectorXd>(&waypoints_ys[0], waypoints_ys.size()),
            3);
          // calculate cte
          double cte = polyeval(coeffs, 0); // coeffs[0]; //

          // calculate epsi
          double epsi = -atan(coeffs[1]);

          Eigen::VectorXd state(6);
          state << 0.0, 0.0, 0.0, delayedV, cte, epsi;

          auto vars = mpc.Solve(state, coeffs);

          json msgJson;
          
          // Divide by deg2rad(25) before you send the steering value back.
          msgJson["steering_angle"] = mpc.steer/(deg2rad(25)*Lf);
          msgJson["throttle"] = mpc.throttle;
          
          //Display the MPC predicted trajectory
          msgJson["mpc_x"] = mpc.x_pred;
          msgJson["mpc_y"] = mpc.y_pred;

          //Display the waypoints/reference line
          msgJson["next_x"] = waypoints_xs; //next_x_vals;
          msgJson["next_y"] = waypoints_ys; //next_y_vals;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << "\n" << std::endl;
          mpc.clear_prediction();
          
          // Latency
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });


  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
   size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
   char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
