#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */

  VectorXd rmse(4);
  rmse << 0,0,0,0;


  if (estimations.size() == 0 || estimations.size() != ground_truth.size()){
    cout << "CalculateRMSE() - Error - invalid estimations array size." << endl;
    return rmse;
  }

  //squared residuals
  VectorXd dif(4);
  for (int i = 0; i < estimations.size(); ++i){
    dif = estimations[i] - ground_truth[i];
    rmse = rmse.array() + dif.array() * dif.array(); 
  }

  rmse = rmse / estimations.size();

  rmse = rmse.array().sqrt();

  return rmse;

}
