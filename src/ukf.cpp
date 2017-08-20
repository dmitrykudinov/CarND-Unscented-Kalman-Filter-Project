#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI / 8.0;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  
  NIS_radar_ = 0.0;
  NIS_lidar_ = 0.0;

  //state size
  n_x_ = 5;

  //augmented state size
  n_aug_ = 7;

  //sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //sigma points matrixes
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_ + 1);
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //init flag
  is_initialized_ = false;

  //weights initialization
  weights_ = VectorXd(2 * n_aug_ + 1);
  for (unsigned int i = 0; i < 2 * n_aug_ + 1; i++) //for all sigma points  
    weights_(i) = (i == 0? lambda_ : 0.5) / (lambda_ + n_aug_);

}

UKF::~UKF() {}

/**
 * Augmentation part
 */ 
void UKF::AugmentedSigmaPoints() {
  //create augmented mean state
  VectorXd x_aug = VectorXd(7);
  x_aug.fill(0.0);
  x_aug.head(5) = x_;
  
  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;
  
  //create square root matrix
  MatrixXd P_sqrt = P_aug.llt().matrixL();
  
  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  float mu_sqrt = sqrt(lambda_ + n_aug_);
  for (unsigned int i = 0; i < n_aug_; i++)
  {
    Xsig_aug_.col(i + 1) = x_aug + mu_sqrt * P_sqrt.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - mu_sqrt * P_sqrt.col(i);
  }  
}


/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  if (!is_initialized_) {
    //initialization 
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float ro_dot = meas_package.raw_measurements_[2];
      float vx = ro * cos(phi);
      float vy = ro * sin(phi);
      float v = sqrt(vx * vx + vy * vy);
      x_ << ro * cos(phi), ro * sin(phi), v, 0, 0;

      P_ <<	.5,0,0,0,0,
	          0,.5,0,0,0,
	          0,0,375,0,0,
	          0,0,0,500,0,
	          0,0,0,0,.1;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0],
                 meas_package.raw_measurements_[1],
                 0,
                 0,
                 0;
                 
      P_ <<	.9,0,0,0,0,
	          0,.9,0,0,0,
	          0,0,.9,0,0,
	          0,0,0,.9,0,
	          0,0,0,0,.9;
    }

    //time stamp
    previous_timestamp_ = meas_package.timestamp_;
    
    is_initialized_ = true;
    
    return;
  }
  
  //sicne the init part is done, let's process the measurememnt
  
  
  //PREDICTION
  
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  //creating augmented sigma points
  AugmentedSigmaPoints();
  
  //calling the Prediction routine
  Prediction(dt);
  
  
  //UPDATE
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
  else 
    UpdateLidar(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
  
  float delta_t_sq = delta_t * delta_t;
  
  //predict sigma points
  for (unsigned int i = 0; i < 2 * n_aug_ + 1; i++) {
    float px = Xsig_aug_(0,i);
    float py = Xsig_aug_(1,i);
    float v = Xsig_aug_(2,i);
    float yaw = Xsig_aug_(3,i);
    float yawd = Xsig_aug_(4,i);
    float noise_a = Xsig_aug_(5,i);
    float noise_yawdd = Xsig_aug_(6,i);

    float px_p, py_p;

    //checking for division by zero
    if (yawd != 0.0) {
        px_p = px + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    } else {
        px_p = px + v * delta_t * cos(yaw);
        py_p = py + v * delta_t * sin(yaw);
    }

    float v_p = v;
    float yaw_p = yaw + yawd * delta_t;
    float yawd_p = yawd;

    //adding noise
    float noise_coef = noise_a * delta_t_sq / 2;
    px_p = px_p + noise_coef * cos(yaw);
    py_p = py_p + noise_coef * sin(yaw);
    v_p = v_p + noise_a * delta_t;
    yaw_p = yaw_p + noise_yawdd * delta_t_sq / 2;
    yawd_p = yawd_p + noise_yawdd * delta_t;

    //results:
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //predicted state mean
  x_.fill(0.0);
  for (unsigned int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (unsigned int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0 * M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0 * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  */
  
  //only x and y for LIDAR package
  int n_z = 2; 
  
  //measurement
  VectorXd z = meas_package.raw_measurements_;

  //sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    VectorXd z_dif = Zsig.col(i) - z_pred;
    S = S + weights_(i) * z_dif * z_dif.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_laspx_ * std_laspx_, 0,
        0, std_laspy_ * std_laspy_;
  S = S + R;
  
  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_dif = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_dif(1) > M_PI) z_dif(1) -= 2.0 * M_PI;
    while (z_dif(1) < -M_PI) z_dif(1) += 2.0 * M_PI;

    // state difference
    VectorXd x_dif = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_dif(3) > M_PI) x_dif(3) -= 2.0 * M_PI;
    while (x_dif(3) < -M_PI) x_dif(3) += 2.0 * M_PI;

    Tc = Tc + weights_(i) * x_dif * z_dif.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_dif = z - z_pred;

  //angle normalization
  while (z_dif(1) > M_PI) z_dif(1) -= 2.0 * M_PI;
  while (z_dif(1) < -M_PI) z_dif(1) += 2.0 * M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_dif;
  P_ = P_ - K * S * K.transpose();
  
  //NIS Update
  NIS_lidar_ = z_dif.transpose() * S.inverse() * z_dif;
  
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  */
  
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  
  //measurement
  VectorXd z = meas_package.raw_measurements_;

  //sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //for all simga points
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double vx = v * cos(yaw);
    double vy = v * sin(yaw);

    // measurement model
    double ro = sqrt(px * px + py * py);
    Zsig(0,i) = ro;
    Zsig(1,i) = atan2(py, px);
    Zsig(2,i) = (px * vx + py * vy ) / ro;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  for (int i = 0; i < 2 * n_aug_+1; i++) //for all sigma points
    z_pred = z_pred + weights_(i) * Zsig.col(i);

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    //residual
    VectorXd z_dif = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_dif(1) > M_PI) z_dif(1) -= 2.0 * M_PI;
    while (z_dif(1) < -M_PI) z_dif(1) += 2.0 * M_PI;

    S = S + weights_(i) * z_dif * z_dif.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  R <<  std_radr_ * std_radr_, 0, 0,
        0, std_radphi_ * std_radphi_, 0,
        0, 0, std_radrd_ * std_radrd_;
  S = S + R;  

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    //residual
    VectorXd z_dif = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_dif(1) > M_PI) z_dif(1) -= 2.0 * M_PI;
    while (z_dif(1) < -M_PI) z_dif(1) += 2.0 * M_PI;

    // state difference
    VectorXd x_dif = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_dif(3) > M_PI) x_dif(3) -= 2.0 * M_PI;
    while (x_dif(3) < -M_PI) x_dif(3) += 2.0 * M_PI;

    Tc = Tc + weights_(i) * x_dif * z_dif.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_dif = z - z_pred;

  //angle normalization
  while (z_dif(1) > M_PI) z_dif(1) -= 2.0 * M_PI;
  while (z_dif(1) < -M_PI) z_dif(1) += 2.0 * M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_dif;
  P_ = P_ - K * S * K.transpose();
  
  //NIS Update
  NIS_lidar_ = z_dif.transpose() * S.inverse() * z_dif;
}
