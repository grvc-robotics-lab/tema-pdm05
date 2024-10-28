import numpy as np


class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        # State transition matrix (assumes constant velocity model)
        self.F = np.eye(state_dim)

        # Measurement matrix
        self.H = np.eye(measurement_dim)

        # Process noise covariance (tune for your use case)
        self.Q = np.eye(state_dim) * 1e-5

        # Measurement noise covariance (tune for your use case)
        self.R = np.eye(measurement_dim) * 1e-5

        # Estimate uncertainty
        self.P = np.eye(state_dim)

        # Initial state estimate (longitude, latitude, elevation)
        self.x = np.zeros((state_dim, 1))

    def predict(self):
        # Predicted state
        self.x = np.dot(self.F, self.x)
        # Predicted estimate covariance
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Measurement residual
        y = z - np.dot(self.H, self.x)
        # Residual covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Updated state estimate
        self.x = self.x + np.dot(K, y)
        # Updated estimate covariance
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))