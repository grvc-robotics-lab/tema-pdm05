# import numpy as np
#
#
# class KalmanFilter:
#     def __init__(self, state_dim, measurement_dim, F=None, H=None, Q=None, R=None, P=None, x0=None):
#         """
#         Initialize the Kalman Filter.
#         Parameters:
#             state_dim (int): Dimension of the state vector.
#             measurement_dim (int): Dimension of the measurement vector.
#             F (np.ndarray): State transition matrix (default: identity).
#             H (np.ndarray): Measurement matrix (default: identity).
#             Q (np.ndarray): Process noise covariance matrix (default: small identity matrix).
#             R (np.ndarray): Measurement noise covariance matrix (default: small identity matrix).
#             P (np.ndarray): Initial estimate uncertainty covariance matrix (default: identity).
#             x0 (np.ndarray): Initial state estimate (default: zero vector).
#         """
#         self.state_dim = state_dim
#         self.measurement_dim = measurement_dim
#
#         # State transition matrix
#         self.F = F if F is not None else np.eye(state_dim)
#
#         # Measurement matrix
#         if H is None:
#             if measurement_dim != state_dim:
#                 raise ValueError("Default H matrix initialization requires measurement_dim == state_dim.")
#             self.H = np.eye(measurement_dim, state_dim)
#         else:
#             self.H = H
#
#         # Process noise covariance
#         self.Q = Q if Q is not None else np.eye(state_dim) * 1e-6
#
#         # Measurement noise covariance
#         self.R = R if R is not None else np.eye(measurement_dim) * 1e-6
#
#         # Estimate uncertainty covariance
#         self.P = P if P is not None else np.eye(state_dim)
#
#         # Initial state estimate
#         self.x = x0 if x0 is not None else np.zeros((state_dim, 1))
#
#     def predict(self):
#         # Predicted state
#         self.x = np.dot(self.F, self.x)
#         # Predicted estimate covariance
#         self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
#
#     def update(self, z):
#         # Measurement residual
#         y = z - np.dot(self.H, self.x)
#         # Residual covariance
#         S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
#         # Check for numerical stability during inversion
#         if np.linalg.cond(S) < 1 / np.finfo(S.dtype).eps:
#             # Kalman gain
#             K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
#
#             # Updated state estimate
#             self.x = self.x + np.dot(K, y)
#
#             # Updated estimate covariance
#             I = np.eye(self.state_dim)
#             self.P = np.dot(I - np.dot(K, self.H), self.P)
#         else:
#             raise np.linalg.LinAlgError("Residual covariance matrix is near-singular.")
#
#     def set_process_noise(self, Q):
#         """
#         Update the process noise covariance matrix.
#         """
#         self.Q = Q
#
#     def set_measurement_noise(self, R):
#         """
#         Update the measurement noise covariance matrix.
#         """
#         self.R = R
#
#     def set_initial_state(self, x0, P0=None):
#         """
#         Set the initial state and covariance.
#         """
#         self.x = x0
#         if P0 is not None:
#             self.P = P0


import numpy as np
from numpy.linalg import LinAlgError

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim,
                 F=None, H=None, Q=None, R=None, P=None, x0=None):
        """
        A generic linear Kalman Filter.
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # F: State transition
        self.F = F if F is not None else np.eye(state_dim)

        # H: Measurement matrix
        if H is None:
            if measurement_dim != state_dim:
                raise ValueError("Default H requires measurement_dim == state_dim.")
            self.H = np.eye(measurement_dim, state_dim)
        else:
            self.H = H

        # Q: Process noise
        self.Q = Q if Q is not None else np.eye(state_dim) * 1e-8
        # R: Measurement noise
        self.R = R if R is not None else np.eye(measurement_dim) * 1e-8
        # P: Estimate uncertainty
        self.P = P if P is not None else np.eye(state_dim) * 1.0
        # x0: Initial state
        self.x = x0 if x0 is not None else np.zeros((state_dim, 1))

    def predict(self):
        # Predict next state
        self.x = self.F @ self.x
        # Predict next covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        # z should be shape (measurement_dim, 1)
        y = z - (self.H @ self.x)            # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        # Check for near-singular S
        cond_num = np.linalg.cond(S)
        if cond_num > 1 / np.finfo(S.dtype).eps:
            raise LinAlgError("Residual covariance is near-singular or ill-conditioned.")

        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        self.x = self.x + (K @ y)                # update state
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P       # update covariance

    def set_process_noise(self, Q):
        self.Q = Q

    def set_measurement_noise(self, R):
        self.R = R

    def set_initial_state(self, x0, P0=None):
        self.x = x0
        if P0 is not None:
            self.P = P0

    def get_state(self):
        """
        Returns the current state estimate as a 1D array.
        """
        return self.x.flatten()
