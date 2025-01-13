import numpy as np


class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, F=None, H=None, Q=None, R=None, P=None, x0=None):
        """
        Initialize the Kalman Filter.
        Parameters:
            state_dim (int): Dimension of the state vector.
            measurement_dim (int): Dimension of the measurement vector.
            F (np.ndarray): State transition matrix (default: identity).
            H (np.ndarray): Measurement matrix (default: identity).
            Q (np.ndarray): Process noise covariance matrix (default: small identity matrix).
            R (np.ndarray): Measurement noise covariance matrix (default: small identity matrix).
            P (np.ndarray): Initial estimate uncertainty covariance matrix (default: identity).
            x0 (np.ndarray): Initial state estimate (default: zero vector).
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State transition matrix
        self.F = F if F is not None else np.eye(state_dim)

        # Measurement matrix
        if H is None:
            if measurement_dim != state_dim:
                raise ValueError("Default H matrix initialization requires measurement_dim == state_dim.")
            self.H = np.eye(measurement_dim, state_dim)
        else:
            self.H = H

        # Process noise covariance
        self.Q = Q if Q is not None else np.eye(state_dim) * 1e-3

        # Measurement noise covariance
        self.R = R if R is not None else np.eye(measurement_dim) * 1e-3

        # Estimate uncertainty covariance
        self.P = P if P is not None else np.eye(state_dim)

        # Initial state estimate
        self.x = x0 if x0 is not None else np.zeros((state_dim, 1))

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
        # Check for numerical stability during inversion
        if np.linalg.cond(S) < 1 / np.finfo(S.dtype).eps:
            # Kalman gain
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

            # Updated state estimate
            self.x = self.x + np.dot(K, y)

            # Updated estimate covariance
            I = np.eye(self.state_dim)
            self.P = np.dot(I - np.dot(K, self.H), self.P)
        else:
            raise np.linalg.LinAlgError("Residual covariance matrix is near-singular.")

    def set_process_noise(self, Q):
        """
        Update the process noise covariance matrix.
        """
        self.Q = Q

    def set_measurement_noise(self, R):
        """
        Update the measurement noise covariance matrix.
        """
        self.R = R

    def set_initial_state(self, x0, P0=None):
        """
        Set the initial state and covariance.
        """
        self.x = x0
        if P0 is not None:
            self.P = P0
