"""
Kalman Filter (linear) — enhanced implementation.

Key improvements vs. a minimal textbook implementation:
- Robust gain computation (no explicit matrix inverse; Cholesky/solve based).
- Joseph-form covariance update for numerical stability.
- Input shaping/validation for x0 and z (accepts 1D or 2D inputs).
- Optional handling of missing measurements (z=None).
- Convenience methods: innovation, NIS/Mahalanobis distance, log-likelihood.

Notes:
- This is a *linear* Kalman filter (not EKF/UKF).
- Q and R defaults are placeholders; tune them for your motion/sensor model.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import LinAlgError


class KalmanFilter:
    def __init__(
        self,
        state_dim: int,
        measurement_dim: int,
        F: np.ndarray | None = None,
        H: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        P: np.ndarray | None = None,
        x0: np.ndarray | None = None,
        B: np.ndarray | None = None,
        dtype=np.float64,
    ):
        """
        Parameters
        ----------
        state_dim:
            Dimension of the state vector (n).
        measurement_dim:
            Dimension of the measurement vector (m).
        F:
            State transition matrix (n x n). Defaults to I.
        H:
            Measurement matrix (m x n). Defaults to [I_m | 0] only when m==n.
        Q:
            Process noise covariance (n x n). Defaults to small diagonal.
        R:
            Measurement noise covariance (m x m). Defaults to small diagonal.
        P:
            State covariance (n x n). Defaults to I.
        x0:
            Initial state (n,) or (n,1). Defaults to zeros.
        B:
            Optional control-input matrix (n x u). If provided, use predict(u=...).
        dtype:
            Numpy dtype for internal matrices.
        """
        self.state_dim = int(state_dim)
        self.measurement_dim = int(measurement_dim)

        if self.state_dim <= 0 or self.measurement_dim <= 0:
            raise ValueError("state_dim and measurement_dim must be positive integers.")

        self.F = np.array(F if F is not None else np.eye(self.state_dim), dtype=dtype, copy=True)
        if self.F.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"F must have shape {(self.state_dim, self.state_dim)}, got {self.F.shape}.")

        if H is None:
            if self.measurement_dim != self.state_dim:
                raise ValueError("Default H requires measurement_dim == state_dim.")
            H = np.eye(self.measurement_dim, self.state_dim)
        self.H = np.array(H, dtype=dtype, copy=True)
        if self.H.shape != (self.measurement_dim, self.state_dim):
            raise ValueError(f"H must have shape {(self.measurement_dim, self.state_dim)}, got {self.H.shape}.")

        # Defaults are placeholders; tune for your application.
        self.Q = np.array(Q if Q is not None else np.eye(self.state_dim) * 1e-3, dtype=dtype, copy=True)
        if self.Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q must have shape {(self.state_dim, self.state_dim)}, got {self.Q.shape}.")

        self.R = np.array(R if R is not None else np.eye(self.measurement_dim) * 1e-2, dtype=dtype, copy=True)
        if self.R.shape != (self.measurement_dim, self.measurement_dim):
            raise ValueError(f"R must have shape {(self.measurement_dim, self.measurement_dim)}, got {self.R.shape}.")

        self.P = np.array(P if P is not None else np.eye(self.state_dim), dtype=dtype, copy=True)
        if self.P.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"P must have shape {(self.state_dim, self.state_dim)}, got {self.P.shape}.")

        self.x = self._as_col(x0 if x0 is not None else np.zeros(self.state_dim, dtype=dtype), self.state_dim, "x0").astype(dtype, copy=False)

        self.B = None
        if B is not None:
            self.B = np.array(B, dtype=dtype, copy=True)
            if self.B.shape[0] != self.state_dim:
                raise ValueError(f"B must have {self.state_dim} rows; got shape {self.B.shape}.")

        # Cached innovation terms from the most recent update
        self._last_y = None
        self._last_S = None
        self._last_K = None

    @staticmethod
    def _as_col(v: np.ndarray, dim: int, name: str) -> np.ndarray:
        """Convert v to a (dim, 1) column vector."""
        a = np.asarray(v)
        if a.ndim == 1:
            if a.shape[0] != dim:
                raise ValueError(f"{name} must have length {dim}, got {a.shape[0]}.")
            return a.reshape(dim, 1)
        if a.ndim == 2 and a.shape == (dim, 1):
            return a
        if a.ndim == 2 and a.shape == (1, dim):
            return a.reshape(dim, 1)
        raise ValueError(f"{name} must have shape ({dim},), ({dim},1) or (1,{dim}); got {a.shape}.")

    def predict(self, u: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Time update (prediction).

        Parameters
        ----------
        u:
            Optional control input (u,) or (u,1). Requires B to be set.

        Returns
        -------
        x, P:
            Predicted state and covariance (returned as references to internal arrays).
        """
        if u is not None:
            if self.B is None:
                raise ValueError("Control input u provided but B is not set.")
            u_col = np.asarray(u)
            if u_col.ndim == 1:
                u_col = u_col.reshape(-1, 1)
            elif u_col.ndim == 2 and u_col.shape[1] != 1:
                raise ValueError("u must be a vector with shape (u,) or (u,1).")
            if self.B.shape[1] != u_col.shape[0]:
                raise ValueError(f"u has dim {u_col.shape[0]} but B expects {self.B.shape[1]}.")

            self.x = self.F @ self.x + self.B @ u_col
        else:
            self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry
        return self.x, self.P

    def innovation(self, z: np.ndarray, H: np.ndarray | None = None, R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute innovation y and innovation covariance S for a measurement z,
        without modifying the filter state.
        """
        Hm = self.H if H is None else np.asarray(H)
        Rm = self.R if R is None else np.asarray(R)

        z_col = self._as_col(z, self.measurement_dim, "z")
        y = z_col - (Hm @ self.x)
        S = Hm @ self.P @ Hm.T + Rm
        S = 0.5 * (S + S.T)  # enforce symmetry
        return y, S

    def update(self, z: np.ndarray | None, *, allow_singular: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Measurement update (correction).

        Parameters
        ----------
        z:
            Measurement vector (m,) or (m,1). If z is None, the update is skipped.
        allow_singular:
            If True, fall back to pseudo-inverse when S is not PD / ill-conditioned.
            If False, raise LinAlgError for near-singular cases.

        Returns
        -------
        x, P:
            Updated state and covariance (returned as references to internal arrays).
        """
        if z is None:
            # Useful for missed detections in tracking
            self._last_y = None
            self._last_S = None
            self._last_K = None
            return self.x, self.P

        y, S = self.innovation(z)

        # Prefer Cholesky for SPD S; fall back to solve/pinv if needed.
        PHt = self.P @ self.H.T

        K = None
        try:
            L = np.linalg.cholesky(S)
            # Solve S^{-1} * PHt.T via two triangular solves
            tmp = np.linalg.solve(L, PHt.T)
            K = np.linalg.solve(L.T, tmp).T
        except LinAlgError:
            # Cholesky failed: S not SPD (or numerically so). Try generic solve/cond check.
            cond_num = np.linalg.cond(S)
            if not np.isfinite(cond_num):
                raise LinAlgError("Residual covariance contains NaN/Inf and cannot be inverted/solved.")
            if (cond_num > 1e12) and not allow_singular:
                raise LinAlgError(f"Residual covariance is ill-conditioned (cond={cond_num:.2e}).")
            if allow_singular:
                Sinv = np.linalg.pinv(S)
                K = PHt @ Sinv
            else:
                # Use solve without forming inverse (may still raise if singular)
                K = np.linalg.solve(S.T, PHt.T).T

        # State update
        self.x = self.x + (K @ y)

        # Joseph-form covariance update (more stable than (I-KH)P)
        I = np.eye(self.state_dim, dtype=self.P.dtype)
        KH = K @ self.H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

        # Cache
        self._last_y = y
        self._last_S = S
        self._last_K = K
        return self.x, self.P

    # --- Diagnostics / convenience methods ---

    def nis(self, z: np.ndarray, *, allow_singular: bool = False) -> float:
        """
        Normalized Innovation Squared (NIS): y^T S^{-1} y (a Mahalanobis distance squared).
        Useful for gating/data association and for consistency checks.
        """
        y, S = self.innovation(z)
        try:
            L = np.linalg.cholesky(S)
            v = np.linalg.solve(L, y)
            return float((v.T @ v).item())
        except LinAlgError:
            if not allow_singular:
                raise
            Sinv = np.linalg.pinv(S)
            return float((y.T @ Sinv @ y).item())

    def log_likelihood(self, z: np.ndarray, *, allow_singular: bool = False) -> float:
        """
        Gaussian log-likelihood of measurement z under the current predicted state.
        """
        y, S = self.innovation(z)
        m = self.measurement_dim
        try:
            L = np.linalg.cholesky(S)
            v = np.linalg.solve(L, y)
            quad = float((v.T @ v).item())
            logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        except LinAlgError:
            if not allow_singular:
                raise
            # S may be singular; pseudo-determinant approximation
            U, s, _ = np.linalg.svd(S)
            s_clipped = np.clip(s, 1e-300, None)
            Sinv = (U * (1.0 / s_clipped)) @ U.T
            quad = float((y.T @ Sinv @ y).item())
            logdet = float(np.sum(np.log(s_clipped)))

        return -0.5 * (m * np.log(2.0 * np.pi) + logdet + quad)

    # --- Setters / getters ---

    def set_process_noise(self, Q: np.ndarray) -> None:
        Q = np.asarray(Q)
        if Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(f"Q must have shape {(self.state_dim, self.state_dim)}, got {Q.shape}.")
        self.Q = Q

    def set_measurement_noise(self, R: np.ndarray) -> None:
        R = np.asarray(R)
        if R.shape != (self.measurement_dim, self.measurement_dim):
            raise ValueError(f"R must have shape {(self.measurement_dim, self.measurement_dim)}, got {R.shape}.")
        self.R = R

    def set_initial_state(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        self.x = self._as_col(x0, self.state_dim, "x0")
        if P0 is not None:
            P0 = np.asarray(P0)
            if P0.shape != (self.state_dim, self.state_dim):
                raise ValueError(f"P0 must have shape {(self.state_dim, self.state_dim)}, got {P0.shape}.")
            self.P = P0

    def get_state(self) -> np.ndarray:
        """Return current state estimate as a 1D copy."""
        return self.x.reshape(-1).copy()

    def get_covariance(self) -> np.ndarray:
        """Return current covariance as a copy."""
        return self.P.copy()

    def last_update_terms(self):
        """
        Return (y, S, K) from the most recent update() call.
        Returns None entries if update was skipped or not yet called.
        """
        return self._last_y, self._last_S, self._last_K
