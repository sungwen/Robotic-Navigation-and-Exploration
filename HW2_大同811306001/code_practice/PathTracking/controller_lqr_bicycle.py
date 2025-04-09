import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBicycle(Controller):
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.R = R * 5000
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01):
        P = Q.copy()
        for _ in range(max_iter):
            Pn = A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # TODO: LQR Control for Bicycle Kinematic Model

        # Extract State
        x, y, yaw, delta, v, l, dt = info["x"], info["y"], np.deg2rad(info["yaw"]), info["delta"], info["v"], info["l"], info["dt"]

        # Search Nearest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        target = self.path[min_idx]

        path_yaw = np.arctan2(target[1] - y, target[0] - x)
        yaw_error = utils.angle_norm(path_yaw - yaw)
        ep = min_dist * np.sign(np.sin(yaw_error))

        # State-space representation
        A = np.array([[1, dt, 0, 0],
                      [0, 0, v, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 0]])

        B = np.array([[0], [0], [0], [v / l]])
        X = np.array([[ep], [(ep - self.pe) / dt], [yaw_error], [(yaw_error - self.pth_e) / dt]])

        P = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        u_star = -K @ X

        next_delta = u_star.item()
        self.pe, self.pth_e = ep, yaw_error

        return next_delta, target
