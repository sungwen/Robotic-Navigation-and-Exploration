# controller_lqr_bicycle.py

import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBasic(Controller):
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

        # TODO: LQR Control for Basic Kinematic Model
        # Extract State
        x, y, yaw, delta, v, l, dt = info["x"], info["y"], np.deg2rad(info["yaw"]), info["delta"], info["v"], info["l"], info["dt"]

        # Search Nesrest Target
        lookahead = 10
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        target_idx = min(min_idx + lookahead, len(self.path) - 1)
        target = self.path[target_idx]

        # Compute heading error
        path_yaw = np.arctan2(target[1] - y, target[0] - x)
        yaw_error = utils.angle_norm(path_yaw - yaw)
        ep = min_dist * np.sign(np.sin(yaw_error))

        max_v = 5.0
        min_v = 1.0
        curvature_factor = np.clip(1 - abs(yaw_error) / np.pi, 0.0, 1.0)
        dynamic_v = min_v + (max_v - min_v) * curvature_factor

        # Update system matrix
        A = np.array([[1, dt, 0, 0],
                    [0, 0, dynamic_v, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 0]])

        B = np.array([[0], [0], [0], [dynamic_v / l]])
        X = np.array([[ep], [(ep - self.pe) / dt], [yaw_error], [(yaw_error - self.pth_e) / dt]])

        # LQR
        P = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P @ B) @ (B.T @ P @ A)
        u_star = -K @ X

        next_delta = u_star.item()
        self.pe, self.pth_e = ep, yaw_error
        
        goal = self.path[-1]
        dist_to_goal = np.hypot(x - goal[0], y - goal[1])
        if dist_to_goal < 1.0:
            return 0.0, goal

        return next_delta, target