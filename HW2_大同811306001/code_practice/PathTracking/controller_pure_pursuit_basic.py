import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBasic(Controller):
    def __init__(self, kp=1, Lfc=10):
        self.path = None
        self.kp = kp      # look-ahead gain
        self.Lfc = Lfc    # look-ahead constant

    def feedback(self, info):
        if self.path is None:
            print("No path !!")
            return None, None

        x, y, yaw_deg, v = info["x"], info["y"], info["yaw"], info["v"]
        yaw = np.deg2rad(yaw_deg)

        # Look-ahead distance
        Ld = self.kp * v + self.Lfc

        # Find target point
        min_idx, _ = utils.search_nearest(self.path, (x, y))
        target_idx = min_idx
        for i in range(min_idx, len(self.path) - 1):
            dist = np.linalg.norm(self.path[i + 1, :2] - np.array([x, y]))
            if dist > Ld:
                target_idx = i + 1
                break
        target = self.path[target_idx]
        
        # TODO: Pure Pursuit Control for Basic Kinematic Model

        # Heading error
        alpha = np.arctan2(target[1] - y, target[0] - x) - yaw
        alpha = utils.angle_norm(alpha)

        # Angular velocity (Pure Pursuit formula)
        next_w = (2 * v * np.sin(alpha)) / Ld

        MAX_W = np.deg2rad(60)
        next_w = np.clip(next_w, -MAX_W, MAX_W)

        max_v = 1.0
        min_v = 0.5
        next_v = max(min_v, max_v * (1 - abs(next_w) / MAX_W))

        return next_v, next_w