import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPurePursuitBicycle(Controller):
    def __init__(self, kp=1, Lfc=25):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

    # State: [x, y, yaw, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, v, l = info["x"], info["y"], np.deg2rad(info["yaw"]), info["v"], info["l"]

        # Search Front Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        Ld = self.kp * v + self.Lfc

        target_idx = min_idx
        for i in range(min_idx, len(self.path) - 1):
            dist = np.linalg.norm(self.path[i + 1, :2] - np.array([x, y]))
            if dist > Ld:
                target_idx = i + 1
                break
        target = self.path[target_idx]

        # TODO: Pure Pursuit Control for Bicycle Kinematic Model

        alpha = np.arctan2(target[1] - y, target[0] - x) - yaw
        alpha = utils.angle_norm(alpha)

        next_delta = np.arctan2(2 * l * np.sin(alpha), Ld)

        return next_delta, target
