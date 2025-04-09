import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, delta, v, l = info["x"], info["y"], np.deg2rad(info["yaw"]), info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l * np.cos(yaw)
        front_y = y + l * np.sin(yaw)
        vf = v / np.cos(delta)

        min_idx, min_dist = utils.search_nearest(self.path, (front_x, front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model

        path_yaw = np.arctan2(target[1] - front_y, target[0] - front_x)
        theta_e = utils.angle_norm(path_yaw - yaw)

        e = min_dist * np.sign(np.sin(theta_e))

        next_delta = np.arctan2(-self.kp * e, vf) + theta_e
        next_delta = utils.angle_norm(next_delta)

        return next_delta, target
