# controller_pid_basic.py

import sys
import numpy as np
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBasic(Controller):
    def __init__(self, kp=0.4, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0

    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None

        # Extract State
        x, y, yaw, dt = info["x"], info["y"], info["yaw"], info["dt"]

        # Search Nearest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        target = self.path[min_idx]
        
        # TODO: PID Control for Basic Kinematic Model 

        path_yaw = np.arctan2(target[1] - y, target[0] - x)
        yaw_error = utils.angle_norm(path_yaw - np.deg2rad(yaw))
        ep = np.sin(yaw_error) * min_dist

        self.acc_ep += ep * dt
        diff_ep = (ep - self.last_ep) / dt

        next_w = self.kp * ep + self.ki * self.acc_ep + self.kd * diff_ep
        self.last_ep = ep
        
        goal = self.path[-1]
        dist_to_goal = np.hypot(x - goal[0], y - goal[1])
        if dist_to_goal < 1.0:
            return 0.0, goal

        return next_w, target