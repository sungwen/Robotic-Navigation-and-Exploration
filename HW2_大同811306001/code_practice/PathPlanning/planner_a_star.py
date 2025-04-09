import cv2
import sys
import numpy as np
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.g = {}
        self.f = {}
        self.goal_node = None

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        # Initialize 
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.f[start] = utils.distance(start, goal)

        # TODO: HW1 A* Algorithm 
        while self.queue:
            current = min(self.queue, key=lambda x: self.f[x])
            self.queue.remove(current)

            if utils.distance(current, goal) <= inter:
                self.goal_node = current
                break

            neighbors = [
                (current[0] + inter, current[1]),  # Right
                (current[0] - inter, current[1]),  # Left
                (current[0], current[1] + inter),  # Down
                (current[0], current[1] - inter),  # Up
            ]

            #neighbors = utils.neighbors(current, self.map.shape)
            for neighbor in neighbors:
                if self.map[neighbor[1], neighbor[0]] < 0.2:
                    continue

                tentative_g = self.g[current] + utils.distance(current, neighbor)
                
                if neighbor not in self.g or tentative_g < self.g[neighbor]:
                    self.parent[neighbor] = current
                    self.g[neighbor] = tentative_g
                    self.f[neighbor] = tentative_g + utils.distance(neighbor, goal)
                    if neighbor not in self.queue:
                        self.queue.append(neighbor)

        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path

        while p is not None:
            path.insert(0, p)
            p = self.parent[p]

        if path[-1] != goal:
            path.append(goal)

        return path
