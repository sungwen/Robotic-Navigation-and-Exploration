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
        self.h = {} # Distance from start to node
        self.g = {} # Distance from node to goal
        self.goal_node = None

    # 新增 euclidean_distance 方法，取代 utils.distance
    def euclidean_distance(self, node, goal):
        return np.sqrt((goal[0] - node[0])**2 + (goal[1] - node[1])**2)

    def planning(self, start=(100, 200), goal=(375, 520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = self.euclidean_distance(start, goal)
        while self.queue:
            # 選擇 f(n) = g(n) + h(n) 最小的節點
            current = min(self.queue, key=lambda n: self.g[n] + self.h[n])
            self.queue.remove(current)

            # 檢查當前節點是否為終點
            if current == goal:
                self.goal_node = current
                break

            # 尋找周邊 4 個方向的移動
            neighbors = [
                (current[0] + inter, current[1]),  # Right
                (current[0] - inter, current[1]),  # Left
                (current[0], current[1] + inter),  # Down
                (current[0], current[1] - inter),  # Up
            ]

            for neighbor in neighbors:
                # 確保鄰近節點不超出地圖範圍
                if not (0 <= neighbor[0] < self.map.shape[1] and 0 <= neighbor[1] < self.map.shape[0]):
                    continue

                # 確保節點不在障礙物內
                if self.map[neighbor[1], neighbor[0]] < 0.2:  # Adjusted threshold
                    continue

                # g(n) 計算與更新
                move_cost = 1.414 if (neighbor[0] != current[0] and neighbor[1] != current[1]) else 1
                new_g = self.g[current] + move_cost

                if neighbor not in self.g or new_g < self.g[neighbor]:
                    self.g[neighbor] = new_g
                    self.h[neighbor] = self.euclidean_distance(neighbor, goal)  # Use Euclidean heuristic
                    self.parent[neighbor] = current
                    self.queue.append(neighbor)

        # Extract the path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while True:
            path.insert(0, p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
