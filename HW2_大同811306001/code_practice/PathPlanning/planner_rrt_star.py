import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerRRTStar(Planner):
    def __init__(self, m, extend_len=20):
        super().__init__(m)
        self.extend_len = extend_len
        self.goal_bias = 0.3
        self.smoothing_iterations = 15

    def _random_node(self, goal, shape, iteration):
        if np.random.rand() < min(0.9, self.goal_bias + iteration / 10000):
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        min_dist = float("inf")
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            if self.map[int(pts[1]), int(pts[0])]<0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])
        if extend_len > v_len:
            extend_len = v_len
        new_node = from_node[0]+extend_len*np.cos(v_theta), from_node[1]+extend_len*np.sin(v_theta)
        if new_node[1]<0 or new_node[1]>=self.map.shape[0] or new_node[0]<0 or new_node[0]>=self.map.shape[1] or self._check_collision(from_node, new_node):
            return False, None
        else:
            return new_node, utils.distance(new_node, from_node)


    def planning(self, start, goal, extend_len=None, img=None):
        if extend_len is None:
            extend_len = self.extend_len
        self.ntree = {start: None}
        self.cost = {start: 0}
        goal_node = None
        radius = extend_len * 2.5 
        for it in range(20000):
            #print("\r", it, len(self.ntree), end="")
            samp_node = self._random_node(goal, self.map.shape, it)
            near_node = self._nearest_node(samp_node)
            new_node, cost = self._steer(near_node, samp_node, extend_len)
            if new_node is False or new_node is None:
                continue
            best_parent = near_node
            min_cost = self.cost[near_node] + cost
            neighbors = [n for n in self.ntree if utils.distance(n, new_node) < radius]

            # TODO: HW1 Re-Parent & Re-Wire
            for n in neighbors:
                if n in self.cost:
                    temp_cost = self.cost[n] + utils.distance(n, new_node)
                    if temp_cost < min_cost and not self._check_collision(n, new_node):
                        best_parent = n
                        min_cost = temp_cost

            self.ntree[new_node] = best_parent
            self.cost[new_node] = min_cost

            for n in neighbors:
                if n in self.cost:
                    new_cost = self.cost[new_node] + utils.distance(new_node, n)
                    if new_cost < self.cost[n] and not self._check_collision(new_node, n):
                        self.ntree[n] = new_node
                        self.cost[n] = new_cost

            if utils.distance(new_node, goal) < extend_len:
                goal_node = new_node
                break

            # Draw
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0, 1, 0), 1)
                # Near Node
                if new_node:
                    img_ = img.copy()
                    cv2.circle(img_, utils.pos_int(new_node),5,(0,0.5,1),3)
                    # Draw Image
                    img_ = cv2.flip(img_, 0)
                    cv2.imshow("Path Planning", img_)
                    k = cv2.waitKey(1)
                    if k == 27:
                        break

        path = self.extract_path(goal_node, goal)
        path = self.smooth_path(path)
        return path

    # Extract Path
    def extract_path(self, goal_node, goal):
        path = []
        n = goal_node
        while n:
            path.insert(0, n)
            n = self.ntree[n]
        path.append(goal)
        return path

    def smooth_path(self, path):
        for _ in range(self.smoothing_iterations):
            if len(path) <= 2:
                break
            i, j = sorted(np.random.randint(0, len(path), size=2))
            if j - i > 1 and not self._check_collision(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return path
