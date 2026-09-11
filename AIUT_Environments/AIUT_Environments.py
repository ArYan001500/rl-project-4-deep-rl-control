# In the Name of Allah
# Programmer : Maziar Palhang
# Last Edit : Sun. 1405/2/27

import numpy as np
import random
import math

def __version__():
    print('AIUT_Environments vesion 0.0.0.0.1')
    return

###########################################
# Mountain Car class
# added Sun. 1405/2/27
class MountainCar():
    def __init__(self):
        self.Tolerance = 1e-6
        self.reset()
        return

    def reset(self):
        self.terminated = False
        self.truncated = False
        self.p = random.uniform(-0.6,-0.4)
        self.v = 0
        self.stepsLeft = 200
        return [self.p, self.v]

    def step(self,a):
        if not a in [-1,0,1]:
            raise ValueError(f"invalid action value {a}")
            
        self.v = self.v + 0.001*a - 0.0025* math.cos(3*self.p)
        self.v = max(-0.07, min(self.v,0.07))
        self.p = self.p + self.v
        self.p = max(-1.2, min(self.p,0.5))
        self.reward = -1
        self.stepsLeft -= 1
        if abs(self.p - 0.5) < self.Tolerance:
            self.reward = 0
            self.terminated = True
        elif abs(self.p - (-1.2)) < self.Tolerance:
            self.v = 0
        if self.stepsLeft == 0:
            self.truncated = True
        return [self.p,self.v], self.reward, self.terminated, self.truncated
