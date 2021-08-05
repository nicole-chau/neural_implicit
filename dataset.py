from numpy.random import rand
import torch
import numpy as np
from geometry import Circle, Polygon, ComposedShape, plot_sdf_using_opencv
#import math
from math import sqrt

"""
circle function:
(x-cx)^2 + (y-cy)^2 = r^2
"""
class CircleSample(torch.utils.data.Dataset):
    def __init__(
        self,
        center_x:float,
        center_y:float,
        radius:float,
        num_base_pts=1e3,
        num_sample_pts=1e5,
        train_per_shape = False
    ):
        self.shape = Circle(np.float_([center_x, center_y]), radius)
        self.train_per_shape = train_per_shape
        
        ## get surface points
        rands = np.random.rand(int(num_base_pts))*2.0*3.142 ## map uniform dist [0, 1) to [0, 2Pi)
        x = center_x + radius*np.cos(rands)
        y = center_y + radius*np.sin(rands)
        self.surfpts = np.stack([x, y], axis=0) ## [2, N]
        
        ## generate sample points
        noise = np.random.randn(2, int(num_sample_pts)) ## standard normal dist
        random_indices = np.random.choice(range(0,int(num_base_pts)), int(num_sample_pts))
        self.samples = self.surfpts[:,random_indices] + noise*0.1
        self.sdf = self.shape.sdf(self.samples)
        self.sdf = np.clip(self.sdf, -1.0, 1.0)
        
        ## if we need this one
        self.data = np.concatenate([self.samples, np.expand_dims(self.sdf, axis=0)], axis=0)

    def draw(self):
        plot_sdf_using_opencv(self.shape.sdf, device=None, filename=self.filename)

    def __len__(self):
        if self.train_per_shape:
            return 1
        else:
            return self.data.shape[-1]

    def __getitem__(self, idx):
        if self.train_per_shape:
            return self.samples, self.sdf
        else:
            pt = torch.from_numpy(self.samples[:, idx]).to(torch.float)
            sdf = torch.Tensor([self.sdf[idx]]).to(torch.float)
            return pt, sdf


class PolygonSample(torch.utils.data.Dataset):
    def __init__(
        self,
        v,
        num_base_pts = 1e3,
        num_sample_pts = 1e5,
        train_per_shape = False
    ):
        self.shape = Polygon(v)
        self.train_per_shape = train_per_shape

        ## get surface points
        # calculate perimeter
        seg_length = []
        for i in range(len(v)):
            pt1 = v[i]
            if i == (len(v) - 1):
                pt2 = v[0]
            else:
                pt2 = v[i + 1]
            
            x1, y1 = pt1[0], pt1[1]
            x2, y2 = pt2[0], pt2[1]

            seg_length.append(sqrt((x2 - x1)**2 + (y2 - y1)**2))

        perimeter = sum(seg_length)        

        # array of random numbers between 0 and perimemter
        rands = np.random.rand(int(num_base_pts)) * perimeter ## map uniform dist [0, 1) to [0, perimeter)
        x = np.empty(0)
        y = np.empty(0)

        # walk perimeter to find point 
        for z in rands:
            idx, cum_length = 0, 0
            while cum_length < z:
                cum_length += seg_length[idx]
                idx += 1
            
            if idx == 0: # z == 0
                x = np.append(x, 0)
                y = np.append(y, 0)
            else:
                idx -= 1

                # find 2 points
                if idx == (0):
                    pt1, pt2 = v[0], v[1]
                elif idx == (len(v) - 1):
                    pt1, pt2 = v[-1], v[0]
                else:
                    pt1, pt2 = v[idx], v[idx + 1]

                # unit vectors to calculate surface point
                uv_x = (pt1[0] - pt2[0]) / seg_length[idx]
                uv_y = (pt1[1] - pt2[1]) / seg_length[idx]

                x = np.append(x, pt2[0] + uv_x * (cum_length - z))
                y = np.append(y, pt2[1] + uv_y * (cum_length - z))

        self.surfpts = np.stack([x, y], axis=0) ## [2, N]

        ## generate sample points
        noise = np.random.randn(2, int(num_sample_pts)) ## standard normal dist
        random_indices = np.random.choice(range(0,int(num_base_pts)), int(num_sample_pts))
        self.samples = self.surfpts[:,random_indices] + noise*0.1
        self.sdf = self.shape.sdf(self.samples)
        self.sdf = np.clip(self.sdf, -1.0, 1.0)
        
        ## if we need this one
        self.data = np.concatenate([self.samples, np.expand_dims(self.sdf, axis=0)], axis=0)


    def draw(self):
        plot_sdf_using_opencv(self.shape.sdf, device=None, filename=self.filename)
    
    def __len__(self):
        if self.train_per_shape:
            return 1
        else:
            return self.data.shape[-1]

    def __getitem__(self, idx):
        if self.train_per_shape:
            return self.samples, self.sdf
        else:
            pt = torch.from_numpy(self.samples[:, idx]).to(torch.float)
            sdf = torch.Tensor([self.sdf[idx]]).to(torch.float)
            return pt, sdf