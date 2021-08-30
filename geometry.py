import numpy as np
import cv2
import torch

##
class Geometry:
    EPS = 1e-12
    def distance_from_segment_to_point(a, b, p):
        if len(p.shape) == 2:
            a_expand = a.reshape(2, 1)
            b_expand = b.reshape(2, 1)
            ans = np.minimum(np.linalg.norm(a_expand - p, axis=0), np.linalg.norm(b_expand - p, axis=0))

            num_col = np.shape(p)[1]

            for i in range(num_col):
                curr = p[:, i]
                if (np.linalg.norm((a - b)) > Geometry.EPS 
                and np.dot(curr - a, b - a) > Geometry.EPS 
                and np.dot(curr - b, a - b) > Geometry.EPS):
                    ans[i] = abs(np.cross(curr - a, b - a) / np.linalg.norm(b - a))
            return ans
        elif len(p.shape) == 1:
            ans = min(np.linalg.norm(a - p), np.linalg.norm(b - p))

            if (np.linalg.norm((a - b)) > Geometry.EPS 
                and np.dot((p - a), b - a) > Geometry.EPS 
                and np.dot((p - b), a - b) > Geometry.EPS):
                ans = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
            return ans
        else:
            raise NotImplementedError


## parent class of other following shapes
class Shape:
    def sdf(self, p):
        pass
    
    
class Circle(Shape):
    def __init__(self, c, r):
        self.c = c
        self.r = r
    
    def sdf(self, p):
        # print("p.shape: ", p.shape)
        if len(p.shape) == 2 or p.shape[0] > 2: ## 2d array
            c = self.c.reshape(2, 1)
            return np.linalg.norm(p - c, axis=0) - self.r
        elif len(p.shape) == 1: ## a point
            c = self.c
            return np.linalg.norm(p - c) - self.r
        else:
            raise NotImplementedError
    
    
class Polygon(Shape):
    def __init__(self, v):
        self.v = v
    
    def sdf(self, p):
        if len(p.shape) == 2: ## 2d array
            is_inside = self.point_is_inside(p)
            distance = self.distance(p)
            sign_array = np.where(is_inside, -1, 1)
            return distance * sign_array
        elif len(p.shape) == 1: ## a point
            return -self.distance(p) if self.point_is_inside(p) else self.distance(p)
        else:
            raise NotImplementedError
            
    def point_is_inside(self, p):
        if len(p.shape) == 2:
            num_col = np.shape(p)[1]
            angle_sum = np.zeros(num_col)
            is_inside = np.empty(num_col, dtype=bool)

            for i in range(num_col):
                curr = p[:, i]                
                L = len(self.v)
                for j in range(L):
                    a = self.v[j]
                    b = self.v[(j + 1) % L]
                    angle_sum[i] += np.arctan2(np.cross(a - curr, b - curr), np.dot(a - curr, b - curr))
                is_inside[i] = abs(angle_sum[i]) > 1
            return is_inside
        elif len(p.shape) == 1:
            angle_sum = 0
            L = len(self.v)
            for i in range(L):
                a = self.v[i]
                b = self.v[(i + 1) % L]
                angle_sum += np.arctan2(np.cross(a - p, b - p), np.dot(a - p, b - p))
            return abs(angle_sum) > 1
        else: 
            raise NotImplementedError

    ## return all segments in the polygon to a point p             
    def distance(self, p):
        ans = Geometry.distance_from_segment_to_point(self.v[-1], self.v[0], p)
        for i in range(len(self.v) - 1):
            if len(p.shape) == 2:
                ans = np.minimum(ans, Geometry.distance_from_segment_to_point(self.v[i], self.v[i + 1], p))
            elif len(p.shape) == 1:
                ans = min(ans, Geometry.distance_from_segment_to_point(self.v[i], self.v[i + 1], p))
            else:
                raise NotImplementedError
        return ans


class ComposedShape(Shape):
    def __init__(self, shapes):
        self.shapes = shapes

    ## p is a point; not a point set
    def sdf(self, p):
        sdfs = [s.sdf(p) for s in self.shapes]
        return np.min(sdfs)


def plot_sdf_using_opencv(sdf_func, device, id=-1, filename=None, is_net=False):
    # See https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    
    ## this is the rasterization step that samples the 2D domain as a regular grid
    COORDINATES_LINSPACE = np.linspace(-4, 4, 200)
    y, x = np.meshgrid(COORDINATES_LINSPACE, COORDINATES_LINSPACE)

    y = np.reshape(y, (y.shape[0], y.shape[1], 1))
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    if id != -1:
        id_arr = np.tile(id, (x.shape[0], x.shape[1], 1))
        id_x = np.concatenate((id_arr, x), axis=2)

    if not is_net:
        if id == -1: 
            z = [[sdf_func(np.float_([x_, y_])) 
                    for y_ in  COORDINATES_LINSPACE] 
                    for x_ in COORDINATES_LINSPACE]
        else:
            z = [[sdf_func(np.concatenate((id_x, y), axis=2))]]
            
    else:
        ## convert []
        if id == -1:
            z = [[sdf_func(torch.Tensor([x_, y_]).to(device)).detach().cpu().numpy() 
                    for y_ in  COORDINATES_LINSPACE] 
                    for x_ in COORDINATES_LINSPACE]
        else:
            z = [[sdf_func(torch.Tensor(np.concatenate((id_x, y), axis=2)).to(device)).detach().cpu().numpy()]]

    z = np.float_(z)
    z = np.reshape(z, (200, 200))

    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()

    pos = np.where(np.abs(z) < 0.03)
    
    z = (z - z_min) / (z_max - z_min) * 255
    z = np.uint8(z)
    z = cv2.applyColorMap(z, cv2.COLORMAP_JET)

    # add black pixels
    z[pos] = [0, 0, 0]

    if filename is None:
        filename = "tmp_res.png"
    
    print(filename)

    cv2.imwrite(filename, z)