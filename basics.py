## copy from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
 
import os
import numpy as np
# import matplotlib.pyplot as plt ## gpufarm does not support interactive visualization
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import argparse

class Geometry:
    EPS = 1e-12
    def distance_from_segment_to_point(a, b, p):
        ans = min(np.linalg.norm(a - p), np.linalg.norm(b - p))
        if (np.linalg.norm(a - b) > Geometry.EPS 
            and np.dot(p - a, b - a) > Geometry.EPS 
            and np.dot(p - b, a - b) > Geometry.EPS):
            ans = abs(np.cross(p - a, b - a) / np.linalg.norm(b - a))
        return ans


## parent class of other following shapes
class Shape:
    def sdf(self, p):
        pass
    
    
class Circle(Shape):
    def __init__(self, c, r):
        self.c = c
        self.r = r
    
    def sdf(self, p):
        return np.linalg.norm(p - self.c) - self.r
    
    
class Polygon(Shape):
    def __init__(self, v):
        self.v = v
    
    def sdf(self, p):
        return -self.distance(p) if self.point_is_inside(p) else self.distance(p)
    
    def point_is_inside(self, p):
        angle_sum = 0
        L = len(self.v)
        for i in range(L):
            a = self.v[i]
            b = self.v[(i + 1) % L]
            angle_sum += np.arctan2(np.cross(a - p, b - p), np.dot(a - p, b - p))
        return abs(angle_sum) > 1

    ## return all segments in the polygon to a point p             
    def distance(self, p):
        ans = Geometry.distance_from_segment_to_point(self.v[-1], self.v[0], p)
        for i in range(len(self.v) - 1):
            ans = min(ans, Geometry.distance_from_segment_to_point(self.v[i], self.v[i + 1], p))
        return ans


# default for is_net is false
def plot_sdf_using_opencv(sdf_func, device, filename=None, is_net=False):
    # See https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    
    ## this is the rasterization step that samples the 2D domain as a regular grid
    COORDINATES_LINSPACE = np.linspace(-4, 4, 100)
    y, x = np.meshgrid(COORDINATES_LINSPACE, COORDINATES_LINSPACE)
    if not is_net:
        z = [[sdf_func(np.float_([x_, y_])) 
                for y_ in  COORDINATES_LINSPACE] 
                for x_ in COORDINATES_LINSPACE]
    else:
        ## convert []
        z = [[sdf_func(torch.Tensor([x_, y_]).to(device)).detach().cpu() 
                for y_ in  COORDINATES_LINSPACE] 
                for x_ in COORDINATES_LINSPACE]
        
    z = np.float_(z)
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    
    ## TODO: use color to differentiate negative and positive
    z = (z - z_min) / (z_max - z_min) * 255
    z = np.uint8(z)
    z = cv2.applyColorMap(z, cv2.COLORMAP_JET)
    if filename is None:
        filename = "tmp_res.png"
    cv2.imwrite(filename, z)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.utils.weight_norm(nn.Linear(2, 512))
        self.fc2 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc3 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc4 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc5 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc6 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc7 = nn.utils.weight_norm(nn.Linear(512, 512))
        self.fc8 = nn.utils.weight_norm(nn.Linear(512, 1))
        
        # dropout with probability 0.2
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = F.relu(self.fc4(x))
        x = self.dropout(x)

        x = F.relu(self.fc5(x))
        x = self.dropout(x)

        x = F.relu(self.fc6(x))
        x = self.dropout(x)

        x = F.relu(self.fc7(x))
        x = self.dropout(x)

        x = torch.tanh(self.fc8(x))
        return x

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description='DeepSDF 2D Neural Implicit')
    # parser.add_argument('--shape', help='input shape type: circle, square, rectangle')

    ## make a folder for storing the output images
    res_dir = 'res_dir'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    ## training sdf
    # define the circle shape
    circle = Circle(np.float_([0, 0]), 2)
    # 2D points for training
    points_train = np.float_([[x_, y_] 
                    for y_ in  np.linspace(-3, 3, 40) 
                    for x_ in np.linspace(-3, 3, 40)])
    # sdf value at these 2d points 
    sdf_train = np.float_(list(map(circle.sdf, points_train)))
    # visualize the 2d points with sdf values
    plot_sdf_using_opencv(circle.sdf, device=None, filename='circle.png')
    # plt.scatter(points_train[:,0], points_train[:,1], color=(1, 1, 1, 0), edgecolor="#000000")


    # define rectangle
    # rectangle = Polygon(np.float_([[-1, -1], [-1, 1], [2, 1], [2, -1]]))
    # points_train = np.float_([[x_, y_]
    #                 for y_ in np.linspace(-3, 3, 40)
    #                 for x_ in np.linspace(-3, 3, 40)])
    # sdf_train = np.float_(list(map(rectangle.sdf, points_train)))
    # plot_sdf_using_opencv(rectangle.sdf, device=None, filename='rectangle.png')

    # define triangle
    # triangle = Polygon(np.float_([[0, -1], [0, 1], [1, 0]]))
    # points_train = np.float_([[x_, y_]
    #                 for y_ in np.linspace(-3, 3, 40)
    #                 for x_ in np.linspace(-3, 3, 40)])
    # sdf_train = np.float_(list(map(triangle.sdf, points_train)))
    # plot_sdf_using_opencv(triangle.sdf, device=None, filename='triangle.png')
    

    ## now we make the dataset and dataloader for training
    train_ds = TensorDataset(torch.Tensor(points_train), torch.Tensor(sdf_train))
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=len(train_ds))


    ## use cuda or not?
    use_cuda = torch.cuda.is_available()
    print("do you have cuda?", use_cuda)


    ## this is to instantiate a network defined
    net = Net()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print("device: ", device)
    net = net.to(device)

    ## this is to set an pytorch optimizer
    # opt = optim.SGD(net.parameters(), lr=1e-5)

    # use Adam optimizer
    opt = optim.Adam(net.parameters(), lr=1e-5)

    clamp_dist = 0.1

    ## main training process
    epochs = 1000
    for epoch in range(epochs):
        net.train() # set network to the train mode
        total_loss = 0 
        for points_b, sdfs_b in train_dl:
            # send points_b (a batch of points) to network; this is equivalent to net.forward(points_b)
            if use_cuda:
                points_b = points_b.to(device)
                sdfs_b = sdfs_b.to(device)
            pred = net(points_b)
            # reshape the pred; you need to check this torch function -- torch.squeeze() -- out
            pred = pred.squeeze()
            
            # compute loss for this batch with clamp
            pred = torch.clamp(pred, -clamp_dist, clamp_dist)
            sdfs_b = torch.clamp(sdfs_b, -clamp_dist, clamp_dist)
            loss = F.l1_loss(pred, sdfs_b)
            # aggregate losses in an epoch to check the loss for the entire shape
            total_loss += loss
            # backpropagation optimization 
            loss.backward()
            opt.step()
            # make sure you empty the grad (set them to zero) after each optimizer step.
            opt.zero_grad()
        
        # print("Epoch:", epoch, "Loss:", total_loss.item())
        
        if (epoch % 100 == 0):
            filename = os.path.join(res_dir, "res_"+str(epoch)+".png")
            plot_sdf_using_opencv(net.forward, device=device, filename=filename, is_net=True)

        test_total_loss = 0
        # set to evaluation mode
        net.eval()
        with torch.no_grad():
            for points_b, sdfs_b in train_dl:
                # send points_b (a batch of points) to network; this is equivalent to net.forward(points_b)
                if use_cuda:
                    points_b = points_b.to(device)
                    sdfs_b = sdfs_b.to(device)
                pred = net(points_b)
                # reshape the pred; you need to check this torch function -- torch.squeeze() -- out
                pred = pred.squeeze()

                # loss function
                pred = torch.clamp(pred, -clamp_dist, clamp_dist)
                sdfs_b = torch.clamp(sdfs_b, -clamp_dist, clamp_dist)
                test_loss = F.l1_loss(pred, sdfs_b)
                test_total_loss += test_loss

        if (epoch % 100 == 0):
            filename = os.path.join(res_dir, "test_res_"+str(epoch)+".png")
            plot_sdf_using_opencv(net.forward, device=device, filename=filename, is_net=True)

        print("Epoch:", epoch, "Loss:", total_loss.item(), "Test Loss:", test_total_loss.item())
                


            


        
    