## copy from https://github.com/Oktosha/DeepSDF-explained/blob/master/deepSDF-explained.ipynb
## 

## 
import os
import numpy as np
# import matplotlib.pyplot as plt ## gpufarm does not support interactive visualization
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

# I move the geometry related functions and classes to the package file "geometry"
from geometry import Circle, Polygon, plot_sdf_using_opencv
# dataset file includes the CircleSample class which can return sample points on the circle
from dataset import CircleSample, PolygonSample

from torch.utils.data import TensorDataset, DataLoader

# Set the following to True will load the 8-layer MLP network
if True:
    from network import Net
else:
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 50)
            self.fc2 = nn.Linear(50, 1)

        def forward(self, x):
            # x = torch.Tensor(x)
            x = F.relu(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return x


if __name__ == "__main__":

    ## make a folder for storing the output images
    res_dir = 'res_dir'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    ## training sdf
    # define the circle shape
    # circle = Circle(np.float_([0, 0]), 2)
    # # 2D points for training
    # points_train = np.float_([[x_, y_] 
    #                 for y_ in  np.linspace(-3, 3, 40) 
    #                 for x_ in np.linspace(-3, 3, 40)])
    # # sdf value at these 2d points 
    # sdf_train = np.float_(list(map(circle.sdf, points_train)))
    # # visualize the 2d points with sdf values
    # plot_sdf_using_opencv(circle.sdf, device=None, filename='circle.png')
    # # plt.scatter(points_train[:,0], points_train[:,1], color=(1, 1, 1, 0), edgecolor="#000000")

    # # ## now we make the dataset and dataloader for training
    # # train_ds = TensorDataset(torch.Tensor(points_train), torch.Tensor(sdf_train))
    # # train_dl = DataLoader(train_ds, shuffle=True, batch_size=len(train_ds))
    
    # # I replace the above dataloader with the following one, using the customized dataset CircleSample
    # dataset = CircleSample(center_x=0,center_y=0,radius=2)
    # dataloader = data_utils.DataLoader(
    #     dataset,
    #     batch_size=int(1e4),
    #     shuffle=True,
    #     drop_last=False,
    # )

    # define rectangle
    # v = np.float_([[-1, -1], [-1, 1], [2, 1], [2, -1]])
    # rectangle = Polygon(v)
    # points_train = np.float_([[x_, y_]
    #                 for y_ in np.linspace(-3, 3, 40)
    #                 for x_ in np.linspace(-3, 3, 40)])
    # sdf_train = np.float_(list(map(rectangle.sdf, points_train)))
    # plot_sdf_using_opencv(rectangle.sdf, device=None, filename='rectangle.png')

    # define triangle 
    v = np.float_([[0, -2], [0, 2], [2, 0]])
    triangle = Polygon(v)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(triangle.sdf, points_train)))
    plot_sdf_using_opencv(triangle.sdf, device=None, filename='triangle.png')

    dataset = PolygonSample(v=v)
    dataloader = data_utils.DataLoader(
        dataset,
        batch_size=int(1e4),
        shuffle=True,
        drop_last=False,
    )


    ## use cuda or not?
    use_cuda = torch.cuda.is_available()
    print("do you have cuda?", use_cuda)


    ## this is to instantiate a network defined
    # You may change net to a 8-layer MLP for a better result (goto: Line21)
    net = Net()
    device = torch.device("cuda" if use_cuda else "cpu")    
    print("device: ", device)
    net = net.to(device)

    ## this is to set an pytorch optimizer
    # opt = optim.SGD(net.parameters(), lr=1e-5)
    opt = optim.Adam(net.parameters(), lr=1e-5)

    clamp_dist = 0.1

    writer = SummaryWriter()

    ## main training process
    epochs = 1000
    for epoch in range(epochs):
        net.train() # set network to the train mode
        total_loss = 0 
        for points_b, sdfs_b in dataloader:
            # send points_b (a batch of points) to network; this is equivalent to net.forward(points_b)
            if use_cuda:
                points_b = points_b.to(device)
                sdfs_b = sdfs_b.to(device)
            pred = net(points_b)
            # reshape the pred; you need to check this torch function -- torch.squeeze() -- out
            """
            """
            pred = pred.squeeze() 

            # compute loss for this batch
            """
            attention: this loss is different from eq.9 in DeepSDF
            """
            pred = torch.clamp(pred, -clamp_dist, clamp_dist)
            sdfs_b = torch.clamp(sdfs_b, -clamp_dist, clamp_dist)
            #pred = pred.reshape([10000, 1])
            sdfs_b = sdfs_b.squeeze()

            loss = F.l1_loss(pred, sdfs_b)
            # aggregate losses in an epoch to check the loss for the entire shape
            total_loss += loss
            # backpropagation optimization 
            loss.backward()
            opt.step()
            # make sure you empty the grad (set them to zero) after each optimizer step.
            opt.zero_grad()
        
        #print("Epoch:", epoch, "Loss:", total_loss.item())
        
        if (epoch % 100 == 0 or epoch == 10):
            filename = os.path.join(res_dir, "res_"+str(epoch)+".png")
            plot_sdf_using_opencv(net.forward, device=device, filename=filename, is_net=True)

        test_total_loss = 0
        # set to evaluation mode
        net.eval()
        with torch.no_grad():
            for points_b, sdfs_b in dataloader:
                # send points_b (a batch of points) to network; this is equivalent to net.forward(points_b)
                if use_cuda:
                    points_b = points_b.to(device)
                    sdfs_b = sdfs_b.to(device)
                pred = net(points_b)
                # reshape the pred; you need to check this torch function -- torch.squeeze() -- out
                # pred = pred.squeeze()
                #sdfs_b = sdfs_b.squeeze()
                # pred = pred.squeeze() 

                # loss function
                pred = torch.clamp(pred, -clamp_dist, clamp_dist)
                sdfs_b = torch.clamp(sdfs_b, -clamp_dist, clamp_dist)
                test_loss = F.l1_loss(pred, sdfs_b)
                test_total_loss += test_loss

        if (epoch == 0 or ((epoch + 1) % 100 == 0)):
            filename = os.path.join(res_dir, "test_res_"+str(epoch)+".png")
            plot_sdf_using_opencv(net.forward, device=device, filename=filename, is_net=True)
        
        writer.add_scalar('Loss/Train', total_loss.item(), epoch)
        writer.add_scalar('Loss/Test', test_total_loss.item(), epoch)

        print("Epoch:", epoch, "Loss:", total_loss.item(), "Test Loss:", test_total_loss.item())