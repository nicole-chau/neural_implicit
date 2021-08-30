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
    res_dir1 = 'res_dir1'
    if not os.path.exists(res_dir1):
        os.makedirs(res_dir1)


    ## training sdf
    # define square
    v_rec = np.float_([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    rectangle = Polygon(v_rec)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(rectangle.sdf, points_train)))
    plot_sdf_using_opencv(rectangle.sdf, device=None, filename='square.png')
    dataset_rec = PolygonSample(v=v_rec, id=0)

    # define rectangle1
    v_rec = np.float_([[-0.5, -0.5], [-0.5, 0.5], [1, 0.5], [1, -0.5]])
    rectangle = Polygon(v_rec)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(rectangle.sdf, points_train)))
    plot_sdf_using_opencv(rectangle.sdf, device=None, filename='rectangle1.png')
    dataset_rec = PolygonSample(v=v_rec, id=1)

    # define rectangle1
    v_rec = np.float_([[-1, -1], [-1, 1], [2, 1], [2, -1]])
    rectangle = Polygon(v_rec)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(rectangle.sdf, points_train)))
    plot_sdf_using_opencv(rectangle.sdf, device=None, filename='rectangle2.png')
    dataset_rec = PolygonSample(v=v_rec, id=2)

    # define rectangle1
    v_rec = np.float_([[-1.5, -1.5], [-1.5, 1.5], [2.5, 1.5], [2.5, -1.5]])
    rectangle = Polygon(v_rec)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(rectangle.sdf, points_train)))
    plot_sdf_using_opencv(rectangle.sdf, device=None, filename='rectangle3.png')
    dataset_rec = PolygonSample(v=v_rec, id=3)

    # define the circle shape
    circle = Circle(np.float_([0, 0]), 2)
    points_train = np.float_([[x_, y_] 
                    for y_ in  np.linspace(-3, 3, 40) 
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(circle.sdf, points_train)))
    plot_sdf_using_opencv(circle.sdf, device=None, filename='circle.png')
    dataset_circle = CircleSample(center_x=0,center_y=0,radius=2, id=4)

    # define triangle 
    v_tri = np.float_([[0, -2], [0, 2], [2, 0]])
    triangle = Polygon(v_tri)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(triangle.sdf, points_train)))
    plot_sdf_using_opencv(triangle.sdf, device=None, filename='triangle.png')
    dataset_tri = PolygonSample(v=v_tri, id=5)

    # define pentagon 
    v_pent = np.float_([[0, 2], [2, 0], [0, -2], [-2, -1], [-2, 1]])
    pentagon = Polygon(v_pent)
    points_train = np.float_([[x_, y_]
                    for y_ in np.linspace(-3, 3, 40)
                    for x_ in np.linspace(-3, 3, 40)])
    sdf_train = np.float_(list(map(pentagon.sdf, points_train)))
    plot_sdf_using_opencv(pentagon.sdf, device=None, filename='pentagon.png')
    dataset_pent = PolygonSample(v=v_pent, id=6)

    all_datasets = torch.utils.data.ConcatDataset([dataset_circle, dataset_rec, dataset_tri, dataset_pent])

    batch_size = int(1e4)
    dataloader = data_utils.DataLoader(
        all_datasets,
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

    lat_size = 128
    lat_vec = nn.init.normal_(torch.empty(lat_size), mean=0, std=0.01)

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
        
        if (epoch == 0 or ((epoch + 1) % 100 == 0)):
            filename = os.path.join(res_dir1, "res_"+str(epoch)+".png")
            train_name = "res_"+str(epoch)+".png"

            for j in range(7):
                plot_sdf_using_opencv(net.forward, device=device, id=j, filename=os.path.join(res_dir1, str(j)+"_"+train_name), is_net=True)

            lats = net.get_lat_vecs()
            index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6])
            if use_cuda:
                index = index.to(device)
            lat_vecs = lats(index)

            affinity = np.empty((len(lat_vecs), len(lat_vecs)))

            for i in range(len(lat_vecs)):
                for j in range(len(lat_vecs)):
                    affinity[i][j] = torch.dot(lat_vecs[i], lat_vecs[j]) / (torch.norm(lat_vecs[i]) * torch.norm(lat_vecs[j]))

            print(affinity)

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
            filename = os.path.join(res_dir1, "test_res_"+str(epoch)+".png")
            test_name = "test_res_"+str(epoch)+".png"
            # plot_sdf_using_opencv(net.forward, device=device, filename=filename, is_net=True)

            for j in range(7):
                plot_sdf_using_opencv(net.forward, device=device, id=j, filename=os.path.join(res_dir1, str(j)+"_"+train_name), is_net=True)

            lats = net.get_lat_vecs()
            index = torch.LongTensor([0, 1, 2, 3, 4, 5, 6])
            if use_cuda:
                index = index.to(device)
            lat_vecs = lats(index)

            affinity = np.empty((len(lat_vecs), len(lat_vecs)))

            for i in range(len(lat_vecs)):
                for j in range(len(lat_vecs)):
                    affinity[i][j] = torch.dot(lat_vecs[i], lat_vecs[j]) / (torch.norm(lat_vecs[i]) * torch.norm(lat_vecs[j]))

            print(affinity)

        writer.add_scalar('Loss/Train', total_loss.item(), epoch)
        writer.add_scalar('Loss/Test', test_total_loss.item(), epoch)

        print("Epoch:", epoch, "Loss:", total_loss.item(), "Test Loss:", test_total_loss.item())
