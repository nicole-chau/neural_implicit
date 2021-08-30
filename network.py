import torch
import torch.nn as nn
from copy import deepcopy

class Net(nn.Module):

    def __init__(self, skip_link_at = 4):
        super(Net, self).__init__()

        num_neurons = 512
        lat_size = 128

        self.skip_link_at = skip_link_at
        self.mlp_list = nn.ModuleList()
        self.actv_list = nn.ModuleList()
        self.mlp_list.append(nn.utils.weight_norm(nn.Linear(2 + lat_size, num_neurons)))
        self.actv_list.append(nn.ReLU(inplace=False))
        for i in range(1,7):
            if i == self.skip_link_at:
                self.mlp_list.append(nn.utils.weight_norm(nn.Linear(num_neurons+ 2 + lat_size, num_neurons)))
            else:
                self.mlp_list.append(nn.utils.weight_norm(nn.Linear(num_neurons, num_neurons)))
            self.actv_list.append(nn.ReLU(inplace=False))
        self.mlp_list.append(nn.utils.weight_norm(nn.Linear(num_neurons, 1)))
        self.actv_list.append(nn.Tanh())
        assert(len(self.mlp_list)==len(self.actv_list))

        self.dropout = nn.Dropout(0.2)

        self.embedding = torch.nn.Embedding(4, lat_size)

        print(self)

    def forward(self, x):
        lat_vecs = None
        if (len(x.shape) == 2 and x.shape[1] == 3):
            pt = deepcopy(x[:, 1:])
            id = x[:, :1]
            input = x[:, 1:]
            # get latent vectors
            lat_vecs = self.embedding(id.long())
            lat_vecs = lat_vecs.squeeze()
        elif (len(x.shape) == 3 and x.shape[2] == 3):
            pt = deepcopy(x[:, :, 1:])
            id = x[:, :, :1]
            input = x[:, :, 1:]
            # get latent vectors
            lat_vecs = self.embedding(id.long())
            lat_vecs = lat_vecs.squeeze()
        else:
            pt = deepcopy(x)
            input = x
    
        for idx, layer in enumerate(zip(self.mlp_list, self.actv_list)):
            fc, actv = layer

            # concatenate latent vector with (x, y)
            if (idx == 0 or idx == self.skip_link_at) and lat_vecs is not None:
                input = torch.cat([lat_vecs, input], dim=-1)
            
            if idx == self.skip_link_at:
                input = torch.cat([input, pt], dim=-1)

            # if idx == self.skip_link_at:
            #     input = torch.cat([input, pt], dim=-1)

            #print("input: ", input.shape, input)
            input = actv(fc(input))

            if idx != (len(self.mlp_list) - 1):
                input = self.dropout(input)

        return input

    def get_lat_vecs(self):
        return self.embedding

