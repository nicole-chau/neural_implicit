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

        print(self)

    def forward(self, x):
        pt = deepcopy(x)
        for idx, layer in enumerate(zip(self.mlp_list, self.actv_list)):
            fc, actv = layer
            if idx == self.skip_link_at:
                x = torch.cat([x, pt], dim=-1)
            
            print("idx: ", idx, x.shape)

            x = actv(fc(x))

            if idx != (len(self.mlp_list) - 1):
                x = self.dropout(x)

        return x
