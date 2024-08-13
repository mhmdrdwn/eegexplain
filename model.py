import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import networkx as nx
import copy

class ChebNetConv(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(ChebNetConv, self).__init__()

        self.K = k
        self.linear = nn.Linear(in_features*k, out_features)

    def forward(self, x: torch.Tensor, laplacian: torch.sparse_coo_tensor):
        x = self.__transform_to_chebyshev(x, laplacian)
        #print(x.shape)
        x = self.linear(x)
        #print(x.shape)
        return x

    def __transform_to_chebyshev(self, x, laplacian):
        cheb_x = x.unsqueeze(2)
        x0 = x
        if self.K > 1:
            x1 = torch.bmm(laplacian, x0)
            cheb_x = torch.cat((cheb_x, x1.unsqueeze(2)), 2)
            for _ in range(2, self.K):
                
                x2 = 2 * torch.bmm(laplacian, x1) - x0
                cheb_x = torch.cat((cheb_x, x2.unsqueeze(2)), 2)
                x0, x1 = x1, x2
        cheb_x = cheb_x.view(cheb_x.shape[0], cheb_x.shape[1], cheb_x.shape[2]*cheb_x.shape[3])
        return cheb_x

    
import torch.nn.functional as F

class ChebNetGCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_electrodes, out_channels, 
                 num_hidden_layers=2, dropout=0, k=3):
        super(ChebNetGCN, self).__init__()

        self.dropout = dropout

        self.input_conv = ChebNetConv(input_size, hidden_size, k)
        self.hidden_convs = nn.ModuleList([ChebNetConv(hidden_size, hidden_size, k) for _ in range(num_hidden_layers)])
        self.output_conv = ChebNetConv(hidden_size, out_channels, k)
        #self.conv = nn.Conv1d(in_channels=21, out_channels=21, kernel_size=3)

        #self.BN1 = nn.BatchNorm1d(out_channels*2)
        #self.fc = nn.Linear(out_channels*2, num_classes)
        
    def forward(self, x: torch.Tensor, A: torch.sparse_coo_tensor):
        #print(x.shape)
        #x = self.conv(x)
        #print(x.shape)
        x = F.relu(self.input_conv(x, A))
        #print(A.shape, x.shape)
        for conv in self.hidden_convs:
            #x = x.transpose(2, 1)
            x = F.relu(conv(x, A))
            
        #x = x.transpose(2, 1)
        x = self.output_conv(x, A)
        x = x.squeeze()
        return x


class CorrChebNetGCN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CorrChebNetGCN, self).__init__()
        
        self.model_Delta = ChebNetGCN(in_features, 32, 19, 32)
        self.model_Theta = ChebNetGCN(in_features, 32, 19, 32)
        self.model_Alpha = ChebNetGCN(in_features, 32, 19, 32)
        self.model_Beta = ChebNetGCN(in_features, 32, 19, 32)
        self.model_Gamma = ChebNetGCN(in_features, 32, 19, 32)
        self.fc = nn.Linear(158, num_classes)
        self.BN1 = nn.BatchNorm1d(158)
        self.conv = nn.Conv1d(in_channels=19, out_channels=10, kernel_size=3)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x: torch.Tensor, A: torch.Tensor):               
        out1 = self.model_Delta(x[:, :, :, 0], A[:, 0])
        out2 = self.model_Theta(x[:, :, :, 1], A[:, 1])
        out3 = self.model_Alpha(x[:, :, :, 2], A[:, 2])
        out4 = self.model_Beta(x[:, :, :, 3], A[:, 3])
        out5 = self.model_Gamma(x[:, :, :, 4], A[:, 4])
        
        out = torch.cat((out1, out2, out3, out4, out5), dim=-1)
        #out = out.reshape(out.shape[0], 21*80)
        #out = out.reshape(out.shape, out.shape[-1]*out.shape[-2])
        #print(out.shape)
        out = self.conv(out)
        #print(out.shape)
        batch =None
        out = global_mean_pool(out, batch)
        #out = torch.cat(out, axis=)
        #print(out.shape)
        out = self.BN1(out)
        out = self.fc(out)
        #out = self.softmax(out)
        #here we do bilinear pooling
        return out
    
class EnsembleGCN(nn.Module):
    def __init__(self, num_models, in_features, num_classes):
        super(EnsembleGCN, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.models = nn.ModuleList([CorrChebNetGCN(in_features=in_features, 
                                                    num_classes=num_classes) 
                                     for _ in range(num_models)])
        self.softmax = nn.Softmax(-1)
        
    def forward(self, x: torch.Tensor, A: torch.Tensor):
        #model = random.choice(self.models)
        out = torch.zeros(x.shape[0], self.num_classes)
        for model in self.models:
            out+=model(x, A)
        out /= (len(self.models))
        #out = self.softmax(out)
        return out