import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from VBMF import VBMF

def cp_decomposition_conv_layer(layer, rank):
    
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly.
    weights, factors= parafac(layer.weight.data.detach().cpu().numpy(), rank=rank, init='random')
    print("CP")
    last=factors[0]
    first=factors[1]
    vertical=factors[2]
    horizontal=factors[3]
    #last, first, vertical, horizontal = \
#        parafac(layer.weight.data.detach().cpu().numpy(), rank=rank, init='svd')
    
    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1], 
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data
    
    
    temp1=torch.from_numpy(np.flip(horizontal,axis=0).copy())
    temp2=torch.from_numpy(np.flip(vertical,axis=0).copy())
    temp3=torch.from_numpy(np.flip(first,axis=0).copy())
    temp4=torch.from_numpy(np.flip(last,axis=0).copy())
    
    
    depthwise_horizontal_layer.weight.data = \
        torch.transpose(temp1, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(temp2, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(temp3, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = temp4.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data
    tensor_array=weights.detach().cpu().numpy()
    unfold_0 = tl.base.unfold(tensor_array, 0) 
    unfold_1 = tl.base.unfold(tensor_array, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    #print(unfold_0)
    #print(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data.detach().cpu().numpy(), \
            modes=[0, 1], rank=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    temp1=torch.from_numpy(np.flip(first,axis=0).copy())
    temp2=torch.from_numpy(np.flip(last,axis=0).copy())
    temp3=torch.from_numpy(np.flip(core,axis=0).copy())
    #print(first_layer.weight.data.type())
    first_layer.weight.data = \
        torch.transpose(temp1, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = temp2.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = temp3

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)
