from turtle import forward
from typing import ForwardRef
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from zmq import device

def index_swap(even_length):
    """Return an index tensor with appropriate device for swapping the odd and even positions
    https://stackoverflow.com/questions/952914/how-do-i-make-a-flat-list-out-of-a-list-of-lists
    """
    assert even_length % 2 == 0
    
    end = even_length // 2
    index_list = [[2*i+1,2*i] for i in range(0,end)] #* [1,0], [3,2], [5,4] ... swapping odd and even
    flat_list = [item for sublist in index_list for item in sublist] 
    return flat_list

    # b = [[2*i+1,2*i] for i in range(0,5)] 
    # flat_list = [item for sublist in b for item in sublist] 
    # swap_index_2 = torch.tensor(flat_list, device = device)
        
def swap_convolution(x, convolution, in_length, out_length):

    in_swap_index = torch.tensor(index_swap(in_length), device = x.device, dtype=torch.long)
    out_swap_index = torch.tensor(index_swap(out_length), device = x.device, dtype=torch.long)
    x = x.index_select(dim=-1, index=in_swap_index) #* (Pa,Va,Pb,Vb) -> (Va,Pa,Vb,Pb)
    x = convolution(x)
    x = x.index_select(dim=-1, index=out_swap_index) #* Swap back 

    return x

def index_combine_AABB_into_ABAB(combine_length):

    assert combine_length % 2 == 0
    
    single_length = combine_length//2 #* e.g. 20
    a = [i for i in range(0,single_length)] #* [0,1,2,3...19]
    b = [i + single_length for i in range(0,single_length)] #* [20, 21, ..., 39]
    zip_list = [[x, y] for x,y in zip(a,b)] #* [0, 20], [1,21], ... [19, 39]
    flat_list_c = [item for sublist in zip_list for item in sublist]
    return flat_list_c

def combine_tensor_AABB_to_ABAB(A, B):

    assert A.size(-1) == B.size(-1)
    in_length = A.size(-1)
    C = torch.cat((A,B), axis = -1)
    index = torch.tensor(index_combine_AABB_into_ABAB(in_length*2), device = C.device, dtype=torch.long)
    C = C.index_select(dim=-1, index=index) #* 
    return C

##### 40to40 layers
class layer_PV_dilate_40to40(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.out_dim
        self.init_dim = config.init_dim

        self.conv_40To20_PaVaPbVbToPaPb123 = nn.Sequential(
            nn.ZeroPad2d((4,4,2,0)),
            nn.Conv2d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=(3,3), groups = self.init_dim, stride=(1,2), dilation=(1,4)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # (Pa1,Pa2,Pa3),(Pb1,Pb2,Pb3) ->(PA123, PB123): 3 level smoothed sell and buy prices, 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-4(3-1)-1]/2+1=0 => p=8 on one side
        ) 

        self.conv_40To20_PaVaPbVbToVaVb123 = nn.Sequential(
            nn.ZeroPad2d((4,4,2,0)),
            nn.Conv2d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=(3,3), groups = self.init_dim, stride=(1,2), dilation=(1,4)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # (Va1,Va2,Va3),(Vb1,Vb2,Vb3) ->(VA123, VB123): 3 level smoothed sell and buy prices, 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-4(3-1)-1]/2+1=0 => p=8 on one side
        )

    def forward(self, x_40):
        return combine_tensor_AABB_to_ABAB(self.conv_40To20_PaVaPbVbToPaPb123(x_40),
            swap_convolution(x_40, self.conv_40To20_PaVaPbVbToVaVb123, 40 ,20)) # dim 40 -> 40



class dim40_layers(nn.Module): # channel = 1 ->  channel = init_dim = 11
    def __init__(self, config):
        super().__init__()
        
        self.init_dim = config.init_dim
        self.out_dim = config.out_dim
        self.layer_1 = nn.LayerNorm(40)
        self.layer_2 = nn.LayerNorm(40)
        self.dilate_convs = nn.ModuleList(
            [layer_PV_dilate_40to40(config) for i in range(6)]
        )
        self.conv_max_pool = nn.Sequential(
            nn.ZeroPad2d((4,4,2,0)),
            nn.MaxPool2d(kernel_size=(3,3),stride = (1,2),dilation=(1,4))
        )

        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.init_dim, kernel_size=(1,1)),
            nn.SiLU(),
        )

        # self.pointwise_40 = nn.Sequential(
        #     nn.LayerNorm(40), 
        #     nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1),),
        #     nn.GELU(),
            
        #     nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1),),
        #     nn.GELU(),
        # )
        
    def forward(self, x_40):
        # for conv in self.dilate_convs:
        #     x = self.layer(conv(x)) + x  
        x_0 = self.conv_up(x_40)
        x_1 = self.dilate_convs[0](x_0)
        x_2 = self.dilate_convs[1](x_1)
        x_3 = self.dilate_convs[1](x_2)
        # x_3 = self.layer_1(self.dilate_convs[2](x_2)) + x_1
        x_4 = self.dilate_convs[3](x_3) 
        x_5 = self.dilate_convs[4](x_4)
        # x_6 = self.layer_2(self.dilate_convs[5](x_5)) + x_3
        x_6 = self.dilate_convs[4](x_5)
        # x_7 = self.dilate_convs[6](x_6)
        # x_8 = combine_tensor_AABB_to_ABAB(self.conv_max_pool(x_1),
        #     swap_convolution(x, self.conv_max_pool, 40 ,20))
        # x_9 = combine_tensor_AABB_to_ABAB(self.conv_max_pool(x_4),
        #     swap_convolution(x_4, self.conv_max_pool, 40 ,20))
        # x_10 = combine_tensor_AABB_to_ABAB(self.conv_max_pool(x_7),
        #     swap_convolution(x_7, self.conv_max_pool, 40 ,20))
        x_all = torch.cat((
            x_0, x_1, x_2, x_3, x_4, x_5,x_6, # , x_7,x_10 x_9 ,x_8
        ),dim=1)
        # x_all = self.pointwise_40(x_all)

        # assert x_all.size(1) == self.out_dim
        
        # assert x_all.size(-1) == 40
        
        return x_all 
##### 

##### 40to10 layers 
class Pa_40to10_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.out_dim
        self.init_dim = config.init_dim

        self.conv_40To20_PaVaPbVbToPaPb123 = nn.Sequential(
            nn.ZeroPad2d((4,4,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,4)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),
            
            # (Pa1,Pa2,Pa3),(Pb1,Pb2,Pb3) ->(PA123, PB123): 3 level smoothed sell and buy prices, 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-4(3-1)-1]/2+1=0 => p=8 on one side
        ) 

        self.conv_20To10_PaPb123ToPa = nn.Sequential(
            nn.ZeroPad2d((2,2,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,2)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            # (Pa1, Pb1),(Pa2, Pb2) ->(Pa123, Pa234): 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-2(3-1)-1]/2+1=0 => p=4 on one side
        )

    def forward(self, x_40):
        x_20 = self.conv_40To20_PaVaPbVbToPaPb123(x_40)
        return self.conv_20To10_PaPb123ToPa(x_20)# dim 40 -> 10

class Pb_40to10_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.out_dim
        self.init_dim = config.init_dim

        self.conv_40To20_PaVaPbVbToPaPb123 = nn.Sequential(
            nn.ZeroPad2d((4,4,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,4)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),
            # (Pa1,Pa2,Pa3),(Pb1,Pb2,Pb3) ->(PA123, PB123): 3 level smoothed sell and buy prices, 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-4(3-1)-1]/2+1=0 => p=8 on one side
        ) 

        self.conv_20To10_PaPb123ToPb = nn.Sequential(
            nn.ZeroPad2d((2,2,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,2)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),
            
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            # (Pa1, Pb1),(Pa2, Pb2) ->(Pa123, Pa234): 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-2(3-1)-1]/2+1=0 => p=4 on one side
        )

    def forward(self, x_40):
        x_20 = self.conv_40To20_PaVaPbVbToPaPb123(x_40)
        return swap_convolution(x_20, self.conv_20To10_PaPb123ToPb, 20 ,10) # dim 40 -> 10

class Va_40to10_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.out_dim
        self.init_dim = config.init_dim

        self.conv_40To20_PaVaPbVbToVaVb123 = nn.Sequential(
            nn.ZeroPad2d((4,4,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,4)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),
            # (Va1,Va2,Va3),(Vb1,Vb2,Vb3) ->(VA123, VB123): 3 level smoothed sell and buy prices, 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-4(3-1)-1]/2+1=0 => p=8 on one side
        ) 

        self.conv_20To10_VaVb123ToVa = nn.Sequential(
            nn.ZeroPad2d((2,2,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,2)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            # (Va1, Vb1),(Va2, Vb2) ->(Va123, Va234): 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-2(3-1)-1]/2+1=0 => p=4 on one side
        )

    def forward(self, x_40):
        x_20 = swap_convolution(x_40, self.conv_40To20_PaVaPbVbToVaVb123, 40 ,20)
        x_10 = self.conv_20To10_VaVb123ToVa(x_20)
        return x_10# dim 40 -> 10

class Vb_40to10_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.out_dim
        self.init_dim = config.init_dim

        self.conv_40To20_PaVaPbVbToVaVb123 = nn.Sequential(
            nn.ZeroPad2d((4,4,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,4)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),
            # (Va1,Va2,Va3),(Vb1,Vb2,Vb3) ->(VA123, VB123): 3 level smoothed sell and buy prices, 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-4(3-1)-1]/2+1=0 => p=8 on one side
        ) 

        self.conv_20To10_VaVb123ToVb = nn.Sequential(
            nn.ZeroPad2d((2,2,0,0)),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,3), groups = self.init_dim, stride=(1,2), dilation=(1,2)),
            nn.SiLU(),
            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            nn.SiLU(),
            # nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=(1,1)),
            # nn.SiLU(),

            nn.Conv2d(in_channels=self.out_dim, out_channels=self.init_dim, kernel_size=(1,1)),
            # (Va1, Vb1),(Va2, Vb2) ->(Va123, Va234): 
            # [n+2p-d(k-1)-1]/s+1 = n => [n+2p-2(3-1)-1]/2+1=0 => p=4 on one side
        )

    def forward(self, x_40):
        x_20 = swap_convolution(x_40, self.conv_40To20_PaVaPbVbToVaVb123, 40 ,20)
        x_10 = swap_convolution(x_20, self.conv_20To10_VaVb123ToVb, 20 ,10)
        return x_10# dim 40 -> 10


class dim40to10_PV_layers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_dim = config.out_dim 

        self.conv_max_pool = nn.MaxPool2d(kernel_size=(1,2),stride = (1,2))
        self.conv_avg_pool = nn.AvgPool2d(kernel_size=(1,2),stride = (1,2))
 
        # 3 AB dilate rate
        self.Pa_40to10_layer = Pa_40to10_layer(config)
        self.Pb_40to10_layer = Pb_40to10_layer(config)
        self.Va_40to10_layer = Va_40to10_layer(config)
        self.Vb_40to10_layer = Vb_40to10_layer(config)
 
        self.ffn = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(10),
            nn.Linear(10,30),
            nn.Dropout(config.cnn_dropout_prob),
            nn.SiLU(),
            nn.Linear(30,30),
            nn.Dropout(config.cnn_dropout_prob),
            nn.SiLU(),
            nn.Linear(30,30),
            nn.SiLU(),
            nn.Dropout(config.cnn_dropout_prob),
            nn.Linear(30,4),
        ) for i in range (4)])

        self.mlp_proj = nn.ModuleList([nn.Sequential(
            nn.Linear(10,10),
            nn.Dropout(config.cnn_dropout_prob),
            nn.SiLU(),
            nn.Linear(10,10),
            nn.Dropout(config.cnn_dropout_prob),
            nn.SiLU(),
            nn.Linear(10,4),
            # nn.SiLU(), 
        ) for i in range (4)])
 
        # self.conv_up_10to20s = nn.ModuleList([nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size = (1,2), stride=(1,2)),
        #     nn.GELU(),
        #     # nn.LayerNorm(20),  
        # ) for i in range (2)])
      
    def forward(self, x_40): 
        Pa = self.Pa_40to10_layer(x_40)
        Pb = self.Pb_40to10_layer(x_40)
        Va = self.Va_40to10_layer(x_40)
        Vb = self.Vb_40to10_layer(x_40)

        Pa = self.ffn[0](Pa)
        Pb = self.ffn[1](Pb)
        Va = self.ffn[2](Va)
        Vb = self.ffn[3](Vb)

        return torch.cat((
            Pa, Pb, Va, Vb,
            # P,V 
        ),dim = 1)

class CNNExtractor(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.y_len = config.num_classes
        self.out_dim = config.out_dim
        self.num_classes = config.num_classes
        self.init_dim = config.init_dim
        
        self.conv_max_pool = nn.MaxPool2d(kernel_size=(1,2),stride = (1,2))
        self.conv_avg_pool = nn.AvgPool2d(kernel_size=(1,2),stride = (1,2))

        self.dim40_layers = dim40_layers(config=config)
        self.dim40to10_PV_layers = dim40to10_PV_layers(config=config)
 
        # self.dim20to5_layers = dim20to5_layers(config=config)

        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels=self.init_dim*4, out_channels=self.init_dim*4, kernel_size=(1,1), ),
            nn.SiLU(),
            nn.LayerNorm(4),

            nn.Conv2d(in_channels=self.init_dim*4, out_channels=self.init_dim*4, kernel_size=(1,1), ),
            nn.SiLU(),
            nn.LayerNorm(4),

            nn.Conv2d(in_channels=self.init_dim*4, out_channels=self.init_dim, kernel_size=(1,1), ),
        )
        
        self.mlp_time_smoothed = nn.Sequential(
            nn.LayerNorm(100),
            nn.Linear(100,100),
            nn.SiLU(),
            nn.Dropout(config.cnn_time_dropout),
            nn.Linear(100,100),
            nn.Dropout(config.cnn_time_dropout),
            nn.SiLU(),
            nn.Linear(100,100),
            nn.Dropout(config.cnn_time_dropout),
            nn.SiLU(),
        )

        #* Time2Vec implementation
        self.weights_cos = nn.Parameter(torch.Tensor(config.time_length))
        self.bias_cos = nn.Parameter(torch.Tensor(config.time_length))
        self.weights_sin = nn.Parameter(torch.Tensor(config.time_length))
        self.bias_sin = nn.Parameter(torch.Tensor(config.time_length))
        self.weights = nn.Parameter(torch.Tensor(config.time_length))
        self.bias = nn.Parameter(torch.Tensor(config.time_length))
    
    def seasonality(self, input_tensor):
        B,C,S,D = input_tensor.size(0), input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)
        x_permuted = torch.permute(input_tensor,(0,1,3,2)) #* [B, C, D, S]
        
        # time_vec_cos = torch.empty(input_tensor, device=input_tensor.device)
        # time_vec_cos = torch.cos(torch.mul(self.weights_cos, x_permuted) + self.bias_cos)
        # position_ids = torch.arange(S, device=input_tensor.device)
        x_permuted = self.weights*(torch.cos(torch.mul(self.weights_cos, x_permuted) + self.bias_cos) ) + self.bias
        
        out = torch.permute(x_permuted, (0,1,3,2))
        return out
        # time_vec_sin = torch.sin(torch.mul(self.weights_sin, x_permuted) + self.bias_sin)
        # self.weights * (time_vec_cos+time_vec_sin) + self.bias
        # Create a vector representing time
        
    def time_ffn(self,input_tensor):
        x_permuted = torch.permute(input_tensor,(0,1,3,2))
        x_permuted = self.mlp_time_smoothed(x_permuted)
        out_tensor = torch.permute(x_permuted, (0,1,3,2))
        return out_tensor
    
    def forward(self, input):
        x = input + self.time_ffn(input) # + self.seasonality(input)
        
        x = self.dim40_layers(x)
        x = self.dim40to10_PV_layers(x)
        # x = self.dim20to5_layers(x)
        x = self.bridge(x)

        N,C,H,W = x.size(0), x.size(1), x.size(2), x.size(3)
        x = x.reshape(N,  H, C*W)

        return x

def get_patches(self,image, patch_size, flatten_channels=True):
        """
        Inputs:
            image - torch.Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                              as a feature vector instead of a image grid.
        Output : torch.Tensor representing the sequence of shape [B,patches,patch_size*patch_size] for flattened.
        """
        # ==========================
        #* https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/7
        # ==========================
        kc, kh, kw = patch_size, patch_size ,patch_size # kernel size
        dc, dh, dw = patch_size, patch_size, patch_size # stride
        patches = image.unfold(2,kh,dh).unfold(3,kh,dh)
        # print(patches.size(), "patches unfold")
        patches = patches.contiguous().view(image.size(0), image.size(1), -1, kh, kw)
        # print(patches.size(), "patches tighten")
        patches = patches.permute(0, 2, 1, 3,4).contiguous()
        # print(patches.size(), "patches permute")
        

        if not flatten_channels:
            return patches
        if flatten_channels:
            output = torch.flatten(patches, 2)
            # print(output.size(), "patch flatten")
            return output
            
def scaled_dot_product_attention(query, key, value, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1,2)) / math.sqrt(dim_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)
    

class AttentionHeads(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim) # head_dim is dim for single head
        self.k = nn.Linear(embed_dim, head_dim) # [B,S,D] -> [B,S,H]
        self.v = nn.Linear(embed_dim, head_dim)
        
    def forward(self, hidden_state):
        seq_len = hidden_state.size(1)
        mask = torch.tril(torch.ones(seq_len,seq_len)).unsqueeze(0).to(device=hidden_state.device)
        attn_output = scaled_dot_product_attention(self.q(hidden_state), 
                                                   self.k(hidden_state), 
                                                   self.v(hidden_state),
                                                   mask = mask)
        return attn_output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHeads(embed_dim=embed_dim, head_dim=head_dim) for i in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state):
        x = torch.concat([single_head(hidden_state) for single_head in self.heads], dim = -1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout_1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_2 = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x): # hidden_size -> x
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        x = self.gelu(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        
    def forward(self, x):
        # Pre-norm
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x 
# layer_norm(): argument 'input' (position 1) must be Tensor, not NoneType
# forgot to return

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_CNN = CNNExtractor(config)
        self.positional_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps = 1e-12)
        self.dropout = nn.Dropout()
        
        #* Time2Vec implementation
        self.weights = nn.Parameter(torch.Tensor(config.time_length))
        self.bias = nn.Parameter(torch.Tensor(config.time_length))
        
    def forward(self, input): # input =  original series
        # Create Position ID
        seq_length = input.size(1) # [B, S, D]->[S]
        position_ids = torch.arange(seq_length, device=input.device, dtype=torch.long).unsqueeze(0)
        
        # Create token and positional embeddings
        token_embedding = self.token_embedding_CNN(input) # [B, S, hidden - 1]
        # token_embedding = input
        position_embeddings = self.positional_embeddings(position_ids)
        
        
        # Create a vector representing time
        time_ids = torch.where(position_ids == 0 , 
                               torch.mul(self.weights, position_ids) + self.bias, 
                               torch.cos(torch.mul(self.weights, position_ids) + self.bias)
        ).unsqueeze(-1)
        
        # Concat Time Vector 
        # embeddings = torch.concat((embeddings, time_ids), dim=-1)
        embeddings = token_embedding + position_embeddings
        # embeddings = self.layer_norm(embeddings)
        return embeddings
    
class LocalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head * heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        # why bias=False
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout) #? dropout on pointwise
        )
    
    def forward(self, fmap):
        shape, p = x.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x,y = map(lambda t: t //p, (x,y)) #? x, y is now # of group

        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p) #? p1 p2 is local map
        q,k,v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2,dim=-1) ) # in dim = c -> out_dim = (h d)
        # q,k,v = map(lambda t:rearrange(t,"b (h d) p1 p2 -> (b h) (p1 p2) d", h = h), (q,k,v))
        q,k,v = map(lambda t:rearrange(t,"(b x y) (h d) p1 p2 -> (b x y h) (p1 p2) d",  x = x, y = y, h = h), (q,k,v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum(' b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)',h = h, x = x, y = y, p1 = p, p2 = p )
        #? back to [b c h w]
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.layers)]
        )
        
    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerForSequenceClassfification(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classfier = nn.Linear(config.hidden_size, config.num_classes)
    
    def forward(self, x):
        x = self.encoder(x)[:,-1,:]
        x = self.dropout(x)
        x = self.classfier(x)
        # pred = torch.softmax(x, dim=1)
        return x
        
 


        
 
 
 
 
 
 
 
 
 
 
 
 
 
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
