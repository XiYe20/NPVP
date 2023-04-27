import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random


class Factorized3DConvAttn(nn.Module):
    def __init__(self,
        in_channels,
        atten_channels_downsample_ratio = 8,        #ratio for projecting input channels to key/query feature channels
        value_channels_downsample_ratio = 2,        #ratio for projecting input channels to value feature channels
        use_bias = True,                            #use bias for the key/query projection
        learn_gamma = True,                         #Learning the gamma of skip connection
        norm_layer_2d = nn.BatchNorm2d,
        norm_layer_1d = nn.BatchNorm1d,
        activ_func = nn.ReLU(),
        conv_first = True,                           #True for conv2d-attn2d-conv1d-attn1d, False for attn2d-conv2d-attn1d-conv1d
        learn_3d = True
    ):
        super().__init__()
        self.in_channels = in_channels

        self.spatial_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                                          norm_layer_2d(in_channels),
                                          activ_func)
        self.attn2d = NonLocalAttenion2D(in_channels, atten_channels_downsample_ratio, value_channels_downsample_ratio, True, learn_gamma,
                                         norm_layer_2d(in_channels), activ_func=activ_func)
        self.learn_3d = learn_3d
        self.temporal_conv = None
        self.attn1d = None
        if self.learn_3d:
            self.temporal_conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding='same', bias=use_bias),
                                            norm_layer_1d(in_channels),
                                            activ_func)
            self.attn1d = NonLocalAttenion1D(in_channels, atten_channels_downsample_ratio, value_channels_downsample_ratio, True, learn_gamma,
                                            norm_layer_1d(in_channels), activ_func=activ_func)

        self.conv_first = conv_first
        self.forward_func = self.conv_forward
        if not self.conv_first:
            self.forward_func = self.attn_forward
    
    def forward(self, x, T):
        return self.forward_func(x, T)

    def conv_forward(self, x, T):
        """
        x: (N*T, C, H, W)
        N: batch_size
        T: video clip length
        out: (N*T, C, H, W)
        """
        NT, C, H, W = x.size()
        N = NT//T

        skip = x
        x = self.spatial_conv(x) + x #(N*T, C, H, W)
        x = self.attn2d(x)

        if self.learn_3d:
            x = x.reshape(N, T, C, H, W).permute(0, 3, 4, 2, 1).flatten(0, 2) #(N*H*W, C, T)
            x = self.temporal_conv(x) + x
            x = self.attn1d(x)
            x = x.reshape(N, H, W, C, T).permute(0, 4, 3, 1, 2).flatten(0, 1)
        
        x = x + skip

        return x
    
    def attn_forward(self, x, T):
        """
        x: (N*T, C, H, W)
        N: batch_size
        T: video clip length
        out: (N*T, C, H, W)
        """
        NT, C, H, W = x.size()
        N = NT//T

        skip = x
        x = self.attn2d(x)
        x = self.spatial_conv(x) + x #(N*T, C, H, W)

        if self.learn_3d:
            x = x.reshape(N, T, C, H, W).permute(0, 3, 4, 2, 1).flatten(0, 2) #(N*H*W, C, T)
            x = self.attn1d(x)
            x = self.temporal_conv(x) + x
            
            x = x.reshape(N, H, W, C, T).permute(0, 4, 3, 1, 2).flatten(0, 1)
            
        x = x + skip

        return x


class NonLocalAttenion2D(nn.Module):
    """
    Based on https://github.com/brain-research/self-attention-gan/blob/master/non_local.py
    """
    def __init__(self, 
        in_channels,                                #Number of input feature channels
        atten_channels_downsample_ratio = 8,        #ratio for projecting input channels to key/query feature channels
        value_channels_downsample_ratio = 2,        #ratio for projecting input channels to value feature channels
        bias = True,                                #use bias for the key/query projection
        learn_gamma = True,                         #Learning the gamma of skip connection
        norm_func = None,
        activ_func = None
    ):
        super().__init__()
        self.bias = bias
        self.in_channels = in_channels
        self.attn_dim = in_channels//atten_channels_downsample_ratio
        self.value_dim = in_channels//value_channels_downsample_ratio

        self.Wq = nn.Linear(in_channels, self.attn_dim, bias = bias)
        self.Wk = nn.Linear(in_channels, self.attn_dim, bias = bias)
        self.Wv = nn.Linear(in_channels, self.value_dim, bias = bias)
        self.out_proj = nn.Linear(self.value_dim, in_channels, bias=bias)

        self.max_pool = nn.MaxPool2d((2, 2), stride = 2)
        self.learn_gamma = learn_gamma
        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(0., dtype=torch.float32))
        else:
            self.gamma = 1.0
        
        self.norm_func = nn.Identity()
        if norm_func is not None:
            self.norm_func = norm_func
        self.activ_func = nn.Identity()
        if activ_func is not None:
            self.activ_func = activ_func

        self._reset_parameters()

    def forward(self, x):
        """
        x: (N, C, H, W)
        out: (N, C, H, W)
        """
        N, C, H, W = x.size()
        skip = x

        x = x.flatten(2, 3).permute(0, 2, 1) #(N, H*W, C)

        query = self.Wq(x) #(N, H*W, self.attn_dim)
        key = self.Wk(x).reshape(N, H, W, self.attn_dim).permute(0, 3, 1, 2) #(N, self.attn_dim, H, W)
        #Downsample key length to be H/2*W/2
        key = self.max_pool(key).flatten(2, 3) #(N, self.attn_dim, H*W/4)

        attn_score = torch.matmul(query, key) #(N, H*W, H*W/4)
        attn_score = F.softmax(attn_score, dim = -1)

        value = self.Wv(x).reshape(N, H, W, self.value_dim).permute(0, 3, 1, 2) #(N, self.value_dim, H, W)
        #Downsample value length to be H/2*W/2
        value = self.max_pool(value).flatten(2, 3).permute(0, 2, 1) #(N, H*W/4, self.value_dim)

        attn_out = torch.matmul(attn_score, value) #attention operation, (N, H*W, self.value_dim)

        out = self.out_proj(attn_out) #output projection, (N, H*W, self.in_channels)
        out = out.reshape(N, H, W, self.in_channels).permute(0, 3, 1, 2)

        out = self.activ_func(self.norm_func(out))
        out = skip + self.gamma * out #Skip connection

        return out

    def _reset_parameters(self):
        if self.bias:
            nn.init.constant_(self.Wq.bias, 0.)
            nn.init.constant_(self.Wk.bias, 0.)
            nn.init.constant_(self.Wv.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

class NonLocalAttenion1D(nn.Module):
    """
    Based on https://github.com/brain-research/self-attention-gan/blob/master/non_local.py
    """
    def __init__(self, 
        in_channels,                                #Number of input feature channels
        atten_channels_downsample_ratio = 8,        #ratio for projecting input channels to key/query feature channels
        value_channels_downsample_ratio = 2,        #ratio for projecting input channels to value feature channels
        bias = True,                                #use bias for the key/query projection
        learn_gamma = True,
        norm_func = None,
        activ_func = None
    ):
        super().__init__()
        self.bias = bias
        self.in_channels = in_channels
        self.attn_dim = in_channels//atten_channels_downsample_ratio
        self.value_dim = in_channels//value_channels_downsample_ratio

        self.Wq = nn.Linear(in_channels, self.attn_dim, bias = bias)
        self.Wk = nn.Linear(in_channels, self.attn_dim, bias = bias)
        self.Wv = nn.Linear(in_channels, self.value_dim, bias = bias)
        self.out_proj = nn.Linear(self.value_dim, in_channels, bias=bias)

        self.learn_gamma = learn_gamma
        if learn_gamma:
            self.gamma = nn.Parameter(torch.tensor(0., dtype=torch.float32))
        else:
            self.gamma = 1.0
        
        self.norm_func = nn.Identity()
        if norm_func is not None:
            self.norm_func = norm_func
        self.activ_func = nn.Identity()
        if activ_func is not None:
            self.activ_func = activ_func

        self._reset_parameters()

    def forward(self, x):
        """
        x: (N, C, T)
        out: (N, C, T)
        """
        x = x.permute(0, 2, 1)
        N, T, C = x.size()

        query = self.Wq(x) #(N, T, self.attn_dim)
        key = self.Wk(x) #(N, T, self.attn_dim)

        attn_score = torch.matmul(query, key.permute(0, 2, 1)) #(N, T, T)
        attn_score = F.softmax(attn_score, dim = -1)

        value = self.Wv(x) #(N, T, self.value_dim)

        attn_out = torch.matmul(attn_score, value) #attention operation, (N, T, self.value_dim)
        out = self.out_proj(attn_out) #output projection, (N, T, self.in_channels)
        out = out.permute(0, 2, 1)
        out = self.activ_func(self.norm_func(out))
        out = x.permute(0, 2, 1) + self.gamma * out #Skip connection

        return out

    def _reset_parameters(self):
        if self.bias:
            nn.init.constant_(self.Wq.bias, 0.)
            nn.init.constant_(self.Wk.bias, 0.)
            nn.init.constant_(self.Wv.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)


class NRMLP(nn.Module):
    def __init__(self, out_channels, dim_x = 3, d_model = 256, MLP_layers = 4, scale = 10, fix_B = False, fuse_method = 'SPADE'):
        """
        Modified based on https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
        The output layer is moved to the "PosFeatFuser"
        """
        super().__init__()
        self.scale = scale
        self.dim_x = dim_x
        self.out_channels = out_channels
        self.MLP_layers = MLP_layers
        self.d_model = d_model
        self.fix_B = fix_B
        
        self.MLP = []

        self.mapping_fn = self.gaussian_mapping
        self.MLP.append(nn.Linear(2*self.d_model, self.d_model))
        if self.fix_B:
            self.register_buffer('B', torch.normal(mean = 0, std = 1.0, size = (self.d_model, self.dim_x)) * self.scale)
        else:
            #Default init for Linear is uniform distribution, would not produce a result as good as gaussian initialization
            #self.B = nn.Linear(self.dim_x, self.d_model, bias = False)
            self.B = nn.Parameter(torch.normal(mean = 0, std = 1.0, size = (self.d_model, self.dim_x)) * self.scale, requires_grad = True)
            
            #Init B with normal distribution or constant would produce much different result.
            #self.B = nn.Parameter(torch.ones(self.d_model, self.dim_x), requires_grad = True)
        
        self.MLP.append(nn.ReLU())
        for i in range(self.MLP_layers - 2):
            self.MLP.append(nn.Linear(self.d_model, self.d_model))
            self.MLP.append(nn.ReLU())
        
        self.MLP = nn.Sequential(*self.MLP)
        
        self.fuse_method = fuse_method

        self.mlp_beta = nn.Linear(self.d_model, out_channels)
        if self.fuse_method == 'SPADE':
            self.mlp_gamma = nn.Linear(self.d_model, out_channels)
        
    def forward(self, x):
        """
        Args:
            x: (N, d), N denotes the number of elements (coordinates)
        Return:
            out: (N, out_channels)
        """
        x = self.mapping_fn(x)
        x = self.MLP(x)
        beta = self.mlp_beta(x)
        if self.fuse_method == 'SPADE':
            gamma = self.mlp_gamma(x)
        else:
            gamma = torch.zeros_like(beta)
        
        return beta, gamma
        

    def gaussian_mapping(self, x):
        """
        Args:
            x: (N, d), N denotes the number of elements (coordinates)
            B: (m, d)
        """
        proj = (2. * float(math.pi) * x) @ self.B.T
        #proj = self.B(2. * float(math.pi) * x)
        out = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)

        return out

class CoorGenerator(nn.Module):
    def __init__(self, max_H, max_W, max_T):
        """
        Normalize the coordinated to be [0,1]
        """
        super().__init__()
        self.max_H = max_H
        self.max_W = max_W
        self.max_T = max_T
    
    def forward(self, t_list, h_list, w_list):
        """
        The h/w/t index starts with 0
        Args:
            h_list: list of h coordinates, Tensor with shape (H,)
            w_list: list of w coordinates, Tensor with shape (W,)
            t_list: list of t coordinates, Tensor with shape (T,)
        Returns:
            coor: Tensor with shape (T*H*W, 3), for the last dim, the coordinate order is (t, h, w)
        """
        assert torch.max(h_list) <= self.max_H and torch.min(h_list) >= 0., "Invalid H coordinates"
        assert torch.max(w_list) <= self.max_W and torch.min(w_list) >= 0., "Invalid W coordinates"
        assert torch.max(t_list) <= self.max_T and torch.min(t_list) >= 0., "Invalid T coordinates"

        norm_h_list = h_list/self.max_H
        norm_w_list = w_list/self.max_W
        norm_t_list = t_list/self.max_T

        hvv, wvv = torch.meshgrid(norm_h_list, norm_w_list)
        s_coor = torch.stack([hvv, wvv], dim=-1)
        t_coor = torch.ones_like(hvv)[None, :, :] * norm_t_list[:, None, None]

        s_coor = s_coor.unsqueeze(0).repeat(norm_t_list.shape[0], 1, 1, 1)
        coor = torch.cat([t_coor.unsqueeze(-1), s_coor], dim = -1)

        coor = coor.flatten(0, 2)

        return coor

class EventEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_layers, stochastic):
        super().__init__()
        self.stochastic = stochastic
        self.n_layers = n_layers
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels),
                                  nn.BatchNorm2d(in_channels),
                                  nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(hidden_channels),
                                  nn.ReLU(True))
        
        for i in range(n_layers):
            setattr(self, f'MLP_{i}', nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, stride=1, bias=False), 
                                                    nn.BatchNorm2d(hidden_channels),
                                                    nn.ReLU(True)))
        self.mu_net = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, bias=True)
        if self.stochastic:
            self.logvar_net = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        """
        x: (N, C, H, W) #the event coding
        Return:
            mu: (N, C, H, W)
            logvar: (N, C, H, W)
            z: (N, C, H, W)
        """
        x = self.conv2(self.conv1(x))
        for i in range(self.n_layers):
            x = getattr(self, f'MLP_{i}')(x)
        
        mu = self.mu_net(x)
        
        if self.stochastic:
            logvar = self.logvar_net(x)
            return self.reparameterize(mu, logvar), mu, logvar
        else:
            return mu

    def reparameterize(self, mu, logvar):
        eps = torch.randn(mu.shape, device = mu.device)
        return mu + torch.exp(0.5*logvar) * eps

class PosFeatFuser(nn.Module):
    def __init__(self, x_channels, param_free_norm_type = 'layer'):
        """
        Modified from https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
        There is no learned parameters in this module
        """
        super().__init__()

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(x_channels, affine=False)
        elif param_free_norm_type == 'syncbatch':
            self.param_free_norm = nn.SyncBatchNorm(x_channels, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(x_channels, affine=False)
        elif param_free_norm_type == 'layer':
            self.param_free_norm = nn.GroupNorm(1, x_channels, affine=False) #equivalent to layernorm
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
            
    def forward(self, x, pos_beta, pos_gamma):
        """
        Args:
            x: (N, T, H, W, C)
            pos_gamma: (T*H*W, C)
            pos_beta: (T*H*W, C)

        Return:
            out: (N, T, H, W, C)
        """

        # Part 1. generate parameter-free normalized activations
        x = x.permute(0, 1, 4, 2, 3) #(N, T, C, H, W)
        N, T, C, H, W = x.shape
        normalized = self.param_free_norm(x.flatten(0, 1)).reshape(N, T, C, H, W)

        # apply scale and bias
        pos_gamma = pos_gamma.reshape(T, H, W, C).permute(0, 3, 1, 2)
        pos_beta = pos_beta.reshape(T, H, W, C).permute(0, 3, 1, 2)

        out = normalized * (1 + pos_gamma) + pos_beta

        return out.permute(0, 1, 3, 4, 2)

class FutureFrameQueryGenerator(nn.Module):
    def __init__(self, T):
        """
        T: number of queries to generate
        """
        super().__init__()
        self.T = T

    def forward(self, evt, pos_beta, pos_gamma, pos_fuser):
        """
        Args:
            evt: (N, C, H, W)
            pos_gamma: (T*H*W, C)
            pos_beta: (T*H*W, C)
            pos_fuser: instance of PosFeatFuser
        Return:
            out: (N, T, C, H, W)
        """
        out = evt.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        out = pos_fuser(out, pos_beta, pos_gamma)

        return out

if __name__ == '__main__':
    """
    coor_generator = CoorGenerator(8, 8, 10)
    h_list, w_list = torch.linspace(0, 7, 8), torch.linspace(0, 7, 8)
    t_list = torch.linspace(0, 9, 10)
    coor = coor_generator(t_list, h_list, w_list)

    nrmlp = NRMLP(out_channels = 512, fuse_method='SPADE').to('cuda:0')
    nrmlp_beta, nrmlp_gamma = nrmlp(coor.to('cuda:0'))

    x = torch.randn(64, 10, 512, 8, 8).to('cuda:0')
    fuser = PosFeatFuser(x_channels=512, param_free_norm_type = 'instance').to('cuda:0')
    out = fuser(x, nrmlp_beta, nrmlp_gamma)
    print(out.shape)
    
    ent = EventEncoder(512, 256, 1).to('cuda:0')
    z, mu, logvar = ent(x)

    query_generator = FutureFrameQueryGenerator(T = 10)
    future_queries = query_generator(z, nrmlp_beta, nrmlp_gamma, fuser)
    print(future_queries.shape)
    """