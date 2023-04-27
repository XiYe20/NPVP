from torch import nn
import torch
import functools
from torch.nn import init
import pytorch_lightning as pl
from .criterion import L1Loss
"""
Modified based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""

from .submodules import NonLocalAttenion2D, NonLocalAttenion1D, Factorized3DConvAttn

class LitAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.VPTR_Enc = ResnetEncoder(cfg.Dataset.img_channels, ngf=cfg.AE.ngf, n_downsampling = cfg.AE.n_downsampling, 
                                      num_res_blocks = cfg.AE.num_res_blocks, norm_layer=nn.BatchNorm2d, 
                                      norm_layer1d=nn.BatchNorm1d, learn_3d = cfg.AE.learn_3d)
        self.VPTR_Dec = ResnetDecoder(cfg.Dataset.img_channels, ngf=cfg.AE.ngf, n_downsampling = cfg.AE.n_downsampling, 
                                      out_layer = cfg.AE.out_layer, norm_layer=nn.BatchNorm2d)
        self.cfg = cfg
        self.l1_loss = L1Loss()
    
    def forward(self, x):
        encoding = self.VPTR_Enc(x)
        rec_frames = self.VPTR_Dec(encoding)
        return rec_frames, encoding
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('L1_loss_train', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('L1_loss_valid', loss)
        return loss

    def shared_step(self, batch, batch_idx):
        past_frames, future_frames = batch
        x = torch.cat([past_frames, future_frames], dim = 1)
        rec_frames = self.VPTR_Dec(self.VPTR_Enc(x))
        loss = self.l1_loss(rec_frames, x)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = list(self.VPTR_Enc.parameters()) + list(self.VPTR_Dec.parameters()), 
                                     lr=self.cfg.AE.AE_lr, betas = (0.5, 0.999))
        return optimizer

class ResnetEncoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_downsampling = 3, num_res_blocks = 2, norm_layer=nn.BatchNorm2d, norm_layer1d=nn.BatchNorm1d, use_dropout=False, padding_type='reflect', learn_3d = True):
        """Construct a Resnet-based Encoder
        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.n_downsampling = n_downsampling
        self.num_res_blocks = num_res_blocks
        self.block0 = nn.Sequential(nn.ReflectionPad2d(3),
                                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                                    norm_layer(ngf),
                                    nn.ReLU(True))
        
        self.block1 = nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                    norm_layer(ngf*2),
                                    nn.ReLU(True))
        ngf = ngf*2
        for i in range(1, n_downsampling):
            setattr(self, f'block{i+1}_3dConvAttn', Factorized3DConvAttn(in_channels=ngf,
                                                                         norm_layer_2d = norm_layer,
                                                                         norm_layer_1d = norm_layer1d,
                                                                         activ_func = nn.ReLU(True),
                                                                         learn_3d = learn_3d))
            setattr(self, f'block{i+1}_conv', nn.Sequential(nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                                            norm_layer(ngf*2),
                                                            nn.ReLU(True)))
            ngf = ngf*2
        """
        self.block2_3dConvAttn = Factorized3DConvAttn(in_channels=ngf*2,
                                                      norm_layer_2d = norm_layer,
                                                      norm_layer_1d = norm_layer1d,
                                                      activ_func = nn.ReLU(True),
                                                      learn_3d = learn_3d)
        self.block2_conv = nn.Sequential(nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                         norm_layer(ngf * 4),
                                         nn.ReLU(True))
        
        self.block3_3dConvAttn = Factorized3DConvAttn(in_channels=ngf*4,
                                                      norm_layer_2d = norm_layer,
                                                      norm_layer_1d = norm_layer1d,
                                                      activ_func = nn.ReLU(True),
                                                      learn_3d = learn_3d)
        self.block3_conv = nn.Sequential(nn.Conv2d(ngf*4, ngf*8, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                         norm_layer(ngf * 8),
                                         nn.ReLU(True))
        """

        #9 resnet-blocks
        for i in range(num_res_blocks):       # add ResNet blocks
            setattr(self, f'res_3dConvAttn_{i}', Factorized3DConvAttn(in_channels=ngf,
                                                                      norm_layer_2d = norm_layer,
                                                                      norm_layer_1d = norm_layer1d,
                                                                      activ_func = nn.ReLU(True),
                                                                      learn_3d = learn_3d))
            setattr(self, f'res_conv_{i}', ResnetBlock(ngf, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias))
        
        self.out_act = nn.ReLU()
    
    def forward(self, x):
        """
        x: (N, T, C, H, W)
        """
        N, T, _, _, _ = x.shape
        x = x.flatten(0, 1)

        x = self.block0(x)
        x = self.block1(x)
        for i in range(1, self.n_downsampling):
            x = getattr(self, f'block{i+1}_3dConvAttn')(x, T)
            x = getattr(self, f'block{i+1}_conv')(x)
        """
        x = self.block2_3dConvAttn(x, T)
        x = self.block2_conv(x)
        x = self.block3_3dConvAttn(x, T)
        x = self.block3_conv(x)
        """
        for i in range(self.num_res_blocks):
            x = getattr(self, f'res_3dConvAttn_{i}')(x, T)
            x = getattr(self, f'res_conv_{i}')(x)
        
        x = self.out_act(x)
        _, C, H, W = x.shape
        x = x.reshape(N, T, C, H, W)

        return x

class ResnetDecoder(nn.Module):
    def __init__(self, output_nc, ngf=64, n_downsampling = 2, norm_layer=nn.BatchNorm2d, use_dropout=False, padding_type='reflect', out_layer = 'Tanh'):
        """Construct a Resnet-based Encoder
        Parameters:
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        #The first up-sampling layer
        mult = 2**n_downsampling
        model += [nn.ConvTranspose2d(ngf*mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        for i in range(1, n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if out_layer == 'Tanh':
            model += [nn.Tanh()]
        elif out_layer == 'Sigmoid':
            model += [nn.Sigmoid()]
        else:
            raise ValueError("Unsupported output layer")

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        x: (N, T, C, H, W)
        """
        N, T, _, _, _ = x.shape
        x = x.flatten(0, 1)
        x = self.model(x)
        NT, C, H, W = x.shape
        x = x.reshape(NT//T, T, C, H, W)
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


if __name__ == '__main__':
    x = torch.randn(16, 3, 64, 64)
    enc = ResnetEncoder(input_nc = 3)
    dec = ResnetDecoder(output_nc = 3)
    out = dec(enc(x))

    print(out.shape)