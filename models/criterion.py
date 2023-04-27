import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import random

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0, lam_gan = 1.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
        
        self.lam_gan = lam_gan

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss*self.lam_gan


class GradientPanelty(nn.Module):
    def __init__(self, lam_wgan, lam_gp = 10.0):
        super().__init__()
        self.lam_gan = lam_wgan
        self.lam_gp = lam_gp
    
    def __call__(self, real_x, fake_x, critic):
        """
        real_x: (N, C, H, W)
        fake_x: (N, C, H, W)
        """
        batchsize = fake_x.shape[0]
        t = torch.rand(batchsize, 1, 1, 1, device = fake_x.device)
        interpolate_x = t*real_x + (1-t)*fake_x
        interpolate_x.requires_grad_(True)

        critic_out = critic(interpolate_x)

        gradients = torch.autograd.grad(outputs=critic_out, inputs=interpolate_x,
                                        grad_outputs=torch.ones(critic_out.size()).to(fake_x.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(batchsize, -1)
        gradients_norm = (gradients+1e-16).norm(2, dim = 1)
        gradient_panelty = torch.clamp(gradients_norm - 1, min = 0.)
        return torch.square(gradient_panelty).mean() * self.lam_gp * self.lam_gan


class L1Loss(nn.Module):
    def __init__(self, norm_dim = None, lam = 1.0):
        """
        Args:
            norm_dim: dimensionality for normalizing the input features, default no normalization
            lam: the weight for the L1 loss
        """
        super().__init__()
        self.norm_dim = norm_dim
        self.lam = lam
        
    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.abs(pred - gt)
        mse = se.mean()
        return mse*self.lam

class TemporalDiff(nn.Module):
    def __init__(self, lam = 1.0):
        super().__init__()
        self.lam = lam #weight for the temporal difference loss
    
    def forward(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        shuffle_pred, shuffle_gt = self.random_shuffle(pred, gt)
        diff_pred = pred - shuffle_pred
        diff_gt = gt - shuffle_gt

        loss = torch.abs(diff_pred - diff_gt).mean()
        
        return self.lam*loss
    
    def random_shuffle(self, pred, gt):
        #Shuffle along the temporal axis to get the product of two marginal distributions
        T = pred.shape[1]

        rand_shift = random.randint(1, T-1)
        return torch.roll(pred, rand_shift, 1), torch.roll(gt, rand_shift, 1)

class MSELoss(nn.Module):
    def __init__(self, temporal_weight = None, norm_dim = None):
        """
        Args:
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.norm_dim = norm_dim
    
    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        if self.norm_dim is not None:
            gt = F.normalize(gt, p = 2, dim = self.norm_dim)
            pred = F.normalize(pred, p = 2, dim = self.norm_dim)

        se = torch.square(pred - gt)
        if self.temporal_weight is not None:
            w = self.temporal_weight.to(se.device)
            if len(se.shape) == 5:
                se = se * w[None, :, None, None, None]
            elif len(se.shape) == 6:
                se = se * w[None, :, None, None, None, None] #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
        mse = se.mean()
        return mse

class GDL(nn.Module):
    def __init__(self, alpha = 1, temporal_weight = None):
        """
        Args:
            alpha: hyper parameter of GDL loss, float
            temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        self.alpha = alpha
        self.temporal_weight = temporal_weight

    def __call__(self, gt, pred):
        """
        pred --- tensor with shape (B, T, ...)
        gt --- tensor with shape (B, T, ...)
        """
        gt_shape = gt.shape
        if len(gt_shape) == 5:
            B, T, _, _, _ = gt.shape
        elif len(gt_shape) == 6: #for warped frames, (N, num_future_frames, num_past_frames, C, H, W)
            B, T, TP, _, _, _ = gt.shape
        gt = gt.flatten(0, -4)
        pred = pred.flatten(0, -4)

        gt_i1 = gt[:, :, 1:, :]
        gt_i2 = gt[:, :, :-1, :]
        gt_j1 = gt[:, :, :, :-1]
        gt_j2 = gt[:, :, :, 1:]

        pred_i1 = pred[:, :, 1:, :]
        pred_i2 = pred[:, :, :-1, :]
        pred_j1 = pred[:, :, :, :-1]
        pred_j2 = pred[:, :, :, 1:]

        term1 = torch.abs(gt_i1 - gt_i2)
        term2 = torch.abs(pred_i1 - pred_i2)
        term3 = torch.abs(gt_j1 - gt_j2)
        term4 = torch.abs(pred_j1 - pred_j2)

        if self.alpha != 1:
            gdl1 = torch.pow(torch.abs(term1 - term2), self.alpha)
            gdl2 = torch.pow(torch.abs(term3 - term4), self.alpha)
        else:
            gdl1 = torch.abs(term1 - term2)
            gdl2 = torch.abs(term3 - term4)
        
        if self.temporal_weight is not None:
            assert self.temporal_weight.shape[0] == T, "Mismatch between temporal_weight and predicted sequence length"
            w = self.temporal_weight.to(gdl1.device)
            _, C, H, W = gdl1.shape
            _, C2, H2, W2= gdl2.shape
            if len(gt_shape) == 5:
                gdl1 = gdl1.reshape(B, T, C, H, W)
                gdl2 = gdl2.reshape(B, T, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None]
            elif len(gt_shape) == 6:
                gdl1 = gdl1.reshape(B, T, TP, C, H, W)
                gdl2 = gdl2.reshape(B, T, TP, C2, H2, W2)
                gdl1 = gdl1 * w[None, :, None, None, None, None]
                gdl2 = gdl2 * w[None, :, None, None, None, None]

        #gdl1 = gdl1.sum(-1).sum(-1)
        #gdl2 = gdl2.sum(-1).sum(-1)

        #gdl_loss = torch.mean(gdl1 + gdl2)
        gdl1 = gdl1.mean()
        gdl2 = gdl2.mean()
        gdl_loss = gdl1 + gdl2
        
        return gdl_loss

class BiPatchNCE(nn.Module):
    """
    Bidirectional patchwise contrastive loss
    Implemented Based on https://github.com/alexandonian/contrastive-feature-loss/blob/main/models/networks/nce.py
    """
    def __init__(self, N, T, h, w, temperature = 0.07, lam=1.0):
        """
        T: number of frames
        N: batch size
        h: feature height
        w: feature width
        temporal_weight: penalty for loss at different time step, Tensor with length T
        """
        super().__init__()
        
        #mask meaning; 1 for postive pairs, 0 for negative pairs
        mask = torch.eye(h*w).long()
        mask = mask.unsqueeze(0).repeat(N*T, 1, 1).requires_grad_(False) #(N*T, h*w, h*w)
        self.register_buffer('mask', mask)
        self.temperature = temperature
        self.lam = lam

    def forward(self, gt_f, pred_f):
        """
        gt_f: ground truth feature/images, with shape (N, T, C, h, w)
        pred_f: predicted feature/images, with shape (N, T, C, h, w)
        """
        mask = self.mask

        gt_f = rearrange(gt_f, "N T C h w -> (N T) (h w) C")
        pred_f = rearrange(pred_f, "N T C h w -> (N T) (h w) C")

        #direction 1, decompose the matmul to two steps, Stop gradient for the negative pairs
        score1_diag = torch.matmul(gt_f, pred_f.transpose(1, 2)) * mask
        score1_non_diag = torch.matmul(gt_f, pred_f.detach().transpose(1, 2)) * (1.0 - mask)
        score1 = score1_diag + score1_non_diag #(N*T, h*w, h*w)
        score1 = torch.div(score1, self.temperature)
        
        #direction 2
        score2_diag = torch.matmul(pred_f, gt_f.transpose(1, 2)) * mask
        score2_non_diag = torch.matmul(pred_f, gt_f.detach().transpose(1, 2)) * (1.0 - mask)
        score2 = score2_diag + score2_non_diag
        score2 = torch.div(score2, self.temperature)

        target = (mask == 1).int()
        target = target.to(score1.device)
        target.requires_grad = False
        target = target.flatten(0, 1) #(N*T*h*w, h*w)
        target = torch.argmax(target, dim = 1)

        loss1 = nn.CrossEntropyLoss()(score1.flatten(0, 1), target)
        loss2 = nn.CrossEntropyLoss()(score2.flatten(0, 1), target)
        loss = (loss1 + loss2)*0.5

        return loss*self.lam


class NoamOpt:
    """
    defatult setup from attention is all you need: 
            factor = 2
            optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, factor, train_loader, warmup_epochs, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = len(train_loader)*warmup_epochs
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def reset_step(self, init_epoch, train_loader):
        print("!!!!Learning rate warmup warning: If you are resume training, keep the same Batchsize as before!!!!")
        self._step = len(train_loader) * init_epoch

class Div_KL(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    
    def forward(self, mu1, logvar1, mu2, logvar2):
        # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
        #   log( sqrt(
        # 
        N = mu1.shape[0]
        sigma1 = logvar1.mul(0.5).exp() 
        sigma2 = logvar2.mul(0.5).exp() 
        kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
        return self.beta * kld.sum() / N

if __name__ == '__main__':
    """
    w = temporal_weight_func(10)
    gdl_loss = GDL(temporal_weight = w)
    a = torch.randn(4, 10, 3, 64, 64)
    b = torch.randn(4, 10, 3, 64, 64)
    gdl = gdl_loss(a, b)

    print(gdl)

    mse_loss = MSELoss()
    mse1 = mse_loss(a, b)

    mse2 = nn.MSELoss()(a, b)
    print((mse1 -  mse2).sum())
    
    a = torch.randn(4, 10, 384)
    b = torch.randn(10, 4, 384)
    pc_ssl = PCSSL(10, 4, w)
    pc_ssl1 = PCSSL(10, 4)

    print(pc_ssl1(a, b))
    print(pc_ssl(a, b))

    tc_ssl1 = TCSSL(10, 4, 3, w)
    tc_ssl2 = TCSSL(10, 4, 3)
    print(tc_ssl1(a), tc_ssl2(a))

    
    a = torch.randn(4, 10, 384, 2, 2)
    b = torch.randn(4, 10, 384, 2, 2)
    dpc_ssl1 = DPCSSL(10, 4, 2, 2, temporal_weight = w)
    dpc_ssl2 = DPCSSL(10, 4, 2, 2)

    print(dpc_ssl1(a, b), dpc_ssl2(a, b))
    
    m = BiPatchNCE(4, 10, 8, 8)
    gt = torch.randn(4, 10, 256, 8, 8).requires_grad_(True)
    pred = torch.randn(4, 10, 256, 8, 8).requires_grad_(True)
    loss, score1,score2 = m(gt, pred)
    loss.backward()
    print(loss, score1.shape, score2.shape)
    #print(score1.grad)
    #print(score2.grad)
    

    feat = torch.randn(1, 5, 384, 5, 5)
    m = TermporalPairwiseMSE(norm_dim = 2)
    loss= m(feat)
    print(loss)

    

    nce = BiPatchNCE(4, 10, 8, 8)
    feat_q = torch.randn(4, 10, 512, 8, 8)
    feat_q = feat_q.to('cuda')
    feat_k = feat_q

    loss1, score1 = nce(F.normalize(feat_q, p=2.0, dim=2), F.normalize(feat_q, p=2.0, dim=2))
    print(loss1.mean())
    """
    gt = torch.randn(4, 64, 64, 3)
    pred = torch.randn(4, 64, 64, 3)
    l1_loss = L1Loss()
    loss1 = l1_loss(gt, pred)
    print(loss1)
    loss2 = l1_loss.patch_l1(gt, pred)
    print(loss2)
    