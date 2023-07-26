from .submodules import CoorGenerator, NRMLP, PosFeatFuser, FutureFrameQueryGenerator, EventEncoder
from .VidHRFormer import VidHRformerDecoderNAR, VidHRFormerEncoder
from .ResNetAutoEncoder import ResnetEncoder, ResnetDecoder, LitAE
import torch
import torch.nn as nn
import functools
#import torch.nn.utils.spectral_norm as spectral_norm
import pytorch_lightning as pl

from models import L1Loss, Div_KL, GANLoss

class LitPredictor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        #load and freeze the pretrained AutoEncoder
        self.VPTR_Dec = LitAE.load_from_checkpoint(cfg.Predictor.resume_AE_ckpt, cfg = cfg).VPTR_Dec
        self.VPTR_Enc = LitAE.load_from_checkpoint(cfg.Predictor.resume_AE_ckpt, cfg = cfg).VPTR_Enc
        for p in self.VPTR_Enc.parameters():
            p.requires_grad_(False)
        for p in self.VPTR_Dec.parameters():
            p.requires_grad_(False)
        self.VPTR_Enc = self.VPTR_Enc.eval()
        self.VPTR_Dec = self.VPTR_Dec.eval()

        #Init predictor and discriminator
        self.h_list = torch.linspace(0, cfg.Predictor.max_H-1, cfg.Predictor.max_H)
        self.w_list = torch.linspace(0, cfg.Predictor.max_W-1, cfg.Predictor.max_W)
        if self.cfg.Predictor.VFI:
            context_p, context_f, num_vi = self.cfg.Predictor.context_num_p, self.cfg.Predictor.context_num_f, self.cfg.Predictor.num_interpolate
            clip_length = context_p + context_f + num_vi
            assert self.cfg.Dataset.num_past_frames+self.cfg.Dataset.num_future_frames == clip_length, "Imcompatible VFI configurations"
            #reset the context and target coordinates
            idx = torch.linspace(0, clip_length-1, clip_length, dtype=torch.int64)
            self.to_list = torch.cat([idx[0:context_p], idx[-context_f:]])
            self.tp_list = idx[context_p:-context_f]
        else:
            self.to_list = torch.linspace(0, cfg.Dataset.num_past_frames-1, cfg.Dataset.num_past_frames)
            self.tp_list = torch.linspace(cfg.Dataset.num_past_frames, cfg.Dataset.num_past_frames+cfg.Dataset.num_future_frames-1, cfg.Dataset.num_future_frames)
        assert self.cfg.Predictor.max_T == self.cfg.Dataset.num_past_frames+self.cfg.Dataset.num_future_frames, "Incompatible max_T and clip length"
        
        self.predictor = Predictor(cfg.Predictor.max_H, cfg.Predictor.max_W, cfg.Predictor.max_T, self.h_list, self.w_list, self.to_list, self.tp_list, 
                                   cfg.Predictor.embed_dim, cfg.Predictor.fuse_method, cfg.Predictor.param_free_norm_type, cfg.Predictor.evt_hidden_channels, 
                                   1, cfg.Predictor.stochastic, cfg.Predictor.transformer_layers, 
                                   evt_former = cfg.Predictor.evt_former, learn_evt_token = False, evt_former_num_layers = cfg.Predictor.evt_former_num_layers,
                                   rand_context = cfg.Predictor.rand_context)
        self.training_step = self.training_step_no_gan
        if cfg.Predictor.use_gan:
            self.discriminator = Discriminator(cfg.Dataset.img_channels, ndf=cfg.Predictor.ndf)
            self.training_step = self.training_step_gan
        
        #init the loss function
        self.l1_loss = L1Loss()
        self.PF_l1_loss = L1Loss(lam=cfg.Predictor.lam_PF_L1)
        self.kl_div = Div_KL(cfg.Predictor.KL_beta)
        if cfg.Predictor.use_gan:
            self.gan_loss = GANLoss('vanilla', target_real_label=1.0, target_fake_label=0.0, lam_gan=cfg.Predictor.lam_gan)

        self.automatic_optimization = False #Manually optimization

        if self.cfg.Predictor.rand_context:
            print("training with random context")
            self.batch_process_fn = self.rand_context_batch_process
        elif self.cfg.Predictor.VFI:
            print("training only for video frame interpolation")
            self.batch_process_fn = self.VFI_batch_process
        else:
            print("training only for video frame prediction")
            self.batch_process_fn = self.normal_batch_process

    def forward(self, past_frames, future_frames = None):
        past_gt_feats = self.VPTR_Enc(past_frames)
        rec_past_frames = self.VPTR_Dec(past_gt_feats)

        rec_future_frames = None
        future_gt_feats = None
        if future_frames is not None:
            future_gt_feats = self.VPTR_Enc(future_frames)
            rec_future_frames = self.VPTR_Dec(future_gt_feats)

        pred_future_feats = self.predictor(past_gt_feats)
        
        pred_future_frames = self.VPTR_Dec(pred_future_feats)

        return rec_past_frames, rec_future_frames, pred_future_frames
    
    def training_step_gan(self, batch, batch_idx):
        opt_G, opt_D = self.optimizers()
        self.predictor.zero_grad()
        
        pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss = self.shared_step(batch, batch_idx)
        #update discriminator
        self.discriminator.zero_grad()
        loss_D, loss_D_fake, loss_D_real = self.cal_lossD(pred_frames, future_frames)
        self.manual_backward(loss_D)
        opt_D.step()
        
        #update predictor
        loss, Image_L1_loss, PF_L1_loss, kl_loss, loss_G_gan = self.cal_lossG(pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss)
        self.manual_backward(loss)

        nn.utils.clip_grad_norm_(self.predictor.transformer.parameters(), max_norm=self.cfg.Predictor.max_grad_norm, norm_type=2)
        opt_G.step()

        self.log('loss_D_train', loss_D)
        self.log('loss_D_fake_train', loss_D_fake)
        self.log('loss_D_real_train', loss_D_real)

        self.log('loss_G_gan_train', loss_G_gan)
        self.log('loss_train', loss)
        self.log('Image_L1_train', Image_L1_loss)
        self.log('PF_L1_train', PF_L1_loss)
        if self.cfg.Predictor.stochastic:
            self.log('KL_loss_train', kl_loss)
        
        if self.cfg.Predictor.use_cosine_scheduler:
            #update lr scheduler
            epoch_batches = self.trainer.datamodule.len_train_loader
            sch_P, sch_D = self.lr_schedulers()
            sch_P.step(self.trainer.current_epoch + batch_idx / epoch_batches)
            sch_D.step(self.trainer.current_epoch + batch_idx / epoch_batches)

    def training_step_no_gan(self, batch, batch_idx):
        opt = self.optimizers()
        self.predictor.zero_grad()

        pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss = self.shared_step(batch, batch_idx)
        Image_L1_loss = self.l1_loss(pred_frames, future_frames)
        PF_L1_loss = self.PF_l1_loss(pred_future_feats, future_gt_feats)
        loss = Image_L1_loss + PF_L1_loss + kl_loss

        self.manual_backward(loss)

        nn.utils.clip_grad_norm_(self.predictor.transformer.parameters(), max_norm=self.cfg.Predictor.max_grad_norm, norm_type=2)
        opt.step()
        
        self.log('loss_train', loss)
        self.log('Image_L1_train', Image_L1_loss)
        self.log('PF_L1_train', PF_L1_loss)
        if self.cfg.Predictor.stochastic:
            self.log('KL_loss_train', kl_loss)
        
        if self.cfg.Predictor.use_cosine_scheduler:
            #update lr scheduler
            epoch_batches = self.trainer.datamodule.len_train_loader
            sch = self.lr_schedulers()
            sch.step(self.trainer.current_epoch + batch_idx / epoch_batches)

    def validation_step(self, batch, batch_idx):
        pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss = self.shared_step(batch, batch_idx)
        if self.cfg.Predictor.use_gan:
            loss_D, loss_D_fake, loss_D_real = self.cal_lossD(pred_frames, future_frames)
            self.log('loss_D_val', loss_D)
            self.log('loss_D_fake_val', loss_D_fake)
            self.log('loss_D_real_val', loss_D_real)

            loss, Image_L1_loss, PF_L1_loss, kl_loss, loss_G_gan = self.cal_lossG(pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss)
            self.log('loss_G_gan_val', loss_G_gan)
        
        else:
            Image_L1_loss = self.l1_loss(pred_frames, future_frames)
            PF_L1_loss = self.PF_l1_loss(pred_future_feats, future_gt_feats)
            loss = Image_L1_loss + PF_L1_loss + kl_loss
        
        self.log('loss_val', loss)
        self.log('Image_L1_val', Image_L1_loss)
        self.log('PF_L1_val', PF_L1_loss)
        if self.cfg.Predictor.stochastic:
            self.log('KL_loss_val', kl_loss)

    def shared_step(self, batch, batch_idx):
        batch = self.batch_process_fn(batch)
        past_frames, future_frames = batch
        
        self.VPTR_Enc = self.VPTR_Enc.eval()
        with torch.no_grad():
            past_gt_feats = self.VPTR_Enc(past_frames)
            future_gt_feats = self.VPTR_Enc(future_frames)
        
        if self.cfg.Predictor.stochastic:
            pred_future_feats, mu_o, logvar_o, mu_p, logvar_p = self.predictor(past_gt_feats, future_gt_feats)
            kl_loss = self.kl_div(mu_o, logvar_o, mu_p, logvar_p)
        else:
            pred_future_feats = self.predictor(past_gt_feats)
            kl_loss = 0.
        
        #one critic line, the pytorchlightning has a bug, it will unfreeze VPTR_Dec even if we have frozen it in the init()
        self.VPTR_Dec = self.VPTR_Dec.eval() 
        self.VPTR_Dec.zero_grad()

        pred_frames = self.VPTR_Dec(pred_future_feats)

        return pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss
        
    def configure_optimizers(self):
        optimizer_P = torch.optim.AdamW(params = self.predictor.parameters(), lr = self.cfg.Predictor.predictor_lr)
        if self.cfg.Predictor.use_gan:
            optimizer_D = torch.optim.Adam(params = self.discriminator.parameters(), lr=self.cfg.Predictor.predictor_lr)
            if self.cfg.Predictor.use_cosine_scheduler:
                scheduler_P = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_P, self.cfg.Predictor.scheduler_T0, 
                                                                                   T_mult=1, eta_min=self.cfg.Predictor.scheduler_eta_min, 
                                                                                   last_epoch= -1)
                scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, self.cfg.Predictor.scheduler_T0, 
                                                                                   T_mult=1, eta_min=self.cfg.Predictor.scheduler_eta_min, 
                                                                                   last_epoch= -1)

                return [optimizer_P, optimizer_D], [scheduler_P, scheduler_D]
            else:
                return optimizer_P, optimizer_D
        else:
            if self.cfg.Predictor.use_cosine_scheduler:
                scheduler_P = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_P, self.cfg.Predictor.scheduler_T0, 
                                                                                   T_mult=1, eta_min=self.cfg.Predictor.scheduler_eta_min, 
                                                                                   last_epoch= -1)
                return [optimizer_P], [scheduler_P]
            else:
                return [optimizer_P]
    
    def cal_lossD(self, fake_imgs, real_imgs):
        pred_fake = self.discriminator(fake_imgs.detach().flatten(0, 1))
        loss_D_fake = self.gan_loss(pred_fake, False)
        # Real
        pred_real = self.discriminator(real_imgs.flatten(0,1))
        loss_D_real = self.gan_loss(pred_real, True)
        # combine loss and calculate gradients
        wd = loss_D_fake + loss_D_real

        return wd, loss_D_fake, loss_D_real
    
    def cal_lossG(self, pred_frames, future_frames, pred_future_feats, future_gt_feats, kl_loss):
        pred_fake = self.discriminator(pred_frames.flatten(0, 1))
        loss_G_gan = self.gan_loss(pred_fake, True)

        Image_L1_loss = self.l1_loss(pred_frames, future_frames)
        PF_L1_loss = self.PF_l1_loss(pred_future_feats, future_gt_feats)
        loss = Image_L1_loss + PF_L1_loss + kl_loss + loss_G_gan

        return loss, Image_L1_loss, PF_L1_loss, kl_loss, loss_G_gan
    
    def rand_context_batch_process(self, batch):
        clip_batch_o, clip_batch_p, idx_o, idx_p = batch

        #reset the coordinate
        coor = self.predictor.all_coor

        self.predictor.observed_coor = coor[idx_o, ...].flatten(0, 2)
        self.predictor.predict_coor = coor[idx_p, ...].flatten(0, 2)
        self.predictor.TP = idx_p.shape[0]

        return (clip_batch_o, clip_batch_p)
    
    def VFI_batch_process(self, batch):
        past_frames, future_frames = batch
        clip = torch.cat([past_frames, future_frames], dim = 1)

        clip_batch_o = clip[:, self.to_list, ...]
        clip_batch_p = clip[:, self.tp_list, ...]
        return (clip_batch_o, clip_batch_p)

    def normal_batch_process(self, batch):
        return batch


class Predictor(nn.Module):
    def __init__(self, max_H, max_W, max_T, h_list, w_list, to_list, tp_list,
                 embed_dim = 512, 
                 fuse_method = 'SPADE', param_free_norm_type = 'layer',
                 evt_hidden_channels = 256, evt_n_layers=1, stochastic = True,
                 transformer_layers=4, num_heads=8, window_size = 4, dropout = 0.1, drop_path = 0.1, Spatial_FFN_hidden_ratio = 4, dim_feedforward = 1024, norm=nn.LayerNorm(512), return_intermediate=False,
                 evt_former = True, learn_evt_token = False, evt_former_num_layers=4, rand_context = False):
        super().__init__()
        self.stochastic = stochastic
        self.evt_former = evt_former
        self.h_list = h_list
        self.w_list = w_list
        self.coor_generator = CoorGenerator(max_H, max_W, max_T)
        if not rand_context:
            self.register_buffer("observed_coor", self.coor_generator(to_list, h_list, w_list))
            self.register_buffer("predict_coor", self.coor_generator(tp_list, h_list, w_list))
        else:
            self.observed_coor = None
            self.predict_coor = None
            self.register_buffer('all_coor', self.coor_generator(torch.cat([to_list, tp_list]), h_list, w_list).reshape(max_T, max_H, max_W, 3))

        self.nrmlp = NRMLP(out_channels = embed_dim, fuse_method=fuse_method)
        self.fuser = PosFeatFuser(x_channels=embed_dim, param_free_norm_type = param_free_norm_type)

        if self.evt_former:
            self.EVT_Former = VidHRFormerEncoder(evt_former_num_layers, max_H, max_W, embed_dim, num_heads, window_size, dropout, drop_path, 
                                                 Spatial_FFN_hidden_ratio, dim_feedforward, norm, learn_evt_token)

        self.evt_posterior = EventEncoder(embed_dim, evt_hidden_channels, evt_n_layers, stochastic)
        self.evt_prior = None
        if self.stochastic:
            self.evt_prior = EventEncoder(embed_dim, evt_hidden_channels, evt_n_layers, stochastic)

        self.TP = tp_list.shape[0]
        self.transformer = VidHRformerDecoderNAR(transformer_layers, max_H, max_W, embed_dim, num_heads, window_size, dropout, drop_path, Spatial_FFN_hidden_ratio, dim_feedforward, norm, return_intermediate)

    def forward(self, observed_features, predict_features_gt = None):
        """
        observed_features: (N, To, H, W, C)
        """
        op_beta, op_gamma = self.nrmlp(self.observed_coor)
        pp_beta, pp_gamma = self.nrmlp(self.predict_coor)

        if self.stochastic:
            observed_features, observe_evt_coding = self.evt_coding_forward(observed_features, op_beta, op_gamma)
            zo, mu_o, logvar_o = self.evt_prior(observe_evt_coding)
            if predict_features_gt is not None:
                _, predict_evt_coding = self.evt_coding_forward(predict_features_gt, pp_beta, pp_gamma)
                zp, mu_p, logvar_p = self.evt_posterior(predict_evt_coding)

            if self.training:
                assert predict_features_gt is not None, "please input groundtruth predict features for storchastic model training/val"
                query_evt = zp.unsqueeze(1).repeat(1, self.TP, 1, 1, 1)
                out = self.transformer(query_evt, observed_features, (op_beta, op_gamma), (pp_beta, pp_gamma), self.fuser)

            else:
                query_evt = zo.unsqueeze(1).repeat(1, self.TP, 1, 1, 1)
                out = self.transformer(query_evt, observed_features, (op_beta, op_gamma), (pp_beta, pp_gamma), self.fuser)
            
            if predict_features_gt is None:
                return out
            else:
                return out, mu_o, logvar_o, mu_p, logvar_p
        else:
            observed_features, observe_evt_coding = self.evt_coding_forward(observed_features, op_beta, op_gamma)
            mu_o = self.evt_posterior(observe_evt_coding)

            query_evt = mu_o.unsqueeze(1).repeat(1, self.TP, 1, 1, 1)
            out = self.transformer(query_evt, observed_features, (op_beta, op_gamma), (pp_beta, pp_gamma), self.fuser)

            return out

    def evt_coding_forward(self, x, pos_beta, pos_gamma):
        """
        x: observed features or predict_features_gt, with shape (N, T, C, H, W)
        """
        if self.evt_former:
            x = self.EVT_Former(x, (pos_beta, pos_gamma), self.fuser)
            if self.EVT_Former.evt_token:
                x, evt_coding = x[:, 0:-1, ...], x[:, -1, ...]
            else:
                evt_coding = x.mean(dim=1)
        else:
            evt_coding = self.fuser(x.permute(0, 1, 4, 2, 3), pos_beta, pos_gamma).permute(0, 1, 3, 4, 2).mean(dim=1)
        
        return x, evt_coding

    def reset_pos_coor(self, to_list, tp_list):
        try:
            device = self.observed_coor.device
        except:
            device = self.all_coor.device
        self.predict_coor = self.coor_generator(tp_list, self.h_list, self.w_list).to(device)
        self.observed_coor = self.coor_generator(to_list, self.h_list, self.w_list).to(device)
        self.TP = tp_list.shape[0]
                


class Discriminator(nn.Module):
    """
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    Defines a PatchGAN discriminator
    """
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)


    def forward(self, input):
        """Standard forward."""
        return self.model(input)
        




    
