import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from models import LitAE, LitPredictor
from models import L1Loss
from utils import LitDataModule, VisCallbackAE, save_code_cfg, get_dataloader

import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf


initialize(config_path="./")
cfg = compose(config_name="config")
cfg.Env.strategy = 'dp'

pl.seed_everything(cfg.Env.rand_seed, workers=True)

import numpy as np
from utils import PSNR, SSIM, MSEScore
from utils import BatchAverageMeter
import lpips
from skimage.metrics import structural_similarity as ssim

from utils import get_fvd_feats, frechet_distance, load_i3d_pretrained

from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

def plot_model_result(pred, fig_name, num_frames, n = 2):
    fig, ax = plt.subplots(1, num_frames, figsize = (num_frames, 1))
    fig.subplots_adjust(wspace=0., hspace = 0.)

    for j in range(num_frames):
        ax[j].set_axis_off()
        
        img = pred[:, j, :, :, :].clone()
        img = renorm_transform(img)
        img = torch.clamp(img, min = 0., max = 1.)
        img = img[n, ...]

        img = transforms.ToPILImage()(img)
        ax[j].imshow(img, cmap = 'gray')
    fig.savefig(f'{fig_name}.pdf', bbox_inches = 'tight')
    

def inference(sample, to_list, tp_list, predictor, rand_sample_num = None, gray_scale = True):
    p, f = sample
    full_frame = torch.cat([p, f], dim = 1).to('cuda:0')
    p = full_frame[:, to_list.to(torch.long), ...]
    try:
        f = full_frame[:, tp_list.to(torch.long), ...]
    except:
        f = None
    predictor.predictor.reset_pos_coor(to_list, tp_list)
    predictor = predictor.eval()
    with torch.no_grad():
        rec_past_frames, rec_future_frames, pred = predictor(p.clone())
        if rand_sample_num is not None and f is not None:
            ssim = cal_metric_single_iter(pred.clone(), f.clone(), SSIM(), use_lpips = False, gray_scale = gray_scale)[0].mean()
            for n in range(rand_sample_num -1): #select the one with best ssim
                _, _, pred_temp = predictor(p.clone())
                temp_ssim = cal_metric_single_iter(pred_temp.clone(), f.clone(), SSIM(), use_lpips = False, gray_scale = gray_scale)[0].mean()
                if temp_ssim > ssim:
                    pred = pred_temp
    return p, f, pred

def VFI_test_single_iter(sample, to_list, tp_list, predictor, device):
    p, f = sample
    full_frame = torch.cat([p, f], dim = 1).to(device)
    o = full_frame[:, to_list.to(torch.long), ...]
    t = full_frame[:, tp_list.to(torch.long), ...]
    pred_t = predictor(o)[-1]
    
    return pred_t, t

def NAR_test_single_iter(sample, lit_predictor, num_pred, device):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    assert num_pred == future_frames.shape[1], "Mismatch between ground truth future frames length and num_pred"
    assert num_pred % lit_predictor.predictor.TP == 0, "Mismatch of num_pred and trained Transformer"
    
    pred_frames = []
    for i in range(0, num_pred//lit_predictor.predictor.TP):
        pred = lit_predictor(past_frames)[-1]

        pred_frames.append(pred)
        past_frames = pred
        

    pred_frames = torch.cat(pred_frames, dim = 1)
    return pred_frames, future_frames

def NAR_BAIR_test_single_iter_new(sample, lit_predictor, num_pred, device):
    past_frames, future_frames = sample
    past_frames = past_frames.to(device)
    future_frames = future_frames.to(device)
    assert num_pred == future_frames.shape[1], "Mismatch between ground truth future frames length and num_pred"

    pred_frames = []
    
    #prediction 1
    pred = lit_predictor(past_frames)[-1]
    pred_frames.append(pred)
    
    #prediction 2
    past_frames = pred[:, -2:, ...]
    pred = lit_predictor(past_frames)[-1]
    pred_frames.append(pred)
    
    #prediction 3
    past_frames = pred[:, -2:, ...]
    pred = lit_predictor(past_frames)[-1]
    pred_frames.append(pred[:, 0:-2, ...])
    
    pred_frames = torch.cat(pred_frames, dim = 1)
    
    return pred_frames, future_frames

"""
def cal_metric_single_iter(pred_frames, gt_frames, metric_func, use_lpips = False, gray_scale = False):
    assert pred_frames.shape[1] == gt_frames.shape[1], "Mismatch temporal length"
    T = gt_frames.shape[1]
    ave_metrics = np.zeros(T)
    N = gt_frames.shape[0]
    
    for i in range(0, T):
        pred_t = pred_frames[:, i, ...]
        gt_t = gt_frames[:, i, ...]

        if not use_lpips:
            pred_t = renorm_transform(pred_t)
            gt_t = renorm_transform(gt_t)
            pred_t = torch.clamp(pred_t, min = 0., max = 1.)
            gt_t = torch.clamp(gt_t, min = 0., max = 1.)
        
        elif use_lpips and gray_scale:
            pred_t = renorm_transform(pred_t)
            gt_t = renorm_transform(gt_t)
            pred_t = torch.clamp(pred_t, min = 0., max = 1.)
            gt_t = torch.clamp(gt_t, min = 0., max = 1.)
            
            pred_t = pred_t.repeat(1, 3, 1, 1)
            gt_t = gt_t.repeat(1, 3, 1, 1)
        
        r = metric_func(pred_t, gt_t)
        try:
            m = r*N
            ave_metrics[i] = m
        except ValueError:
            ave_metrics[i] = r.sum().item()
    return ave_metrics, N
"""

def cal_metric_single_iter(pred_frames, gt_frames, metric_func, use_lpips = False, gray_scale = False, sum_batch= True):
    assert pred_frames.shape[1] == gt_frames.shape[1], "Mismatch temporal length"
    N, T, _, H, W = gt_frames.shape
    gt_frames = gt_frames.clone().flatten(0, 1)
    pred_frames = pred_frames.clone().flatten(0, 1)
    
    if not use_lpips:
        pred_frames = renorm_transform(pred_frames)
        gt_frames = renorm_transform(gt_frames)
        pred_frames = torch.clamp(pred_frames, min = 0., max = 1.)
        gt_frames = torch.clamp(gt_frames, min = 0., max = 1.)

    if use_lpips and gray_scale:
        pred_frames = renorm_transform(pred_frames)
        gt_frames = renorm_transform(gt_frames)
        pred_frames = torch.clamp(pred_frames, min = 0., max = 1.)
        gt_frames = torch.clamp(gt_frames, min = 0., max = 1.)
        pred_frames = pred_frames.repeat(1, 3, 1, 1)
        gt_frames = gt_frames.repeat(1, 3, 1, 1)
    
    try:
        metric_func_name = metric_func.__name__
    except:
        metric_func_name = None
    if use_lpips:
        r = metric_func(pred_frames, gt_frames)
    elif metric_func_name == 'structural_similarity':
        r = np.zeros(N*T)
        pred_frames = pred_frames.clone().cpu().numpy()
        gt_frames = gt_frames.clone().cpu().numpy()
        for i in range(N*T):
            r[i] = metric_func(pred_frames[i, ...], gt_frames[i, ...], channel_axis=0)
        r = torch.from_numpy(r)
    elif metric_func_name == 'frechet_distance':
        if gray_scale:
            pred_frames = pred_frames.repeat(1, 3, 1, 1)
            gt_frames = gt_frames.repeat(1, 3, 1, 1)
        pred_frames = pred_frames.reshape(N, T, 3, H, W).permute(0, 2, 1, 3, 4)
        gt_frames = gt_frames.reshape(N, T, 3, H, W).permute(0, 2, 1, 3, 4)
        fake_embeddings = get_fvd_feats(pred_frames, i3d=i3d, device=pred_frames.device)
        real_embeddings = get_fvd_feats(gt_frames, i3d=i3d, device=gt_frames.device)
        fvd = frechet_distance(fake_embeddings, real_embeddings)
        r = torch.ones(N*T)*fvd
    else:
        r = metric_func(pred_frames, gt_frames, mean_flag = False)
    
    r = r.reshape(N, T)
    if sum_batch:
        return r.sum(0), N
    else:
        return r, N

"""
def metric_test(data_loader, lit_predictor, num_pred, metric_func, inference_func, device, use_lpips = False, gray_scale = False, VFI = False, to_list = None, tp_list = None):
    ave_meter = BatchAverageMeter('lpips', ':.10e')
    lit_predictor = lit_predictor.eval()
    if VFI:
        lit_predictor.predictor.reset_pos_coor(to_list, tp_list)
    with torch.no_grad():
        ave_metrics = torch.zeros(num_pred)
        sample_num = 0
        for idx, sample in enumerate(data_loader):
            if not VFI:
                pred_frames, gt_frames = inference_func(sample, lit_predictor, num_pred, device)
            else:
                pred_frames, gt_frames = inference_func(sample, to_list, tp_list, lit_predictor, device)
                
            batch_metrics, N = cal_metric_single_iter(pred_frames, gt_frames, metric_func, use_lpips, gray_scale)
            sample_num += N
            ave_metrics += batch_metrics.cpu()
        
        return ave_metrics / sample_num
"""

def metric_test(data_loader, lit_predictor, num_pred, metric_func, inference_func, device, use_lpips = False, gray_scale = False, VFI = False, to_list = None, tp_list = None, rand_sample_num = None):
    ave_meter = BatchAverageMeter('lpips', ':.10e')
    lit_predictor = lit_predictor.eval()
    try:
        metric_func_name = metric_func.__name__
    except AttributeError:
        metric_func_name = 'lpips'
    if VFI:
        lit_predictor.predictor.reset_pos_coor(to_list, tp_list)
    with torch.no_grad():
        ave_metrics = torch.zeros(num_pred)
        sample_num = 0
        pgbar = tqdm(total = len(data_loader), desc = 'Testing...')
        for idx, sample in enumerate(data_loader):
            if not VFI:
                pred_frames, gt_frames = inference_func(sample, lit_predictor, num_pred, device)
            else:
                pred_frames, gt_frames = inference_func(sample, to_list, tp_list, lit_predictor, device)
            
            if rand_sample_num is None:
                batch_metrics, N = cal_metric_single_iter(pred_frames, gt_frames, metric_func, use_lpips, gray_scale)
            else:
                #batch_metrics: (N, T)
                batch_metrics, N = cal_metric_single_iter(pred_frames, gt_frames, metric_func, use_lpips, gray_scale, sum_batch = False)
                for n in range(rand_sample_num - 1):
                    if not VFI:
                        pred_frames, gt_frames = inference_func(sample, lit_predictor, num_pred, device)
                    else:
                        pred_frames, gt_frames = inference_func(sample, to_list, tp_list, lit_predictor, device)
                    batch_metrics_temp, N = cal_metric_single_iter(pred_frames, gt_frames, metric_func, use_lpips, gray_scale, sum_batch = False)
                    
                    if metric_func_name == 'SSIM' or metric_func_name == 'PSNR':
                        index = batch_metrics.mean(1) < batch_metrics_temp.mean(1)
                    else:
                        index = batch_metrics.mean(1) > batch_metrics_temp.mean(1)
                    batch_metrics[index, :] = batch_metrics_temp[index, :]
                batch_metrics = batch_metrics.sum(0)
            
            sample_num += N
            ave_metrics += batch_metrics.cpu()
            pgbar.update(1)
        pgbar.close()
        return ave_metrics / sample_num

train_loader, val_loader, test_loader, renorm_transform = get_dataloader('KTH', 96,'./KTH', test_past_frames = 10, test_future_frames = 20, num_workers = 1)
predictor = LitPredictor.load_from_checkpoint('/home/xiyex/scratch/xiyex/VPTR2_ckpts/KTH_VFP_stochastic/Predictor-epoch=899.ckpt', cfg=cfg).to('cuda:0')
if cfg.Predictor.rand_context:
    to_list =torch.linspace(0, cfg.Dataset.num_past_frames-1, cfg.Dataset.num_past_frames)
    tp_list = torch.linspace(cfg.Dataset.num_past_frames, cfg.Dataset.num_past_frames+cfg.Dataset.num_future_frames-1, cfg.Dataset.num_future_frames)
    predictor.predictor.reset_pos_coor(to_list, tp_list)

device = 'cuda:0'
ave_psnr = metric_test(test_loader, predictor, 20, PSNR, NAR_test_single_iter, device, use_lpips = False, gray_scale = True, rand_sample_num = 100)
print(ave_psnr.mena().item())