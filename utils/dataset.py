import torch
from torch import select
from torch.utils import data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch import Tensor
import pytorch_lightning as pl

import numpy as np
from PIL import Image
from pathlib import Path
import os
import copy
from typing import List
from tqdm import tqdm
import random
from typing import Optional
from itertools import groupby
from operator import itemgetter

import cv2

class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.len_train_loader = None
        self.len_val_loader = None
        self.len_test_loader = None

        if cfg.Dataset.name == 'KTH':
            self.norm_transform = VidNormalize(mean = 0.6013795, std = 2.7570653)
            self.renorm_transform = VidReNormalize(mean = 0.6013795, std = 2.7570653)
            self.train_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((64, 64)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((64, 64)), VidToTensor(), self.norm_transform])
            self.val_person_ids = [5]
        
        if cfg.Dataset.name == 'KITTI':
            self.norm_transform = VidNormalize((0.44812047, 0.47147775, 0.4677183),(1.5147436, 1.5871466, 1.5925455))
            self.renorm_transform = VidReNormalize((0.44811612, 0.47147346, 0.46771598),(1.5177081, 1.5897311, 1.5952978))
            self.train_transform = transforms.Compose([VidResize((128, 128)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidResize((128, 128)), VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'SMMNIST':
            self.renorm_transform = VidReNormalize(mean = 0., std = 1.0)
            self.train_transform = self.test_transform = VidToTensor()

        if cfg.Dataset.name == 'BAIR':
            self.norm_transform = VidNormalize((0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673))
            self.renorm_transform = VidReNormalize((0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673))
            self.train_transform = transforms.Compose([VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidToTensor(), self.norm_transform])

        if cfg.Dataset.name == 'CityScapes':
            self.norm_transform = VidNormalize((0.31604213, 0.35114038, 0.3104223),(1.2172801, 1.3219808, 1.2082524))
            self.renorm_transform = VidReNormalize((0.31604213, 0.35114038, 0.3104223),(1.2172801, 1.3219808, 1.2082524))
            self.train_transform = transforms.Compose([VidToTensor(), self.norm_transform])
            self.test_transform = transforms.Compose([VidToTensor(), self.norm_transform])

    def setup(self, stage: Optional[str] = None):
        if not self.cfg.Predictor.rand_context:
            self.cfg.Predictor.min_lo = None
            self.cfg.Predictor.max_lo = None
        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            if self.cfg.Dataset.name == 'KTH':
                KTHTrainData = KTHDataset(self.cfg.Dataset.dir, transform = self.train_transform, train = True, val = True, 
                                          num_past_frames= self.cfg.Dataset.num_past_frames, num_future_frames= self.cfg.Dataset.num_future_frames,
                                          val_person_ids = self.val_person_ids, min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)#, actions = ['walking_no_empty'])
                self.train_set, self.val_set = KTHTrainData()
            
            if self.cfg.Dataset.name == 'KITTI':
                KITTITrainData = KITTIDataset(self.cfg.Dataset.dir, [10, 11, 12, 13], transform = self.train_transform, train = True, val = True,
                                                num_past_frames= self.cfg.Dataset.num_past_frames, num_future_frames= self.cfg.Dataset.num_future_frames,
                                                min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)
                self.train_set, self.val_set = KITTITrainData()

            if self.cfg.Dataset.name == 'BAIR':
                BAIR_train_whole_set = BAIRDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', 
                                                   num_past_frames = self.cfg.Dataset.num_past_frames, num_future_frames = self.cfg.Dataset.num_future_frames,
                                                   min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)()
                train_val_ratio = 0.95
                BAIR_train_set_length = int(len(BAIR_train_whole_set) * train_val_ratio)
                BAIR_val_set_length = len(BAIR_train_whole_set) - BAIR_train_set_length
                self.train_set, self.val_set = random_split(BAIR_train_whole_set, [BAIR_train_set_length, BAIR_val_set_length],
                                                            generator=torch.Generator().manual_seed(2021))
            if self.cfg.Dataset.name == 'CityScapes':
                self.train_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('train'), self.train_transform, color_mode = 'RGB', 
                                                   num_past_frames = self.cfg.Dataset.num_past_frames, num_future_frames = self.cfg.Dataset.num_future_frames,
                                                   min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)()
                self.val_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('val'), self.train_transform, color_mode = 'RGB', 
                                                   num_past_frames = self.cfg.Dataset.num_past_frames, num_future_frames = self.cfg.Dataset.num_future_frames,
                                                   min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)()

            if self.cfg.Dataset.name == 'SMMNIST':
                self.train_set = StochasticMovingMNIST(True, Path(self.cfg.Dataset.dir), self.cfg.Dataset.num_past_frames, self.cfg.Dataset.num_future_frames, self.train_transform, min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)
                train_val_ratio = 0.95
                SMMNIST_train_set_length = int(len(self.train_set) * train_val_ratio)
                SMMNIST_val_set_length = len(self.train_set) - SMMNIST_train_set_length
                self.train_set, self.val_set = random_split(self.train_set, [SMMNIST_train_set_length, SMMNIST_val_set_length],
                                                            generator=torch.Generator().manual_seed(2021))
            
            #Use all training dataset for the final training
            if self.cfg.Dataset.phase == 'deploy':
                self.train_set = ConcatDataset([self.train_set, self.val_set])

            dev_set_size = self.cfg.Dataset.dev_set_size
            if dev_set_size is not None:
                self.train_set, _ = random_split(self.train_set, [dev_set_size, len(self.train_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
                self.val_set, _ = random_split(self.val_set, [dev_set_size, len(self.val_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
            
            self.len_train_loader = len(self.train_dataloader())
            self.len_val_loader = len(self.val_dataloader())

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            if self.cfg.Dataset.name == 'KTH':
                KTHTestData = KTHDataset(self.cfg.Dataset.dir, transform = self.test_transform, train = False, val = False, 
                                        num_past_frames= self.cfg.Dataset.test_num_past_frames, num_future_frames= self.cfg.Dataset.test_num_future_frames, min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)#, actions = ['walking_no_empty'])
                self.test_set = KTHTestData()
            
            if self.cfg.Dataset.name == 'KITTI':
                KITTITrainData = KITTIDataset(self.cfg.Dataset.dir, [10, 11, 12, 13], transform = self.train_transform, train = False, val = False,
                                                num_past_frames= self.cfg.Dataset.num_past_frames, num_future_frames= self.cfg.Dataset.num_future_frames,
                                                min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)
                self.test_set = KITTITrainData()

            if self.cfg.Dataset.name == 'BAIR':
                self.test_set = BAIRDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.test_transform, color_mode = 'RGB', 
                                            num_past_frames= self.cfg.Dataset.test_num_past_frames, num_future_frames= self.cfg.Dataset.test_num_future_frames, min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)()
            if self.cfg.Dataset.name == 'CityScapes':
                self.test_set = CityScapesDataset(Path(self.cfg.Dataset.dir).joinpath('test'), self.train_transform, color_mode = 'RGB', 
                                                   num_past_frames = self.cfg.Dataset.num_past_frames, num_future_frames = self.cfg.Dataset.num_future_frames,
                                                   min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)()

            if self.cfg.Dataset.name == 'SMMNIST':
                self.test_set = StochasticMovingMNIST(False, Path(self.cfg.Dataset.dir), self.cfg.Dataset.test_num_past_frames, self.cfg.Dataset.test_num_future_frames, self.train_transform, min_lo = self.cfg.Predictor.min_lo, max_lo = self.cfg.Predictor.max_lo)
            
            self.len_test_loader = len(self.test_dataloader())
        

    def train_dataloader(self):
        if self.cfg.Predictor.rand_context:
            return DataLoader(self.train_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn=rand_context_collate_fn)
        else:
            return DataLoader(self.train_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True)

    def val_dataloader(self):
        if self.cfg.Predictor.rand_context:
            return DataLoader(self.val_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True, collate_fn=rand_context_collate_fn)
        else:
            return DataLoader(self.val_set, shuffle = True, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = True)

    def test_dataloader(self):
        if self.cfg.Predictor.rand_context:
            return DataLoader(self.test_set, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = False, collate_fn = rand_context_collate_fn)
        else:
            return DataLoader(self.test_set, batch_size=self.cfg.Dataset.batch_size, num_workers=self.cfg.Dataset.num_workers, drop_last = False)

def rand_context_collate_fn(batch_data):
    """
    batch_data: list of tuples, each tuple is (full_clip, min_lo, max_lo)
    """
    _, min_lo, max_lo = batch_data[0]
    clips, _, _ = zip(*batch_data)
    clip_batch = torch.stack(clips, dim=0)
    
    rand_idx = torch.randperm(clip_batch.shape[1])
    rand_lo = random.randint(min_lo, max_lo)
    idx_o = rand_idx[0:rand_lo]
    idx_p = rand_idx[rand_lo:]
    
    clip_batch_o = clip_batch[:, idx_o, ...]
    clip_batch_p = clip_batch[:, idx_p, ...]
    
    return (clip_batch_o, clip_batch_p, idx_o, idx_p)

def get_dataloader(data_set_name, batch_size, data_set_dir, test_past_frames = 10, test_future_frames = 10, dev_set_size = None, bair_crop = False, ngpus = 1, num_workers = 1, KTH_actions = ['boxing', 'handclapping', 'handwaving', 'jogging_no_empty', 'running_no_empty', 'walking_no_empty']):
    if data_set_name == 'KTH':
        norm_transform = VidNormalize(mean = 0.6013795, std = 2.7570653)
        renorm_transform = VidReNormalize(mean = 0.6013795, std = 2.7570653)
        train_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((64, 64)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), norm_transform])
        test_transform = transforms.Compose([VidCenterCrop((120, 120)), VidResize((64, 64)), VidToTensor(), norm_transform])

        val_person_ids = [random.randint(1, 17)]
        KTHTrainData = KTHDataset(data_set_dir, transform = train_transform, train = True, val = True, 
                                num_past_frames= 10, num_future_frames= 10, val_person_ids = val_person_ids, actions = KTH_actions)
        train_set, val_set = KTHTrainData()
        KTHTestData = KTHDataset(data_set_dir, transform = test_transform, train = False, val = False, 
                                num_past_frames= test_past_frames, num_future_frames= test_future_frames, actions = KTH_actions)
        test_set = KTHTestData()
    
    elif data_set_name == 'KITTI':
        norm_transform = VidNormalize((0.44812047, 0.47147775, 0.4677183),(1.5147436, 1.5871466, 1.5925455))
        renorm_transform = VidReNormalize((0.44811612, 0.47147346, 0.46771598),(1.5177081, 1.5897311, 1.5952978))
        train_transform = transforms.Compose([VidResize((128, 128)), VidRandomHorizontalFlip(0.5), VidRandomVerticalFlip(0.5), VidToTensor(), norm_transform])
        test_transform = transforms.Compose([VidResize((128, 128)), VidToTensor(), norm_transform])
        KITTITrainData = KITTIDataset(data_set_dir, [10, 11, 12, 13], transform = train_transform, train = True, val = True,
                                        num_past_frames= 4, num_future_frames= 5)
        train_set, val_set = KITTITrainData()
        KITTITrainData = KITTIDataset(data_set_dir, [10, 11, 12, 13], transform = test_transform, train = False, val = False,
                                        num_past_frames= test_past_frames, num_future_frames=test_future_frames)
        test_set = KITTITrainData()
    
    elif data_set_name == 'SMMNIST':
        renorm_transform = VidReNormalize(mean = 0., std = 1.0)
        train_transform = test_transform = VidToTensor()
        
        dataset_dir = Path(data_set_dir)
        train_set = StochasticMovingMNIST(True, dataset_dir, num_past_frames=5, num_future_frames=10, transform=train_transform)
        val_set = StochasticMovingMNIST(False, dataset_dir, num_past_frames=5, num_future_frames=10, transform=test_transform)
        test_set = StochasticMovingMNIST(False, dataset_dir, num_past_frames=test_past_frames, num_future_frames=test_future_frames, transform=test_transform)
    
    
    elif data_set_name == 'BAIR':
        dataset_dir = Path(data_set_dir)
        norm_transform = VidNormalize((0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673))
        renorm_transform = VidReNormalize((0.61749697, 0.6050092, 0.52180636), (2.1824553, 2.1553133, 1.9115673))
        #norm_transform = VidNormalize((0.6175636, 0.60508573, 0.52188003), (2.8584306, 2.8212209, 2.499153))
        #renorm_transform = VidReNormalize((0.6175636, 0.60508573, 0.52188003), (2.8584306, 2.8212209, 2.499153))
        transform = transforms.Compose([VidToTensor(), norm_transform])

        BAIR_train_whole_set = BAIRDataset(dataset_dir.joinpath('train'), transform, color_mode = 'RGB', 
                                num_past_frames = 2, num_future_frames = 10)()
        train_val_ratio = 0.95
        BAIR_train_set_length = int(len(BAIR_train_whole_set) * train_val_ratio)
        BAIR_val_set_length = len(BAIR_train_whole_set) - BAIR_train_set_length
        train_set, val_set = random_split(BAIR_train_whole_set, [BAIR_train_set_length, BAIR_val_set_length],
                                        generator=torch.Generator().manual_seed(2021))

        test_set = BAIRDataset(dataset_dir.joinpath('test'), transform, color_mode = 'RGB', 
                                num_past_frames = test_past_frames, num_future_frames = test_future_frames)()
    
    elif data_set_name == 'CityScapes':
        dataset_dir = Path(data_set_dir)
        norm_transform = VidNormalize((0.31604213, 0.35114038, 0.3104223),(1.2172801, 1.3219808, 1.2082524))
        renorm_transform = VidReNormalize((0.31604213, 0.35114038, 0.3104223),(1.2172801, 1.3219808, 1.2082524))
        transform = transforms.Compose([VidToTensor(), norm_transform])

        train_set = CityScapesDataset(dataset_dir.joinpath('train'), transform, color_mode = 'RGB', 
                                num_past_frames = 2, num_future_frames = 10)()
        val_set = CityScapesDataset(dataset_dir.joinpath('val'), transform, color_mode = 'RGB', 
                                num_past_frames = 2, num_future_frames = 10)()
        test_set = CityScapesDataset(dataset_dir.joinpath('test'), transform, color_mode = 'RGB', 
                                num_past_frames = test_past_frames, num_future_frames = test_future_frames)()

    if dev_set_size is not None:
        train_set, _ = random_split(train_set, [dev_set_size, len(train_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
        val_set, _ = random_split(val_set, [dev_set_size, len(val_set) - dev_set_size], generator=torch.Generator().manual_seed(2021))
    
    N = batch_size
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = True)
    val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = True)
    test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=num_workers, drop_last = False)

    if ngpus > 1:
        N = batch_size//ngpus
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

        train_loader = DataLoader(train_set, batch_size=N, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=train_sampler, drop_last = True)
        val_loader = DataLoader(val_set, batch_size=N, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=val_sampler, drop_last = True)

    return train_loader, val_loader, test_loader, renorm_transform

class KTHDataset(object):
    """
    KTH dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (120, 160)
    Split the KTH dataset and return the train and test dataset
    """
    def __init__(self, KTH_dir, transform, train, val,
                 num_past_frames, num_future_frames, actions=['boxing', 'handclapping', 'handwaving', 'jogging_no_empty', 'running_no_empty', 'walking_no_empty'], val_person_ids = None,
                 min_lo = None, max_lo = None):
        """
        Args:
            KTH_dir --- Directory for extracted KTH video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
        """
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = 'grey_scale'

        self.KTH_path = Path(KTH_dir).absolute()
        self.actions = actions
        self.train = train
        self.val = val
        if self.train:
            self.person_ids = list(range(1, 17))
            if self.val:
                if val_person_ids is None: #one person for the validation
                    self.val_person_ids = [random.randint(1, 17)]
                    self.person_ids.remove(self.val_person_ids[0])
                else:
                    self.val_person_ids = val_person_ids
        else:
            self.person_ids = list(range(17, 26))
        
        self.min_lo = min_lo
        self.max_lo = max_lo
        
        frame_folders = self.__getFramesFolder__(self.person_ids)
        self.clips = self.__getClips__(frame_folders)
        
        if self.val:
            val_frame_folders = self.__getFramesFolder__(self.val_person_ids)
            self.val_clips = self.__getClips__(val_frame_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        
        clip_set = ClipDataset(self.num_past_frames, self.num_future_frames, self.clips, self.transform, self.color_mode, self.min_lo, self.max_lo)
        if self.val:
            val_clip_set = ClipDataset(self.num_past_frames, self.num_future_frames, self.val_clips, self.transform, self.color_mode, self.min_lo, self.max_lo)
            return clip_set, val_clip_set
        else:
            return clip_set
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    
    def __getFramesFolder__(self, person_ids):
        """
        Get the KTH frames folders for ClipDataset
        Returns:
            return_folders --- ther returned video frames folders
        """

        frame_folders = []
        for a in self.actions:
            action_path = self.KTH_path.joinpath(a)
            frame_folders.extend([action_path.joinpath(s) for s in os.listdir(action_path) if '.avi' not in s])
        frame_folders = sorted(frame_folders)

        return_folders = []
        for ff in frame_folders:
            person_id = int(str(ff.name).strip().split('_')[0][-2:])
            if person_id in person_ids:
                return_folders.append(ff)

        return return_folders

class BAIRDataset(object):
    """
    BAIR dataset, a wrapper for ClipDataset
    the original frame size is (H, W) = (64, 64)
    The train and test frames has been previously splitted: ref "Self-Supervised Visual Planning with Temporal Skip Connections"
    """
    def __init__(self, frames_dir: str, transform, color_mode = 'RGB', 
                 num_past_frames = 10, num_future_frames = 10, min_lo = None, max_lo = None):
        """
        Args:
            frames_dir --- Directory of extracted video frames and original videos.
            transform --- trochvison transform functions
            color_mode --- 'RGB' or 'grey_scale' color mode for the dataset
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
            clip_length --- number of frames for each video clip example for model
            min_lo --- minimum length for observation, for later randomly selection of the observations
            max_lo --- maximum length for observation, for later randomly selection of the observations
        """
        self.frames_path = Path(frames_dir).absolute()
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = color_mode
        self.min_lo = min_lo
        self.max_lo = max_lo

        self.clips = self.__getClips__()


    def __call__(self):
        """
        Returns:
            data_set --- ClipDataset object
        """
        data_set = ClipDataset(self.num_past_frames, self.num_future_frames, self.clips, self.transform, self.color_mode, self.min_lo, self.max_lo)

        return data_set
    
    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            img_files = sorted(list(folder.glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips

class CityScapesDataset(BAIRDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __getClips__(self):
        clips = []
        frames_folders = os.listdir(self.frames_path)
        frames_folders = [self.frames_path.joinpath(s) for s in frames_folders]
        for folder in frames_folders:
            all_imgs = sorted(list(folder.glob('*')))
            obj_dict = {}
            for f in all_imgs:
                id = str(f).split('_')[1]
                if id in obj_dict:
                    obj_dict[id].append(f)
                else:
                    obj_dict[id] = [f]
            for k, img_files in obj_dict.items():
                for k, g in groupby(enumerate(img_files), lambda ix: ix[0]-int(str(ix[1]).split('_')[2])):
                    clip_files = list(list(zip(*list(g)))[1])
                    
                    clip_num = len(clip_files) // self.clip_length
                    rem_num = len(clip_files) % self.clip_length
                    clip_files = clip_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
                    for i in range(clip_num):
                        clips.append(clip_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips

class KITTIDataset(object):
    def __init__(self, KITTI_dir, test_folder_ids, transform, train, val,
                 num_past_frames, num_future_frames,
                 min_lo = None, max_lo = None):
        """
        Args:
            KITTI_dir --- Directory for extracted KITTI video frames
            train --- True for training dataset, False for test dataset
            transform --- trochvison transform functions
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
        """
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clip_length = num_past_frames + num_future_frames
        self.transform = transform
        self.color_mode = 'RGB'

        self.KITTI_path = Path(KITTI_dir).absolute()
        self.train = train
        self.val = val

        self.all_folders = sorted(os.listdir(self.KITTI_path))
        self.num_examples = len(self.all_folders)
        
        self.folder_id = list(range(self.num_examples))
        if self.train:
            self.train_folders = [self.all_folders[i] for i in range(self.num_examples) if i not in test_folder_ids]
            if self.val:
                self.val_folders = self.train_folders[0:2]
                self.train_folders = self.train_folders[2:]
    
        else:
            self.test_folders = [self.all_folders[i] for i in test_folder_ids]
        
        self.min_lo = min_lo
        self.max_lo = max_lo

        if self.train:
            self.train_clips = self.__getClips__(self.train_folders)
            if self.val:
                self.val_clips = self.__getClips__(self.val_folders)
        else:
            self.test_clips = self.__getClips__(self.test_folders)

    def __call__(self):
        """
        Returns:
            clip_set --- ClipDataset object
        """
        if self.train:
            clip_set = ClipDataset(self.num_past_frames, self.num_future_frames, self.train_clips, self.transform, self.color_mode, self.min_lo, self.max_lo)
            if self.val:
                val_clip_set = ClipDataset(self.num_past_frames, self.num_future_frames, self.val_clips, self.transform, self.color_mode, self.min_lo, self.max_lo)
                return clip_set, val_clip_set
            return clip_set
        else:
            return ClipDataset(self.num_past_frames, self.num_future_frames, self.test_clips, self.transform, self.color_mode, self.min_lo, self.max_lo)
    
    def __getClips__(self, frame_folders):
        clips = []
        for folder in frame_folders:
            img_files = sorted(list(self.KITTI_path.joinpath(folder).glob('*')))
            clip_num = len(img_files) // self.clip_length
            rem_num = len(img_files) % self.clip_length
            img_files = img_files[rem_num // 2 : rem_num//2 + clip_num*self.clip_length]
            for i in range(clip_num):
                clips.append(img_files[i*self.clip_length : (i+1)*self.clip_length])

        return clips
    

class ClipDataset(Dataset):
    """
    Video clips dataset
    """
    def __init__(self, num_past_frames, num_future_frames, clips, transform, color_mode, min_lo = None, max_lo = None):
        """
        Args:
            num_past_frames --- number of past frames
            num_future_frames --- number of future frames
            clips --- List of video clips frames file path
            transfrom --- torchvision transforms for the image
            color_mode --- 'RGB' for RGB dataset, 'grey_scale' for grey_scale dataset
            min_lo --- minimum length for observation, for later randomly selection of the observations
            max_lo --- maximum length for observation, for later randomly selection of the observations

        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.clips = clips
        self.transform = transform
        if color_mode != 'RGB' and color_mode != 'grey_scale':
            raise ValueError("Unsupported color mode!!")
        else:
            self.color_mode = color_mode
        
        self.min_lo = min_lo
        self.max_lo = max_lo

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_imgs = self.clips[index]
        imgs = []
        for img_path in clip_imgs:
            if self.color_mode == 'RGB':
                img = Image.open(img_path.absolute().as_posix()).convert('RGB')
            else:
                img = Image.open(img_path.absolute().as_posix()).convert('L')
            imgs.append(img)
        
        original_clip = self.transform(imgs)
        if self.min_lo is not None and self.max_lo is not None:
            return original_clip, self.min_lo, self.max_lo
        else:
            past_clip = original_clip[0:self.num_past_frames, ...]
            future_clip = original_clip[-self.num_future_frames:, ...]
            return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)
        
        videodims = img.size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')    
        video = cv2.VideoWriter(Path(file_name).absolute().as_posix(), fourcc, 10, videodims)
        for img in imgs:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        #imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])

class MovingMNISTDataset(Dataset):
    """
    MovingMNIST dataset
    """
    def __init__(self, data_path, transform, min_lo=None, max_lo=None):
        """
        both num_past_frames and num_future_frames are limited to be 10
        Args:
            data_path --- npz file path
            transfrom --- torchvision transforms for the image
            min_lo --- minimum length for observation, for later randomly selection of the observations
            max_lo --- maximum length for observation, for later randomly selection of the observations
        Return batched Sample:
            past_clip --- Tensor with shape (batch_size, num_past_frames, C, H, W)
            future_clip --- Tensor with shape (batch_size, num_future_frames, C, H, W)
        """
        self.data_path = data_path
        self.data = self.load_data()

        self.transform = transform
        self.min_lo = min_lo
        self.max_lo = max_lo
    
    def load_data(self):
        data = {}
        np_arr = np.load(self.data_path.absolute().as_posix())
        for key in np_arr:
            data[key] = np_arr[key]
        return data

    def __len__(self):
        return self.data['clips'].shape[1]
    
    def __getitem__(self, index: int):
        """
        Returns:
            past_clip: Tensor with shape (num_past_frames, C, H, W)
            future_clip: Tensor with shape (num_future_frames, C, H, W)
        """
        if torch.is_tensor(index):
            index = index.to_list()
        
        clip_index = self.data['clips'][:, index, :]
        psi, pei = clip_index[0, 0], clip_index[0, 0] + clip_index[0, 1]
        past_clip = self.data['input_raw_data'][psi:pei, ...]
        fsi, fei = clip_index[1, 0], clip_index[1, 0] + clip_index[1, 1]
        future_clip = self.data['input_raw_data'][fsi:fei, ...]

        full_clip = torch.from_numpy(np.concatenate((past_clip, future_clip), axis = 0))
        imgs = []
        for i in range(full_clip.shape[0]):
            img = transforms.ToPILImage()(full_clip[i, ...])
            imgs.append(img)
        
        full_clip = self.transform(imgs)
        if self.min_lo is not None and self.max_lo is not None:
            return full_clip, self.min_lo, self.max_lo
        else:
            past_clip = full_clip[0:clip_index[0, 1], ...]
            future_clip = full_clip[-clip_index[1, 1]:, ...]

            return past_clip, future_clip

    def visualize_clip(self, clip, file_name):
        """
        save a video clip to GIF file
        Args:
            clip: tensor with shape (clip_length, C, H, W)
        """
        imgs = []
        for i in range(clip.shape[0]):
            img = transforms.ToPILImage()(clip[i, ...])
            imgs.append(img)

        imgs[0].save(str(Path(file_name).absolute()), save_all = True, append_images = imgs[1:])

class StochasticMovingMNIST(Dataset):
    """https://github.com/edenton/svg/blob/master/data/moving_mnist.py"""
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, train_flag, data_root, num_past_frames, num_future_frames, transform, num_digits=2, image_size=64, deterministic=False, min_lo = None, max_lo = None):
        path = data_root
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.seq_len = num_past_frames + num_future_frames
        self.transform = transform
        self.num_digits = num_digits  
        self.image_size = image_size 
        self.step_length = 0.1
        self.digit_size = 32
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1 

        self.data = datasets.MNIST(
            path,
            train=train_flag,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data) 
        self.min_lo = min_lo
        self.max_lo = max_lo

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        full_clip = torch.from_numpy(self.__getnparray__(idx))
        imgs = []
        for i in range(full_clip.shape[0]):
            img = transforms.ToPILImage()(full_clip[i, ...])
            imgs.append(img)
        
        full_clip = self.transform(imgs)
        if self.min_lo is not None and self.max_lo is not None:
            return full_clip, self.min_lo, self.max_lo
        else:
            past_clip = full_clip[0:self.num_past_frames, ...]
            future_clip = full_clip[self.num_past_frames:, ...]

            return past_clip, future_clip

    def __getnparray__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size, 
                      image_size, 
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit, _ = self.data[idx]

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-32:
                    sy = image_size-32-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)
                    
                if sx < 0:
                    sx = 0 
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-32:
                    sx = image_size-32-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)
                   
                x[t, sy:sy+32, sx:sx+32, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x.transpose(0, 3, 1, 2)

class VidResize(object):
    def __init__(self, *args, **resize_kwargs):
        self.resize_kwargs = resize_kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Resize(*self.args, **self.resize_kwargs)(clip[i])

        return clip

class VidCenterCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.CenterCrop(*self.args, **self.kwargs)(clip[i])

        return clip

class VidCrop(object):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.functional.crop(clip[i], *self.args, **self.kwargs)

        return clip
        
class VidRandomHorizontalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.hflip(clip[i])
        return clip

class VidRandomVerticalFlip(object):
    def __init__(self, p: float):
        assert p>=0 and p<=1, "invalid flip probability"
        self.p = p
    
    def __call__(self, clip: List[Image.Image]):
        if np.random.rand() < self.p:
            for i in range(len(clip)):
                clip[i] = transforms.functional.vflip(clip[i])
        return clip

class VidToTensor(object):
    def __call__(self, clip: List[Image.Image]):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        for i in range(len(clip)):
            clip[i] = transforms.ToTensor()(clip[i])
        clip = torch.stack(clip, dim = 0)

        return clip

class VidNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = transforms.Normalize(self.mean, self.std)(clip[i, ...])

        return clip

class VidReNormalize(object):
    def __init__(self, mean, std):
        try:
            self.inv_std = [1.0/s for s in std]
            self.inv_mean = [-m for m in mean]
            self.renorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0.],
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = [1., 1., 1.])])
        except TypeError:
            #try normalize for grey_scale images.
            self.inv_std = 1.0/std
            self.inv_mean = -mean
            self.renorm = transforms.Compose([transforms.Normalize(mean = 0.,
                                                                std = self.inv_std),
                                            transforms.Normalize(mean = self.inv_mean,
                                                                std = 1.)])

    def __call__(self, clip: Tensor):
        """
        Return: clip --- Tensor with shape (T, C, H, W)
        """
        T, _, _, _ = clip.shape
        for i in range(T):
            clip[i, ...] = self.renorm(clip[i, ...])

        return clip

class VidPad(object):
    """
    If pad, Do not forget to pass the mask to the transformer encoder.
    """
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.args = args

    def __call__(self, clip: List[Image.Image]):
        for i in range(len(clip)):
            clip[i] = transforms.Pad(*self.args, **self.kwargs)(clip[i])

        return clip

def mean_std_compute(dataset, device, color_mode = 'RGB'):
    """
    arguments:
        dataset: pytorch dataloader
        device: torch.device('cuda:0') or torch.device('cpu') for computation
    return:
        mean and std of each image channel.
        std = sqrt(E(x^2) - (E(X))^2)
    """
    data_iter= iter(dataset)
    sum_img = None
    square_sum_img = None
    N = 0

    pgbar = tqdm(desc = 'summarizing...', total = len(dataset))
    for idx, sample in enumerate(data_iter):
        past, future = sample
        clip = torch.cat([past, future], dim = 0)
        N += clip.shape[0]

        img = torch.sum(clip, axis = 0)

        if idx == 0:
            sum_img = img
            square_sum_img = torch.square(img)
            sum_img = sum_img.to(torch.device(device))
            square_sum_img = square_sum_img.to(torch.device(device))
        else:
            img = img.to(device)
            sum_img = sum_img + img
            square_sum_img = square_sum_img + torch.square(img)
        
        pgbar.update(1)
    
    pgbar.close()

    mean_img = sum_img/N
    mean_square_img = square_sum_img/N
    if color_mode == 'RGB':
        mean_r, mean_g, mean_b = torch.mean(mean_img[0, :, :]), torch.mean(mean_img[1, :, :]), torch.mean(mean_img[2, :, :])
        mean_r2, mean_g2, mean_b2 = torch.mean(mean_square_img[0,:,:]), torch.mean(mean_square_img[1,:,:]), torch.mean(mean_square_img[2,:,:])
        std_r, std_g, std_b = torch.sqrt(mean_r2 - torch.square(mean_r)), torch.sqrt(mean_g2 - torch.square(mean_g)), torch.sqrt(mean_b2 - torch.square(mean_b))

        return ([mean_r.cpu().numpy(), mean_g.data.cpu().numpy(), mean_b.cpu().numpy()], [std_r.cpu().numpy(), std_g.cpu().numpy(), std_b.cpu().numpy()])
    else:
        mean = torch.mean(mean_img)
        mean_2 = torch.mean(mean_square_img)
        std = torch.sqrt(mean_2 - torch.square(mean))

        return (mean.cpu().numpy(), std.cpu().numpy())

if __name__ == '__main__':
    """
    transform = transforms.Compose([VidResize((128, 64)),
                                    VidCenterCrop((64, 64)),
                                    VidPad(padding = (20, 20), padding_mode='constant'),
                                    VidRandomHorizontalFlip(p=0.5),
                                    VidRandomVerticalFlip(p=0.5),
                                    VidToTensor()])
                                    
    kth_dataset = KTHDataset('/store/travail/xiyex/VideoFramePrediction/dataset/KTH/walking_no_empty', transform = transform, color_mode = 'RGB')
    train_set, test_set = kth_dataset()
    train_iter = iter(train_set)
    past_clip, future_clip = next(train_iter)
    train_set.visualize_clip(past_clip, 'past_clip_rgb.gif')
    train_set.visualize_clip(future_clip, 'future_clip_rgb.gif')

    
    print(sorted(kth_dataset.train_frame_folders))
    dataloader = DataLoader(train_set, batch_size=4,
                        shuffle=True, num_workers=0)
    for idx, sample in enumerate(dataloader, 0):
        past, future = sample
        print(past.shape)
        print(future.shape)
        break

    transform = transforms.Compose([VidCenterCrop((120, 120)),
                                    VidResize((64, 64)),
                                    VidToTensor()])
    kth_dataset = KTHDataset('/store/travail/xiyex/KTH', transform = transform, train = True, val = False, num_past_frames=10, num_future_frames=10)
    train_set = kth_dataset()
    print(len(train_set))
    #train_statistics = mean_std_compute(train_set, torch.device('cuda:0'), color_mode = 'grey_scale')
    #print(train_statistics) #mean 0.6013795, std 2.7570653
    kth_dataset1 = KTHDataset('/store/travail/xiyex/KTH', transform = transform, train = True, val = True, num_past_frames=10, num_future_frames=10)
    train_set1, val_set = kth_dataset1()
    print(len(train_set1), len(val_set))

    train_iter = iter(train_set)
    for i in range(100):
        past_clip, future_clip = next(train_iter)
    train_set.visualize_clip(past_clip, 'past_clip.gif')
    train_set.visualize_clip(future_clip, 'future_clip.gif')


    transform = transform = transforms.Compose([VidToTensor()])
    BAIR_dataset = BAIRDataset('/store/travail/xiyex/BAIR/softmotion30_44k/train', transform,
                                    color_mode = 'RGB', num_past_frames = 5, num_future_frames = 10)
    bair_train_set = BAIR_dataset()

    statistics = mean_std_compute(bair_train_set, torch.device('cuda:0'), color_mode = 'RGB')
    print(statistics)
    """
    
    """
    print(len(bair_train_set))
    train_iter = iter(bair_train_set)
    past_clip, future_clip = next(train_iter)
    print(past_clip.shape, future_clip.shape)
    bair_train_set.visualize_clip(past_clip, 'past_clip.gif')
    bair_train_set.visualize_clip(future_clip, 'future_clip.gif')
    
    
    #create_occlude_MovingMNIST(Path('/store/travail/xiyex/MovingMNIST').joinpath('moving-mnist-valid.npz'), Path('/store/travail/xiyex/MovingMNIST').joinpath('moving-mnist-valid-blockfp.npz'), block_size = 20)
    num_past_frames = 10
    num_future_frames = 10
    transform = transforms.Compose([VidToTensor()])
    whole_set = data.ConcatDataset([Human_train_set, Human_val_set])
    
    statistics = mean_std_compute(whole_set, torch.device('cuda:0'), color_mode = 'RGB')
    print(statistics)

    transform = transform = transforms.Compose([VidToTensor()])
    city_scapes_dataset = CityScapesDataset('/home/travail/xiyex/cityscapes/processed_seq/train', transform, color_mode = 'RGB', num_past_frames = 2, num_future_frames = 10)()

    statistics = mean_std_compute(city_scapes_dataset, torch.device('cuda:0'), color_mode = 'RGB')
    print(statistics)
    
    transform = transform = transforms.Compose([VidToTensor()])
    city_scapes_dataset = CityScapesDataset('/home/travail/xiyex/CityScapes/train', transform, color_mode = 'RGB', num_past_frames = 2, num_future_frames = 10)()
    """
    transform = transforms.Compose([VidResize((128, 128)), VidToTensor()])
    kitti_train = KITTIDataset('/home/travail/xiyex/KITTI_Processed', [10, 11], transform, True, False, 4, 5)()
    kitti_test = KITTIDataset('/home/travail/xiyex/KITTI_Processed', [10, 11], transform, False, False, 4, 5)()
    kitti_set = ConcatDataset([kitti_test, kitti_train])
    statistics = mean_std_compute(kitti_set, torch.device('cuda:0'), color_mode = 'RGB')
    print(statistics)

    