import os
import torch
import numpy as np
import random
import glob
import torch.utils.data as data
from PIL import Image, ImageFile
from torchvision.transforms.functional import hflip, to_tensor
ImageFile.LOAD_TRUNCATED_IMAGES = True
exts = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']

class Pretrain_Dataset(data.Dataset):
    def __init__(self, opt, mode=None):
        super(Pretrain_Dataset, self).__init__()
        self.train_dataset_dir = opt['pretrain_dataset']
        self.test_dataset_dir = opt['pretrain_testset']
        self.cropsize = opt['cropsize']
        if mode == None:
            self.mode = opt['mode']
        else:
            self.mode = mode
        self.paths = []
        self.paths_test = []
        for ext in exts:
            if self.mode == 'train':
                self.paths += glob.glob(os.path.join(self.train_dataset_dir, f'*.{ext}'))
            elif self.mode == 'test':
                self.paths += glob.glob(os.path.join(self.test_dataset_dir, f'*.{ext}'))
            else:
                raise NotImplementedError('mode [{:s}] is not recognized.'.format(self.mode))

    def __len__(self):
        return len(self.paths)

    def _get_crop_params(self, img):
        w, h = img.size
        top = random.randint(0, h - self.cropsize)
        left = random.randint(0, w - self.cropsize)
        return top, left

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        # crop if training
        if self.mode == 'train':
            top, left = self._get_crop_params(img)
            region = (left, top, left + self.cropsize, top + self.cropsize)
            img = img.crop(region)
        # horizontal flip
        if random.random() < 0.5 and self.mode == 'train':
            img = hflip(img)
        # to tensor
        img = to_tensor(img)
        return img