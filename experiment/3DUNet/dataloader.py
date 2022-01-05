import os
from medpy.io import load
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2


"""
ImageDataset for 3D CT image Segmentation
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
    - slices: slice information from slice_loader function [list]
Outputs:
    - dictionary that containts both image tensor & mask tensor [dict]

"""
class LungDataset_3D:
    def __init__(self,subjlist, mask_name='None', augmentations=None):
        self.subj_paths = subjlist.loc[:,'ImgDir'].values
        self.img_paths = [os.path.join(subj_path,'zunu_vida-ct.img') for subj_path in self.subj_paths]
        if mask_name == 'airway':
            self.mask_paths = [os.path.join(subj_path,'ZUNU_vida-airtree.img.gz') for subj_path in self.subj_paths]
        elif mask_name == 'lung':
            self.mask_paths = [os.path.join(subj_path,'ZUNU_vida-lung.img.gz') for subj_path in self.subj_paths]
        else:
            print('Specify mask_name to either airway or lung')
            return
        self.pat_num = None
        self.img = None
        self.mask = None
        self.mask_name = mask_name
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img, hdr = load(self.img_paths[idx])
        mask, hdr = load(self.mask_paths[idx])
        img[img<-1024] = -1024
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        # Airway mask is stored as 255
        if self.mask_name=='airway':
            mask = mask/255
        elif self.mask_name=='lung':
            mask[mask==20] = 1
            mask[mask==30] = 1
        else:
            print('Specify mask_name (airway or lung)')
            return -1
        mask = mask.astype(int)
        img = img[None,:]
        mask = mask[None,:]
        if self.augmentations is not None:
            augmented = self.augmentations(image=img,mask=mask)
            img,mask = augmented['image'], augmented['mask']

        return {
                'image': torch.tensor(img.copy()),
                'seg': torch.tensor(mask.copy())
                }


"""
Prepare train & valid dataloaders
"""
def prep_dataloader(c):
# n_case: load n number of cases, 0: load all
    df_subjlist = pd.read_csv(os.path.join(c.data_path,c.in_file),sep='\t')
    n_case = c.n_case
    if n_case==0:
        df_train, df_valid = model_selection.train_test_split(
                df_subjlist,
                test_size=0.2,
                random_state=42,
                stratify=None)
    else:
        df_train, df_valid = model_selection.train_test_split(
             df_subjlist[:n_case],
             test_size=0.2,
             random_state=42,
             stratify=None)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_slices = slice_loader(df_train)
    valid_slices = slice_loader(df_valid)
    train_ds = SegDataset_multiC_withZ(df_train,
                            train_slices,
                            mask_name=c.mask,
                            augmentations=get_train_aug())
    valid_ds = SegDataset_multiC_withZ(df_valid, valid_slices, mask_name=c.mask)
    train_loader = DataLoader(train_ds,
                                batch_size=c.train_bs,
                                shuffle=False,
                                num_workers=0)
    valid_loader = DataLoader(valid_ds,
                                batch_size=c.valid_bs,
                                shuffle=False,
                                num_workers=0)

    return train_loader, valid_loader

def get_train_aug():
    return A.Compose([
        A.Rotate(limit=10),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ],p=0.5),
        A.OneOf([
            A.Blur(blur_limit=5),
            A.MotionBlur(blur_limit=7),
            A.GaussianBlur(blur_limit=(3,7)),
        ],p=0.5),
        # ToTensorV2()
    ])

# def get_valid_aug():
#     return A.Compose([
#         A.OneOf([
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#         ],p=0.4),
#     ])
