""" ****************************************** 
    Author: In Kyu Lee
    Deep learning dataloaders are stored here.
    Available:
    - ImageDataset: classification
    - SegDataset: Semantic segmentation
    - slice_loader: load slice information for SegDataset
    - CT_loader: load CT images
    - SlicesDataset: Semantic Segmentation (load all images into memory)
    - check_files: check if ct and mask file exist
****************************************** """
import os
from medpy.io import load
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import exposure


"""
ImageDataset for 3D CT image Segmentation
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
    - slices: slice information from slice_loader function [list]
Outputs:
    - dictionary that containts both image tensor & mask tensor [dict]

"""


class SegDataset:
    def __init__(self, subjlist, slices, mask_name=None, augmentations=None):
        self.subj_paths = subjlist.loc[:, "ImgDir"].values
        self.img_paths = [
            os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in self.subj_paths
        ]
        if mask_name == "airway":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
                for subj_path in self.subj_paths
            ]
        elif mask_name == "lung":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-lung.img.gz")
                for subj_path in self.subj_paths
            ]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.mask = None
        self.mask_name = mask_name
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, _ = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            self.mask, _ = load(self.mask_paths[slc[0]])
            self.pat_num = slc[0]
        img = self.img[:, :, slc[1]]
        mask = self.mask[:, :, slc[1]]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        # Airway mask is stored as 255
        if self.mask_name == "airway":
            mask = mask / 255
        elif self.mask_name == "lung":
            mask[mask == 20] = 1
            mask[mask == 30] = 1
        else:
            print("Specify mask_name (airway or lung)")
            return -1
        mask = mask.astype(int)
        img = img[None, :]
        mask = mask[None, :]

        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        return {"image": torch.tensor(img.copy()), "seg": torch.tensor(mask.copy())}


class SegDataset_histoEq:
    def __init__(
        self, subjlist, slices, mask_name=None, resize=None, augmentations=None
    ):
        self.subj_paths = subjlist.loc[:, "ImgDir"].values
        self.img_paths = [
            os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in self.subj_paths
        ]
        self.mask_paths = [
            os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
            for subj_path in self.subj_paths
        ]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.hdr = None
        self.mask = None
        self.hist_eq = None
        self.mask_name = mask_name
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, self.hdr = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            self.img = (self.img - np.min(self.img)) / (
                np.max(self.img) - np.min(self.img)
            )

            self.mask, _ = load(self.mask_paths[slc[0]])
            if self.mask_name == "airway":
                self.mask = self.mask / 255
            elif self.mask_name == "lung":
                self.mask[self.mask == 20] = 1
                self.mask[self.mask == 30] = 1
            else:
                print("Specify mask_name (airway or lung)")
                return -1

            self.pat_num = slc[0]
            self.hist_eq = exposure.equalize_hist(self.img)

        hist_eq = self.hist_eq[:, :, slc[1]]
        img = self.img[:, :, slc[1]]
        mask = self.mask[:, :, slc[1]]
        mask = mask.astype(int)
        hist_eq = hist_eq[None, :]
        img = img[None, :]
        mask = mask[None, :]
        img = np.concatenate([img, hist_eq], axis=0)
        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
        return {"image": torch.tensor(img.copy()), "seg": torch.tensor(mask.copy())}


class SegDataset_multiC_withZ:
    def __init__(
        self, subjlist, slices, mask_name=None, resize=None, augmentations=None
    ):
        self.subj_paths = subjlist.loc[:, "ImgDir"].values
        self.img_paths = [
            os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in self.subj_paths
        ]
        if mask_name == "airway":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
                for subj_path in self.subj_paths
            ]
        elif mask_name == "lung":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-lung.img.gz")
                for subj_path in self.subj_paths
            ]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.hdr = None
        self.mask = None
        self.mask_name = mask_name
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, self.hdr = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            self.airway_c = np.copy(self.img)
            self.tissue_c = np.copy(self.img)
            self.airway_c[self.airway_c >= -800] = -800
            self.tissue_c[self.tissue_c <= -200] = -200
            self.tissue_c[self.tissue_c >= 200] = 200

            self.img = (self.img - np.min(self.img)) / (
                np.max(self.img) - np.min(self.img)
            )
            self.tissue_c = (self.tissue_c - np.min(self.tissue_c)) / (
                np.max(self.tissue_c) - np.min(self.tissue_c)
            )
            self.airway_c = (self.airway_c - np.min(self.airway_c)) / (
                np.max(self.airway_c) - np.min(self.airway_c)
            )

            self.mask, _ = load(self.mask_paths[slc[0]])
            if self.mask_name == "airway":
                self.mask = self.mask / 255
            elif self.mask_name == "lung":
                self.mask[self.mask == 20] = 1
                self.mask[self.mask == 30] = 1
            else:
                print("Specify mask_name (airway or lung)")
                return -1

            self.pat_num = slc[0]

        z = slc[1] / (self.img.shape[2] + 1)
        # z ranges from 0 to 9
        z = np.floor(z * 10)

        airway_c = self.airway_c[:, :, slc[1]]
        tissue_c = self.tissue_c[:, :, slc[1]]
        img = self.img[:, :, slc[1]]
        mask = self.mask[:, :, slc[1]]
        mask = mask.astype(int)

        airway_c = airway_c[None, :]
        tissue_c = tissue_c[None, :]
        img = img[None, :]
        mask = mask[None, :]
        img = np.concatenate([img, airway_c, tissue_c], axis=0)

        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        return {
            "image": torch.tensor(img.copy()),
            "seg": torch.tensor(mask.copy()),
            "z": torch.tensor(z, dtype=torch.int64),
        }


class SegDataset_withZ:
    def __init__(
        self, subjlist, slices, mask_name=None, resize=None, augmentations=None
    ):
        self.subj_paths = subjlist.loc[:, "ImgDir"].values
        self.img_paths = [
            os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in self.subj_paths
        ]
        if mask_name == "airway":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
                for subj_path in self.subj_paths
            ]
        elif mask_name == "lung":
            self.mask_paths = [
                os.path.join(subj_path, "ZUNU_vida-lung.img.gz")
                for subj_path in self.subj_paths
            ]
        self.slices = slices
        self.pat_num = None
        self.img = None
        self.hdr = None
        self.mask = None
        self.mask_name = mask_name
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slc = self.slices[idx]
        if self.pat_num != slc[0]:
            self.img, self.hdr = load(self.img_paths[slc[0]])
            # some background values are set to -3024
            self.img[self.img < -1024] = -1024
            self.img = (self.img - np.min(self.img)) / (
                np.max(self.img) - np.min(self.img)
            )
            self.mask, _ = load(self.mask_paths[slc[0]])
            if self.mask_name == "airway":
                self.mask = self.mask / 255
            elif self.mask_name == "lung":
                self.mask[self.mask == 20] = 1
                self.mask[self.mask == 30] = 1
            else:
                print("Specify mask_name (airway or lung)")
                return -1

            self.pat_num = slc[0]

        z = slc[1] / (self.img.shape[2] + 1)
        # z ranges from 0 to 9
        z = np.floor(z * 10)
        img = self.img[:, :, slc[1]]
        mask = self.mask[:, :, slc[1]]

        mask = mask.astype(int)
        img = img[None, :]
        mask = mask[None, :]

        if self.augmentations is not None:
            augmented = self.augmentations(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        return {
            "image": torch.tensor(img.copy()),
            "seg": torch.tensor(mask.copy()),
            "z": torch.tensor(z, dtype=torch.int64),
        }


"""
Slice loader which outputs slice information for each CT
and check if mask dimension and image dimension match
Inputs:
    - subjlist: panda's dataframe which contains image & mask paths [df]
Outputs:
    - A slice list of tuples, [list]
        - first index represent subject's number
        - second index represent axial position of CT
    ex) (0,0),(0,1),(0,2) ... (0,750),(1,0),(1,1) ... (300, 650)
"""


def slice_loader(subjlist):
    print("Loading Data")
    subj_paths = subjlist.loc[:, "ImgDir"].values
    img_paths = [
        os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in subj_paths
    ]
    mask_paths = [
        os.path.join(subj_path, "ZUNU_vida-airtree.img.gz") for subj_path in subj_paths
    ]
    slices = []
    for ii in range(len(mask_paths)):
        label, _ = load(mask_paths[ii])
        img, _ = load(img_paths[ii])
        if img.shape != label.shape:
            print("Dimension does not match: ")
            print(subjlist.loc[ii, "ImgDir"])
        for jj in range(label.shape[2]):
            slices.append((ii, jj))
    return slices


"""
CT_loader is noramlly used for inference to load a CT dataset
Inputs:
    - subjlist: pd.df which contains image & mask paths [df]
Outputs:
    - A dictionary that contains 3D image & mask.
"""


def CT_loader(subjlist, mask_name=None):
    print("Loading Data. . . ")
    if type(subjlist) == list:
        subj_paths = subjlist
    else:
        subj_paths = subjlist.loc[:, "ImgDir"].values
    dicom_paths = [
        os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in subj_paths
    ]
    if mask_name == "lung":
        airmask_paths = [
            os.path.join(subj_path, "ZUNU_vida-lung.img.gz") for subj_path in subj_paths
        ]
    elif mask_name == "airway":
        airmask_paths = [
            os.path.join(subj_path, "ZUNU_vida-airtree.img.gz")
            for subj_path in subj_paths
        ]
    subj_ID = subjlist.loc[:, "Subj"].values
    img_ID = subjlist.loc[:, "Img"].values
    out = []
    for ii in range(len(dicom_paths)):
        image, _ = load(dicom_paths[ii])
        # some background values are set to -3024
        image[image < -1024] = -1024
        label, _ = load(airmask_paths[ii])
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = (image-(-1250))/((250)-(-1250))
        if mask_name == "airway":
            label = label / 255
        elif mask_name == "lung":
            label[label == 20] = 1
            label[label == 30] = 1
        else:
            print("Please specify mask_name (airway or lung)")
        label = label.astype(int)
        ID = subj_ID[ii] + "_" + img_ID[ii]
        if ii % 10 == 0:
            print("Processed: " + str(100 * ii / len(subjlist)) + "%")
        out.append({"image": image, "seg": label, "SubjID": ID})
    return np.array(out)


def TE_loader(subjlist, mask_name=None):
    print("Loading Data. . . ")
    dicom_paths = subjlist.loc[:, "ImgDir"].values
    out = []
    for ii in range(len(dicom_paths)):
        image, _ = load(dicom_paths[ii])
        image[image < -1024] = -1024
        image = (image - (np.min(image))) / ((np.max(image) - (np.min(image))))
        out.append({"image": image})
    return np.array(out)


"""
Check files before the train

"""


def check_files(subjlist):
    subj_paths = subjlist.loc[:, "ImgDir"].values
    img_path_ = [
        os.path.join(subj_path, "zunu_vida-ct.img") for subj_path in subj_paths
    ]
    mask_path_ = [
        os.path.join(subj_path, "ZUNU_vida-airtree.img.gz") for subj_path in subj_paths
    ]
    for i in range(len(img_path_)):
        if not os.path.exists(img_path_[i]):
            print(img_path_[i], "Not exists")
        if not os.path.exists(mask_path_[i]):
            print(mask_path_[i], "Not exists")


"""
Prepare train & valid dataloaders
"""


def prep_dataloader(c):
    # n_case: load n number of cases, 0: load all
    df_subjlist = pd.read_csv(os.path.join(c.data_path, c.in_file), sep="\t")
    n_case = c.n_case
    if n_case == 0:
        df_train, df_valid = model_selection.train_test_split(
            df_subjlist, test_size=0.2, random_state=42, stratify=None
        )
    else:
        df_train, df_valid = model_selection.train_test_split(
            df_subjlist[:n_case], test_size=0.2, random_state=42, stratify=None
        )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_slices = slice_loader(df_train)
    valid_slices = slice_loader(df_valid)
    # train_ds = SegDataset_multiC_withZ(df_train,
    #                         train_slices,
    #                         mask_name=c.mask,
    #                         augmentations=get_train_aug())
    # valid_ds = SegDataset_multiC_withZ(df_valid, valid_slices, mask_name=c.mask)
    train_ds = SegDataset(
        df_train, train_slices, mask_name=c.mask, augmentations=get_train_aug()
    )
    valid_ds = SegDataset(df_valid, valid_slices, mask_name=c.mask)
    train_loader = DataLoader(
        train_ds, batch_size=c.train_bs, shuffle=False, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=c.valid_bs, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader


def prep_dataloader_z(c):
    # n_case: load n number of cases, 0: load all
    df_subjlist = pd.read_csv(os.path.join(c.data_path, c.in_file), sep="\t")
    n_case = c.n_case
    if n_case == 0:
        df_train, df_valid = model_selection.train_test_split(
            df_subjlist, test_size=0.2, random_state=42, stratify=None
        )
    else:
        df_train, df_valid = model_selection.train_test_split(
            df_subjlist[:n_case], test_size=0.2, random_state=42, stratify=None
        )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_slices = slice_loader(df_train)
    valid_slices = slice_loader(df_valid)
    train_ds = SegDataset_multiC_withZ(
        df_train, train_slices, mask_name=c.mask, augmentations=get_train_aug()
    )
    valid_ds = SegDataset_multiC_withZ(df_valid, valid_slices, mask_name=c.mask)

    train_loader = DataLoader(
        train_ds, batch_size=c.train_bs, shuffle=False, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=c.valid_bs, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader


def get_train_aug():
    return A.Compose(
        [
            A.Rotate(limit=10),
            A.OneOf(
                [
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=5),
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=(3, 7)),
                ],
                p=0.5,
            ),
            # ToTensorV2()
        ]
    )


# def get_valid_aug():
#     return A.Compose([
#         A.OneOf([
#             A.HorizontalFlip(),
#             A.VerticalFlip(),
#         ],p=0.4),
#     ])