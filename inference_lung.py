import sys
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# from RecursiveUNet import RecursiveUNet
from UNet import UNet
from ZUNet_v1 import ZUNet_v1
from dataloader import TE_loader
import torch
import nibabel as nib


def volume_inference(model, volume, threshold=0.5):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    slices = np.zeros(volume.shape)
    for i in range(volume.shape[2]):
        s = volume[:, :, i]
        s = s.astype(np.single)
        s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        pred = model(s.to(DEVICE))
        pred = torch.sigmoid(pred)
        pred = np.squeeze(pred.cpu().detach())
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        slices[:, :, i] = pred
        # slices[:, :, i] = torch.argmax(pred, dim=0)
    return slices


def volume_inference_z(model, volume, threshold=0.5):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    slices = np.zeros(volume.shape)
    for i in range(volume.shape[2]):
        s = volume[:, :, i]
        s = s.astype(np.single)
        s = torch.from_numpy(s).unsqueeze(0).unsqueeze(0)
        z = i / (volume.shape[2] + 1)
        z = np.floor(z * 10)
        z = torch.tensor(z, dtype=torch.int64)
        pred = model(s.to(DEVICE), z.to(DEVICE))
        pred = torch.sigmoid(pred)
        pred = np.squeeze(pred.cpu().detach())
        pred[pred > threshold] = 1
        pred[pred <= threshold] = 0
        slices[:, :, i] = pred * 255
        # slices[:, :, i] = torch.argmax(pred, dim=0)
    return slices


if __name__ == "__main__":
    infer_path = "data/ProjSubjList.in"
    infer_list = pd.read_csv(infer_path)
    parameter_path = (
        # "/data1/inqlee0704/silicosis/RESULTS/UNet_64_20211002/lung_UNet.pth"
        "/data1/inqlee0704/silicosis/RESULTS/ZUNet_64_lung_20211001/ZUNet_lung.pth"
    )
    # parameter_path = '/home/inqlee0704/src/DL/airway/RESULTS/Recursive_UNet_v2_20201216/model.pth'
    # model = UNet()
    model = ZUNet_v1(in_channels=1)
    model.load_state_dict(torch.load(parameter_path))
    DEVICE = "cuda"
    model.to(DEVICE)
    test_data = TE_loader(infer_list)
    model.eval()
    print("Inference . . .")
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    for i, x in pbar:
        pred_label = volume_inference_z(model, x["image"])
        hdr = nib.Nifti1Header()
        pair_img = nib.Nifti1Pair(pred_label, np.eye(4), hdr)
        nib.save(
            pair_img,
            "data/lung_mask/ZUNet_n64/"
            + str(infer_list.loc[i, "ImgDir"][-9:-7])
            + ".img.gz",
        )
        # break
        # pred_img = nib.Nifti1Image(pred_label, affine=np.eye(4))
        # pred_img.to_filename('lung_mask2/'+str(infer_list.loc[i,'ImgDir'][7:9])+'.nii.gz')
