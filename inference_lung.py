import sys
sys.path.insert(0,'../DL')
import os
import pandas as pd
import numpy as np
from models import RecursiveUNet
from dataloader import TE_loader
import torch
from engine import volume_inference 
from metrics import Dice3d
import nibabel as nib

if __name__=='__main__':
    infer_path='ProjSubjList.in'
    infer_list = pd.read_csv(infer_path)
    # parameter_path = '/home/inqlee0704/LungSeg/train/RESULTS/2020-12-18_2140_Recursive_U_Net_vessel/model.pth'
    parameter_path = '/home/inqlee0704/LungSeg/train/RESULTS/2020-11-18_0510_U_Net/model.pth'
    # parameter_path = '/home/inqlee0704/src/DL/airway/RESULTS/Recursive_UNet_v2_20201216/model.pth'
    model = RecursiveUNet()
    model.load_state_dict(torch.load(parameter_path))
    DEVICE = 'cuda'
    model.to(DEVICE)
    test_data = TE_loader(infer_list)
    model.eval()
    print('Inference . . .')
    for i,x in enumerate(test_data):
        pred_label = volume_inference(model,x['image'])
        hdr = nib.Nifti1Header()
        pair_img = nib.Nifti1Pair(pred_label, np.eye(4),hdr)
        nib.save(pair_img,'vessel_mask/'+str(infer_list.loc[i,'ImgDir'][7:9])+'.img.gz')
        #pred_img = nib.Nifti1Image(pred_label, affine=np.eye(4))  
        #pred_img.to_filename('lung_mask2/'+str(infer_list.loc[i,'ImgDir'][7:9])+'.nii.gz')

