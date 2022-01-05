import os
from dotenv import load_dotenv
import time
import random
import wandb
from engine import Segmentor
from UNet_3D import UNet3D

from sklearn import model_selection
from dataloader import LungDataset_3D

import numpy as np
import pandas as pd
from torch import nn
from torch.cuda import amp
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau
from torchsummary import summary

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def wandb_config():
    project = 'airway'
    run_name = '3D_UNet_num_downs3'
    debug = False
    if debug:
        project = 'debug'

    wandb.init(project=project,name=run_name)
    config = wandb.config
    # ENV
    if debug:
        config.epochs = 40
        config.n_case = 64
    else:
        config.epochs = 30
    # n_case = 0 to run all cases
        config.n_case = 0
    config.save = True
    
    config.data_path = os.getenv('VIDA_PATH')
    # config.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
    config.in_file = 'ENV18PM_ProjSubjList_dim128_IN_train.in'
    config.test_results_dir = "RESULTS"
    config.name = run_name
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.mask = 'airway'
    config.model = '3D_UNet_num_downs3'
    config.activation = 'leakyrelu'
    config.optimizer = 'adam'
    config.scheduler = 'CosineAnnealingWarmRestarts'
    config.loss = 'BCE+dice'
    config.bce_weight = 0.5

    config.learning_rate = 0.0001
    config.train_bs = 4
    config.valid_bs = 8
    config.aug = True

    return config

if __name__ == "__main__": 
    load_dotenv()
    seed_everything()
    config = wandb_config()
    # Data
    df = pd.read_csv(os.path.join(config.data_path,config.in_file),sep='\t')
    if config.n_case != 0:
        df_train, df_valid = model_selection.train_test_split(
             df[:config.n_case],
             test_size=0.2,
             random_state=42,
             stratify=None)
    else:
        df_train, df_valid = model_selection.train_test_split(
             df,
             test_size=0.2,
             random_state=42,
             stratify=None)
    train_ds = LungDataset_3D(df_train, mask_name=config.mask)
    valid_ds = LungDataset_3D(df_valid, mask_name=config.mask)
    train_loader = DataLoader(train_ds, batch_size=config.train_bs,shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=config.valid_bs,shuffle=True)

    # Model #
    model = UNet3D(num_classes=1, in_channels=1, num_downs=3)
    model.to(config.device)
    # summary(model,(1,512,512),3)

    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                T_0=config.epochs, 
                                                T_mult=1, 
                                                eta_min=1e-8,
                                                last_epoch=-1)

    scaler = amp.GradScaler()
    eng = Segmentor(model=model, 
                    optimizer=optimizer,
                    scheduler=scheduler,
                    # loss_fn=loss_fn,
                    device=config.device,
                    scaler=scaler)

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join('RESULTS',dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "airway_UNet.pth")

    best_loss = np.inf
    # Train
    wandb.watch(eng.model,log='all',log_freq=10)
    for epoch in range(config.epochs):
        trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
        val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
        wandb.log({'epoch': epoch,
         'trn_loss': trn_loss, 'trn_dice_loss': trn_dice_loss, 'trn_bce_loss': trn_bce_loss,
         'val_loss': val_loss, 'val_dice_loss': val_dice_loss, 'val_bce_loss': val_bce_loss})

        if config.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        eng.epoch += 1
        print(f'Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}')
        if val_loss < best_loss:
            best_loss = val_loss
            print(f'Best loss: {best_loss} at Epoch: {eng.epoch}')
            if config.save:
                torch.save(model.state_dict(), path)
                wandb.save(path)
