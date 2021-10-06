import os
from dotenv import load_dotenv
import time
import random
import wandb
import matplotlib.pyplot as plt
from UNet import UNet
from ZUNet_v1 import ZUNet_v1, ZUNet_v2
from engine import Segmentor, Segmentor_z, Segmentor_z_v2
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

import torchvision.utils as vutils
from dataloader import prep_dataloader, prep_dataloader_z, prep_dataloader_multiC_z
from medpy.io import load
import numpy as np
from torch import nn
from torch.cuda import amp
import torch
from torchsummary import summary

import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wandb_config():
    project = "silicosis"
    run_name = "ZUNet_v1_multiC_lung_n64"
    debug = False
    if debug:
        project = "debug"

    wandb.init(project=project, name=run_name)
    config = wandb.config
    # ENV
    if debug:
        config.epochs = 1
        config.n_case = 5
    else:
        config.epochs = 30
        # n_case = 0 to run all cases
        config.n_case = 64

    config.save = True

    config.data_path = os.getenv("VIDA_PATH")
    config.in_file = "ENV18PM_ProjSubjList_cleaned_IN.in"
    config.test_results_dir = "RESULTS"
    config.name = run_name
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config.mask = 'airway'
    config.mask = "lung"
    config.model = "ZUNet_v1_multiC"
    config.activation = "leakyrelu"
    config.optimizer = "adam"
    config.scheduler = "CosineAnnealingWarmRestarts"
    config.loss = "BCE+dice"
    config.bce_weight = 0.5

    config.learning_rate = 0.0001
    config.train_bs = 8
    config.valid_bs = 16
    config.aug = True
    config.Z = True

    return config


def prep_test_img(multiC=True):
    # Test
    test_img, _ = load("/data1/inqlee0704/silicosis/data/inputs/02_ct.hdr")
    test_img[test_img < -1024] = -1024
    if multiC:
        narrow_c = np.copy(test_img)
        wide_c = np.copy(test_img)
        narrow_c[narrow_c >= -500] = -500
        wide_c[wide_c >= 300] = 300
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        wide_c = (wide_c - np.min(wide_c)) / (np.max(wide_c) - np.min(wide_c))
        narrow_c = (narrow_c - np.min(narrow_c)) / (np.max(narrow_c) - np.min(narrow_c))
        narrow_c = narrow_c[None, :]
        wide_c = wide_c[None, :]
        test_img = test_img[None, :]
        test_img = np.concatenate([test_img, narrow_c, wide_c], axis=0)
    else:
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))
        test_img = test_img[None, :]
    return test_img


def volume_inference_multiC_z(model, volume, threshold=0.5):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    slices = np.zeros((512, 512, volume.shape[-1]))
    for i in range(volume.shape[-1]):
        s = volume[:, :, :, i]
        s = s.astype(np.single)
        s = torch.from_numpy(s).unsqueeze(0)
        z = i / (volume.shape[-1] + 1)
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


def show_images(test_img, test_pred, epoch):
    test_img = torch.from_numpy(test_img)
    test_img = test_img.permute(3, 0, 1, 2)
    test_img = test_img[:, 0, :, :]
    test_img = test_img.unsqueeze(1)

    test_pred = torch.from_numpy(test_pred)
    test_pred = test_pred.permute(2, 0, 1)
    test_pred = test_pred.unsqueeze(1)
    print(f"Test img: {test_img.shape}")
    print(f"Test pred: {test_pred.shape}")

    test_img_grid = vutils.make_grid(test_img)
    test_pred_grid = vutils.make_grid(test_pred)

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title(f"CT images")
    plt.imshow(test_img_grid.permute(1, 2, 0))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title(f"Lung masks at {epoch}")
    plt.imshow(test_pred_grid.permute(1, 2, 0))

    wandb.log({"plot": plt})
    plt.close()


if __name__ == "__main__":
    load_dotenv()
    seed_everything()
    config = wandb_config()

    scaler = amp.GradScaler()
    if config.Z:
        train_loader, valid_loader = prep_dataloader_multiC_z(config)
        model = ZUNet_v1(in_channels=3)
        # model = ZUNet_v2(in_channels=1)
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs, T_mult=1, eta_min=1e-8, last_epoch=-1
        )
        eng = Segmentor_z(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            # loss_fn=loss_fn,
            device=config.device,
            scaler=scaler,
        )
    else:
        train_loader, valid_loader = prep_dataloader(config)
        model = UNet()
        model.to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=config.epochs, T_mult=1, eta_min=1e-8, last_epoch=-1
        )
        eng = Segmentor(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            # loss_fn=loss_fn,
            device=config.device,
            scaler=scaler,
        )

    if config.save:
        dirname = f'{config.name}_{time.strftime("%Y%m%d", time.gmtime())}'
        out_dir = os.path.join("RESULTS", dirname)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{config.model}_{config.mask}.pth")

    best_loss = np.inf
    # Train
    test_img = prep_test_img(multiC=True)
    wandb.watch(eng.model, log="all", log_freq=10)
    for epoch in range(config.epochs):
        trn_loss, trn_dice_loss, trn_bce_loss = eng.train(train_loader)
        val_loss, val_dice_loss, val_bce_loss = eng.evaluate(valid_loader)
        wandb.log(
            {
                "epoch": epoch,
                "trn_loss": trn_loss,
                "trn_dice_loss": trn_dice_loss,
                "trn_bce_loss": trn_bce_loss,
                # "trn_cls_loss": trn_cls_loss,
                "val_loss": val_loss,
                "val_dice_loss": val_dice_loss,
                "val_bce_loss": val_bce_loss,
                # "val_cls_loss": val_cls_loss,
            }
        )
        test_pred = volume_inference_multiC_z(model, test_img, threshold=0.5)
        show_images(test_img, test_pred, epoch)

        if config.scheduler == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        eng.epoch += 1
        print(f"Epoch: {epoch}, train loss: {trn_loss:5f}, valid loss: {val_loss:5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"Best loss: {best_loss} at Epoch: {eng.epoch}")
            if config.save:
                torch.save(model.state_dict(), path)
                wandb.save(path)
