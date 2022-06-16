import pandas as pd
import numpy as np
import argparse
import os
from medpy.io import load
from tqdm.auto import tqdm
from utils.DCM2IMG import DCMtoVidaCT
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)


parser = argparse.ArgumentParser(description='extract QCT')
parser.add_argument('--ProjSubj', default='', type=str, help='path to ProjSubjList.in')
parser.add_argument('--save_path', default='', type=str, help='save path')
parser.add_argument('--mask_name', default='ZUNU_vida-lobes.img.gz', type=str, help='mask file name')
# ZUNU_lobes.img.gz
def get_HAA(img,lobe):
    # HAA Calculation
    l_threshold = -700
    u_threshold = 0
    # prepare .img
    HAA_img = np.zeros((img.shape),dtype='uint8')
    HAA_img[(l_threshold<=img)&(img<=u_threshold)] = 1
    # 0 if outside lobe
    HAA_img[lobe==0] = 0

    IN_l0 = len(img[lobe==8])
    IN_l1 = len(img[lobe==16])
    IN_l2 = len(img[lobe==32])
    IN_l3 = len(img[lobe==64])
    IN_l4 = len(img[lobe==128])
    IN_t = IN_l0 + IN_l1 + IN_l2 + IN_l3 + IN_l4

    HAA_l0 = HAA_img[lobe==8].sum()
    HAA_l1 = HAA_img[lobe==16].sum()
    HAA_l2 = HAA_img[lobe==32].sum()
    HAA_l3 = HAA_img[lobe==64].sum()
    HAA_l4 = HAA_img[lobe==128].sum()
    HAA_t = HAA_l0 + HAA_l1 + HAA_l2 + HAA_l3 + HAA_l4

    HAA_stat = pd.DataFrame({'Lobes':['Lobe0','Lobe1','Lobe2','Lobe3','Lobe4','total'],
                'HAAratio':np.float16([HAA_l0/IN_l0,HAA_l1/IN_l1,HAA_l2/IN_l2,HAA_l3/IN_l3,HAA_l4/IN_l4,HAA_t/IN_t]),
                'voxels_HAA':[HAA_l0,HAA_l1,HAA_l2,HAA_l3,HAA_l4,HAA_t],
                'Voxels':[IN_l0,IN_l1,IN_l2,IN_l3,IN_l4,IN_t]})
    return HAA_stat

def get_emph(img,lobe):
    # emphysema Calculation
    emphy_threshold = -950
    u_threshold = 0
    # prepare .img
    emphy_img = np.zeros((img.shape),dtype='uint8')
    # 2 if Emphysema
    emphy_img[(img<emphy_threshold)] = 2
    # 0 if outside lobe
    emphy_img[lobe==0] = 0

    # prepare emphysema & fsad stats
    IN_l0 = len(img[lobe==8])
    IN_l1 = len(img[lobe==16])
    IN_l2 = len(img[lobe==32])
    IN_l3 = len(img[lobe==64])
    IN_l4 = len(img[lobe==128])
    IN_t = IN_l0 + IN_l1 + IN_l2 + IN_l3 + IN_l4

    emphy_l0 = len(emphy_img[(lobe==8)&(emphy_img==2)])
    emphy_l1 = len(emphy_img[(lobe==16)&(emphy_img==2)])
    emphy_l2 = len(emphy_img[(lobe==32)&(emphy_img==2)])
    emphy_l3 = len(emphy_img[(lobe==64)&(emphy_img==2)])
    emphy_l4 = len(emphy_img[(lobe==128)&(emphy_img==2)])
    emphy_t = emphy_l0 + emphy_l1 + emphy_l2 + emphy_l3 + emphy_l4

    emphy_stat = pd.DataFrame({'Lobes':['Lobe0','Lobe1','Lobe2','Lobe3','Lobe4','Total'],
                'Emphysratio':np.float16([emphy_l0/IN_l0,emphy_l1/IN_l1,emphy_l2/IN_l2,emphy_l3/IN_l3,emphy_l4/IN_l4,emphy_t/IN_t]),
                'voxels_Emphys':[emphy_l0,emphy_l1,emphy_l2,emphy_l3,emphy_l4,emphy_t],
                'VoxelsAll':[IN_l0,IN_l1,IN_l2,IN_l3,IN_l4,IN_t]})
    return emphy_stat

def get_tiss_frac(img,lobe):
    # tissue fraction
    # prepare .img
    # density_img = np.zeros((img.shape),dtype='uint8')
    tiss_frac = (img+1000)/1055
    # 0 if outside lobe
    tiss_frac[lobe==0] = 0

    tiss_frac_l0 = tiss_frac[lobe==8].mean()
    tiss_frac_l1 = tiss_frac[lobe==16].mean()
    tiss_frac_l2 = tiss_frac[lobe==32].mean()
    tiss_frac_l3 = tiss_frac[lobe==64].mean()
    tiss_frac_l4 = tiss_frac[lobe==128].mean()
    tiss_frac_t = tiss_frac[lobe!=0].mean()

    tiss_frac_stat = pd.DataFrame({'Lobes':['Lobe0','Lobe1','Lobe2','Lobe3','Lobe4','total'],
                'meanTF':np.float16([tiss_frac_l0,tiss_frac_l1,tiss_frac_l2,tiss_frac_l3,tiss_frac_l4,tiss_frac_t]),
                    })
    return tiss_frac_stat

def get_total_volume(lobe, voxel_spacing):
    # total volume
    # prepare .img
    vol_vox = voxel_spacing[0]*voxel_spacing[1]*voxel_spacing[2]
    total_vol_l0 = (lobe==8).sum()*vol_vox
    total_vol_l1 = (lobe==16).sum()*vol_vox
    total_vol_l2 = (lobe==32).sum()*vol_vox
    total_vol_l3 = (lobe==64).sum()*vol_vox
    total_vol_l4 = (lobe==128).sum()*vol_vox
    total_vol_t = (lobe!=0).sum()*vol_vox


    total_vol_stat = pd.DataFrame({'Lobes':['Lobe0','Lobe1','Lobe2','Lobe3','Lobe4','total'],
                'totalVol':np.float32([total_vol_l0,total_vol_l1,total_vol_l2,total_vol_l3,total_vol_l4,total_vol_t]),
                    })
    return total_vol_stat

def get_tiss_volume(tiss_frac_stat,total_vol_stat):
    tiss_vol_l0 = tiss_frac_stat.meanTF[0]*total_vol_stat.totalVol[0]
    tiss_vol_l1 = tiss_frac_stat.meanTF[1]*total_vol_stat.totalVol[1]
    tiss_vol_l2 = tiss_frac_stat.meanTF[2]*total_vol_stat.totalVol[2]
    tiss_vol_l3 = tiss_frac_stat.meanTF[3]*total_vol_stat.totalVol[3]
    tiss_vol_l4 = tiss_frac_stat.meanTF[4]*total_vol_stat.totalVol[4]
    tiss_vol_t = tiss_frac_stat.meanTF[5]*total_vol_stat.totalVol[5]

    tiss_vol_stat = pd.DataFrame({'Lobes':['Lobe0','Lobe1','Lobe2','Lobe3','Lobe4','total'],
                'tissVol':np.float32([tiss_vol_l0,tiss_vol_l1,tiss_vol_l2,tiss_vol_l3,tiss_vol_l4,tiss_vol_t]),
                    })
    return tiss_vol_stat

def get_CT_density(img,lobe):
    # CT_density Calculation
    # prepare .img
    # density_img = np.zeros((img.shape),dtype='uint8')
    density_img = img
    # 0 if outside lobe
    density_img[lobe==0] = 0

    density_l0 = density_img[lobe==8].mean()
    density_l1 = density_img[lobe==16].mean()
    density_l2 = density_img[lobe==32].mean()
    density_l3 = density_img[lobe==64].mean()
    density_l4 = density_img[lobe==128].mean()
    density_t = density_img[lobe!=0].mean()

    density_stat = pd.DataFrame({'Lobes':['Lobe0','Lobe1','Lobe2','Lobe3','Lobe4','total'],
                'meanDensity':np.float16([density_l0,density_l1,density_l2,density_l3,density_l4,density_t]),
                    })
    return density_stat

def main(args):
    ProjSubj_path = args.ProjSubj
    save_path = args.save_path
    lobe_name = args.mask_name
    df = pd.read_csv(ProjSubj_path,sep='\t')
    df_QCT = df.iloc[:,:2] # First two cols are Proj & Subj
    pbar = tqdm(enumerate(df['ImgDir']),total=len(df))
    for i, subj_path in pbar:
        if not os.path.exists(subj_path):
            continue
        img_path = os.path.join(subj_path,'zunu_vida-ct.img')
        lobe_path = os.path.join(subj_path,lobe_name)
        if not os.path.exists(img_path):
            DCMtoVidaCT(subj_path)

        img, hdr = load(img_path)
        lobe, _ = load(lobe_path)
        voxel_spacing = hdr.get_voxel_spacing()

        HAA_stat = get_HAA(img,lobe)
        density_stat = get_CT_density(img,lobe)
        tissFrac_stat = get_tiss_frac(img,lobe)
        emph_stat = get_emph(img,lobe)
        totalVol_stat = get_total_volume(lobe, voxel_spacing)
        tissVol_stat = get_tiss_volume(tissFrac_stat,totalVol_stat)

        df_QCT.loc[i,f"HU_All"] = density_stat.meanDensity[5]
        df_QCT.loc[i,f"HU_LUL"] = density_stat.meanDensity[0]
        df_QCT.loc[i,f"HU_LLL"] = density_stat.meanDensity[1]
        df_QCT.loc[i,f"HU_RUL"] = density_stat.meanDensity[2]
        df_QCT.loc[i,f"HU_RML"] = density_stat.meanDensity[3]
        df_QCT.loc[i,f"HU_RLL"] = density_stat.meanDensity[4]

        df_QCT.loc[i,f"TF_All"] = tissFrac_stat.meanTF[5]
        df_QCT.loc[i,f"TF_LUL"] = tissFrac_stat.meanTF[0]
        df_QCT.loc[i,f"TF_LLL"] = tissFrac_stat.meanTF[1]
        df_QCT.loc[i,f"TF_RUL"] = tissFrac_stat.meanTF[2]
        df_QCT.loc[i,f"TF_RML"] = tissFrac_stat.meanTF[3]
        df_QCT.loc[i,f"TF_RLL"] = tissFrac_stat.meanTF[4]

        # Unit: [L]
        df_QCT.loc[i,f"totalVol_All"] = totalVol_stat.totalVol[5]/10**6
        df_QCT.loc[i,f"totalVol_LUL"] = totalVol_stat.totalVol[0]/10**6
        df_QCT.loc[i,f"totalVol_LLL"] = totalVol_stat.totalVol[1]/10**6
        df_QCT.loc[i,f"totalVol_RUL"] = totalVol_stat.totalVol[2]/10**6
        df_QCT.loc[i,f"totalVol_RML"] = totalVol_stat.totalVol[3]/10**6
        df_QCT.loc[i,f"totalVol_RLL"] = totalVol_stat.totalVol[4]/10**6

        # Unit: [L]
        df_QCT.loc[i,f"tissVol_All"] = tissVol_stat.tissVol[5]/10**6
        df_QCT.loc[i,f"tissVol_LUL"] = tissVol_stat.tissVol[0]/10**6
        df_QCT.loc[i,f"tissVol_LLL"] = tissVol_stat.tissVol[1]/10**6
        df_QCT.loc[i,f"tissVol_RUL"] = tissVol_stat.tissVol[2]/10**6
        df_QCT.loc[i,f"tissVol_RML"] = tissVol_stat.tissVol[3]/10**6
        df_QCT.loc[i,f"tissVol_RLL"] = tissVol_stat.tissVol[4]/10**6

        df_QCT.loc[i,f"HAA_All"] = HAA_stat.HAAratio[5]
        df_QCT.loc[i,f"HAA_LUL"] = HAA_stat.HAAratio[0]
        df_QCT.loc[i,f"HAA_LLL"] = HAA_stat.HAAratio[1]
        df_QCT.loc[i,f"HAA_RUL"] = HAA_stat.HAAratio[2]
        df_QCT.loc[i,f"HAA_RML"] = HAA_stat.HAAratio[3]
        df_QCT.loc[i,f"HAA_RLL"] = HAA_stat.HAAratio[4]

        df_QCT.loc[i,f"Emph_All"] = emph_stat.Emphysratio[5]
        df_QCT.loc[i,f"Emph_LUL"] = emph_stat.Emphysratio[0]
        df_QCT.loc[i,f"Emph_LLL"] = emph_stat.Emphysratio[1]
        df_QCT.loc[i,f"Emph_RUL"] = emph_stat.Emphysratio[2]
        df_QCT.loc[i,f"Emph_RML"] = emph_stat.Emphysratio[3]
        df_QCT.loc[i,f"Emph_RLL"] = emph_stat.Emphysratio[4]
    df_QCT.to_csv(save_path,index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)