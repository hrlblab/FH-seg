import argparse
import os, sys
import pandas as pd

sys.path.append("/Data/DoDNet/")

import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from matplotlib import cm

import skimage

import os.path as osp
# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
from MOTSDataset_2D_Patch_supervise_csv import MOTSValDataSet as MOTSValDataSet_joint

from unet2D_ns import UNet2D as UNet2D_ns
from unet2D_Dodnet_scale import UNet2D as UNet2D_scale
#from unet2D_Dodnet_scale_v2 import UNet2D as UNet2D_scale
import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool
from unet import UNet


def one_hot_2D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")

    #parser.add_argument("--valset_dir", type=str, default='KI_data_testingset_demo/data_list.csv')
    #parser.add_argument("--valset_dir", type=str, default='./data/HC_data_patch/ddd/data_list.csv')
    parser.add_argument("--valset_dir", type=str, default='./data/HC_data_patch/test/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907/')
    #parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907/MOTS_DynConv_fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907_e74.pth')
    #parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white/')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/final/Multi-class/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e106.pth')
    ##################################################################################################################################
    #parser.add_argument("--reload_path", type=str,
    #                    default='snapshots_2D/final/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_e50.pth')

    ###################################################################################################################################
    #parser.add_argument("--best_epoch", type=int, default=74)
    #parser.add_argument("--best_epoch", type=int, default=51)
    parser.add_argument("--best_epoch", type=int, default=106)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='./data/HC_data_patch/train/data_list.csv')
    #parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    #parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--val_list", type=str, default='./data/HC_data_patch/val/data_list.csv')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser

def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.

def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input1.ndim, connectivity)

    S = input1 - morphology.binary_erosion(input1, conn)
    Sprime = input2 - morphology.binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))


    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)


def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds0 = pred[1, ...].detach().cpu().numpy()
        labels0 = label[1, ...].detach().detach().cpu().numpy()

        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        # try:
        hausdorff, meansurfaceDistance = surfd(preds0, labels0)
        Val_HD += hausdorff
        Val_MSD += meansurfaceDistance

        Val_F1 += f1_score(preds1, labels1, average='macro')

        # except:
        #     Val_DICE += 1.
        #     Val_F1 += 1.
        #     Val_HD += 0.
        #     Val_MSD += 0.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)

from PIL import Image

def main():
    mask_folder = "./Dice_Eval/mask01.jpg"
    pred_folder = "./Dice_Eval/preds01.jpg"

    #images = task0_pool_image.query(batch_size)
    labels = Image.open(mask_folder)

    preds = Image.open(pred_folder)

    now_task = torch.tensor(0)

    filename = []


    # now_preds = preds[:,1,...] > preds[:,0,...]
    now_preds = torch.argmax(preds, 1) == now_task
    now_preds_onehot = one_hot_2D(now_preds.long())

    labels_onehot = one_hot_2D(labels.long())

    rmin, rmax, cmin, cmax = mask_to_box(images)

    for pi in range(len(images)):
        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
        #out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
        #img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
        #plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
        #           img)
        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                   labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                   prediction.detach().cpu().numpy(), cmap=cm.gray)

        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                        rmin, rmax, cmin, cmax)
        row = len(single_df_0)
        #single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

        #val_F1[0] += F1
        #val_Dice[0] += DICE
        #val_HD[0] += HD
        #val_MSD[0] += MSD
        #cnt[0] += 1


if __name__ == '__main__':
    main()
