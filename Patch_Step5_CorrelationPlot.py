from random import randint

import matplotlib
import pandas as pd
from shapely.geometry import Polygon, MultiPoint

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2 as cv2
import os
import math

import glob

import argparse
import numpy as np
#import torchvision
import matplotlib.cm as cm
import SimpleITK as sitk
import nibabel as nib

from math import acos
from math import sqrt
from math import pi
import cv2

from numpy import mean
from numpy import std
from numpy import median
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
from numpy import cov

# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(ax, data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    ax.scatter(mean, diff, *args, edgecolor="black", **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    #ax.fill_between(diff, md - 1.96*sd, md + 1.96*sd, color='gray')

if __name__ == "__main__":

    glom = 1
    if glom:
        manual_list = ['/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-7_glom_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-8_glom_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-9_glom_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-10_glom_manual.csv']


        OmniSeg_list = [
            '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_cap256to256/V11M25-279/6793-AF-7.glomeruli/ratio_6793-AF-7.glomeruli.csv',
            '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_cap256to256/V11M25-279/6793-AF-8.glomeruli/ratio_6793-AF-8.glomeruli.csv',
            '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_cap256to256/V11M25-279/6793-AF-9.glomeruli/ratio_6793-AF-9.glomeruli.csv',
            '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_cap256to256/V11M25-279/6793-AF-10.glomeruli/ratio_6793-AF-10.glomeruli.csv']


    else:

        manual_list = ['/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-7_tubule_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-8_tubule_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-9_tubule_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-10_tubule_manual.csv']

        OmniSeg_list = ['/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_haichun_PIPO_pt1024to512/V11M25-279/6793-AF-7.proximalTubules.proximalTubules/ratio_6793-AF-7.proximalTubules.csv',
                             '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_haichun_PIPO_pt1024to512/V11M25-279/6793-AF-8.proximalTubules.proximalTubules/ratio_6793-AF-8.proximalTubules.csv',
                             '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_haichun_PIPO_pt1024to512/V11M25-279/6793-AF-9.proximalTubules.proximalTubules/ratio_6793-AF-9.proximalTubules.csv',
                             '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_haichun_PIPO_pt1024to512/V11M25-279/6793-AF-10.proximalTubules.proximalTubules/ratio_6793-AF-10.proximalTubules.csv']


    data1_sep = (pd.read_csv(f) for f in manual_list)
    data1 = pd.concat(data1_sep, ignore_index=True)

    data7_sep = (pd.read_csv(k) for k in OmniSeg_list)
    data7 = pd.concat(data7_sep, ignore_index=True)

    data1 = (data1['ratio'] / 100).tolist()
    # data2 = data2['ratio'].tolist()
    # data3 = data3['ratio'].tolist()
    data7 = data7['ratio'].tolist()

    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data7: mean=%.3f stdv=%.3f' % (mean(data7), std(data7)))


    corr1, _ = spearmanr(data7, data1)


    # others
    corr2, _ = pearsonr(data7, data1)


    #
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2,6,1)
    ax2 = fig.add_subplot(2,6,2)
    ax3 = fig.add_subplot(2,6,3)
    ax4 = fig.add_subplot(2,6,4)
    ax5 = fig.add_subplot(2,6,5)
    ax6 = fig.add_subplot(2,6,6)
    ax7 = fig.add_subplot(2,6,7)
    ax8 = fig.add_subplot(2,6,8)
    ax9 = fig.add_subplot(2,6,9)
    ax10 = fig.add_subplot(2,6,10)
    ax11 = fig.add_subplot(2,6,11)
    ax12 = fig.add_subplot(2,6,12)

    x = np.linspace(0, 1)
    #y = x

    # corr1, _ = spearmanr(data2, data1)
    # #ax3.set_title('Corr. = %02f' % corr)
    # ax3.scatter(data2, data1,edgecolor="black")
    # #ax3.plot(x, y,  color='r')
    # #ax3.set_xlabel('Manual')
    # #ax3.set_ylabel('Our method')
    # ax3.set_ylim([0, 1])
    # ax3.set_xlim([0, 1])
    #
    # m, b = np.polyfit(data2, data1, 1)
    # ax3.plot(x, m*x + b, color='r')
    #
    # bland_altman_plot(ax6, data2, data1)
    # # ax.set_title('slice')
    # # ax6.set_ylabel('difference')
    # # ax6.set_xlabel('Our method')
    # ax6.set_xlim([0, 1])
    # ax6.set_ylim([-1, 1])
    # # ax6.set_yticks([])
    # # ax6.set_xticks([])
    # # ax.show()
    #
    # #ax3.set_yticks([])
    # #ax3.set_xticks([])
    #
    # corr2, _ = spearmanr(data1, data3)
    # #ax2.set_title('Corr. = %02f' % corr)
    # #ax2.set_ylabel('MPA')
    # ax2.scatter(data3, data1,edgecolor="black")
    # #ax2.set_xlabel('Manual')
    # #ax2.plot(x, y,  color='r')
    # ax2.set_ylim([0, 800000])
    # ax2.set_xlim([0, 800000])
    # #ax2.set_yticks([])
    # #ax2.set_xticks([])
    # m, b = np.polyfit(data3, data1, 1)
    # ax2.plot(x, m * x + b, color='r')
    #
    #
    # corr, _ = pearsonr(data1, data2)
    # print('Pearsons correlation: %.3f' % corr)
    #
    #
