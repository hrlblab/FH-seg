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

def bland_altman_plot(ax, data1, data2, color, dot,*args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    ax.scatter(mean, diff, *args, s=dot, c = color, edgecolor="black", **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    #ax.fill_between(diff, md - 1.96*sd, md + 1.96*sd, color='gray')

if __name__ == "__main__":

    size = 2
    color = '#7E6CE9'
    dot = 10

    glom = 1
    if glom:

        manual_list = ['/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-7_glom_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-8_glom_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-9_glom_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-10_glom_manual.csv']

        #
        # OmniSeg_list = [
        #     '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_scalecap/V11M25-279/6793-AF-7.glomeruli/ratio_6793-AF-7.glomeruli.csv',
        #     '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_scalecap/V11M25-279/6793-AF-8.glomeruli/ratio_6793-AF-8.glomeruli.csv',
        #     '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_scalecap/V11M25-279/6793-AF-9.glomeruli/ratio_6793-AF-9.glomeruli.csv',
        #     '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_scalecap/V11M25-279/6793-AF-10.glomeruli/ratio_6793-AF-10.glomeruli.csv']

        root = '/media/dengr/Seagate Backup Plus Drive/OmniSeg+TableResults/Patch_ratio_haichun_cap_256_20X'
        # root = '/Data2/HumanKidney/OmniSeg_testing'
        method = 'Patch_ratio_haichun_Med3D_scalecap'
        # OmniSeg_list = [
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-7.glomeruli/ratio_6793-AF-7.glomeruli.csv' % (method),
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-8.glomeruli/ratio_6793-AF-8.glomeruli.csv' % (method),
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-9.glomeruli/ratio_6793-AF-9.glomeruli.csv' % (method),
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-10.glomeruli/ratio_6793-AF-10.glomeruli.csv' % (method)]
        OmniSeg_list = [
            '%s/%s/V11M25-279/6793-AF-7.glomeruli/ratio_6793-AF-7.glomeruli.csv' % (root, method),
            '%s/%s/V11M25-279/6793-AF-8.glomeruli/ratio_6793-AF-8.glomeruli.csv' % (root, method),
            '%s/%s/V11M25-279/6793-AF-9.glomeruli/ratio_6793-AF-9.glomeruli.csv' % (root, method),
            '%s/%s/V11M25-279/6793-AF-10.glomeruli/ratio_6793-AF-10.glomeruli.csv' % (root, method)]

    else:
        manual_list = ['/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-7_tubule_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-8_tubule_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-9_tubule_manual.csv',
                       '/Data2/HumanKidney/Mouse_Atubular_Segmentation/Segmentation_ratio/6793-AF-10_tubule_manual.csv']

        OmniSeg_list = ['/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_cap256to256/V11M25-279/6793-AF-7.proximalTubules/ratio_6793-AF-7.proximalTubules.csv',
                             '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_cap256to256/V11M25-279/6793-AF-8.proximalTubules/ratio_6793-AF-8.proximalTubules.csv',
                             '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_cap256to256/V11M25-279/6793-AF-9.proximalTubules/ratio_6793-AF-9.proximalTubules.csv',
                             '/Data2/HumanKidney/OmniSeg_testing/Patch_ratio_haichun_Multi-head_scalecap/V11M25-279/6793-AF-10.proximalTubules/ratio_6793-AF-10.proximalTubules.csv']


    data1_sep = (pd.read_csv(f) for f in manual_list)
    data1 = pd.concat(data1_sep, ignore_index=True)

    data7_sep = (pd.read_csv(k) for k in OmniSeg_list)
    data7 = pd.concat(data7_sep, ignore_index=True)

    data1 = (data1['ratio'] / 100).tolist()
    data7 = data7['ratio'].tolist()

    #
    fig = plt.figure(figsize=(size, size))
    ax1 = fig.add_subplot(1,1,1)
    # ax2 = fig.add_subplot(1,2,2)
    # ax3 = fig.add_subplot(2,6,3)
    # ax4 = fig.add_subplot(2,6,4)
    # ax5 = fig.add_subplot(2,6,5)
    # ax6 = fig.add_subplot(2,6,6)
    # ax7 = fig.add_subplot(2,6,7)
    # ax8 = fig.add_subplot(2,6,8)
    # ax9 = fig.add_subplot(2,6,9)
    # ax10 = fig.add_subplot(2,6,10)
    # ax11 = fig.add_subplot(2,6,11)
    # ax12 = fig.add_subplot(2,6,12)

    x = np.linspace(0, 1)
    #y = x

    corr1, _ = spearmanr(data7, data1)


    # others
    corr2, _ = pearsonr(data7, data1)
    #ax3.set_title('Corr. = %02f' % corr)
    ax1.scatter(data1, data7, s=dot, c= color, edgecolor="black")
    #ax3.plot(x, y,  color='r')
    #ax3.set_xlabel('Manual')
    #ax3.set_ylabel('Our method')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])

    m, b = np.polyfit(data1, data7, 1)
    ax1.plot(x, m*x + b, color='r')
    fig.show()

    fig = plt.figure(figsize=(size, size))
    ax2 = fig.add_subplot(1,1,1)
    bland_altman_plot(ax2, data7, data1, color, dot)
    # ax.set_title('slice')
    # ax6.set_ylabel('difference')
    # ax6.set_xlabel('Our method')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-1, 1])
    fig.show()
    # ax6.set_yticks([])
    # ax6.set_xticks([])


    #ax3.set_yticks([])
    #ax3.set_xticks([])

    corr2, _ = spearmanr(data1, data3)
    #ax2.set_title('Corr. = %02f' % corr)
    #ax2.set_ylabel('MPA')
    ax2.scatter(data1, data2, edgecolor="black")
    #ax2.set_xlabel('Manual')
    #ax2.plot(x, y,  color='r')
    ax2.set_ylim([0, 800000])
    ax2.set_xlim([0, 800000])
    #ax2.set_yticks([])
    #ax2.set_xticks([])
    m, b = np.polyfit(data3, data1, 1)
    ax2.plot(x, m * x + b, color='r')


    corr, _ = pearsonr(data1, data2)
    print('Pearsons correlation: %.3f' % corr)


