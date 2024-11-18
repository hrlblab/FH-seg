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

def bland_altman_plot(ax, data1, data2,  color, dot, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    # ax.scatter(mean, diff, *args, s=5, edgecolor="black", **kwargs)
    ax.scatter(mean, diff, *args, s=dot, c = [color] * len(data1), edgecolor="black", **kwargs)
    ax.axhline(md,           color='gray', linestyle='--')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    #ax.fill_between(diff, md - 1.96*sd, md + 1.96*sd, color='gray')

if __name__ == "__main__":
    size = 2
    color = '#8CF6B3'
    dot = 10

    glom = 0
    if glom == 1:
        list = ['/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_No4/V11M25-279/6793-AF-7.glomeruli/AI_ratio_Gene_6793-AF-7.glomeruli.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_No4/V11M25-279/6793-AF-8.glomeruli/AI_ratio_Gene_6793-AF-8.glomeruli.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_No4/V11M25-279/6793-AF-9.glomeruli/AI_ratio_Gene_6793-AF-9.glomeruli.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_No4/V11M25-279/6793-AF-10.glomeruli/AI_ratio_Gene_6793-AF-10.glomeruli.csv']

    else:
        # root = '/media/dengr/Seagate Backup Plus Drive/OmniSeg+TableResults/Patch_ratio_hairchun_pt_512_10X'
        root = '/Data2/HumanKidney/OmniSeg_testing'
        method = 'Patch_ratio_haichun_residualunet_pt1024to512'
        # OmniSeg_list = [
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-7.glomeruli/ratio_6793-AF-7.glomeruli.csv' % (method),
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-8.glomeruli/ratio_6793-AF-8.glomeruli.csv' % (method),
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-9.glomeruli/ratio_6793-AF-9.glomeruli.csv' % (method),
        #     '/Data2/HumanKidney/OmniSeg_testing/%s/V11M25-279/6793-AF-10.glomeruli/ratio_6793-AF-10.glomeruli.csv' % (method)]
        OmniSeg_list = [
            '%s/%s/V11M25-279/6793-AF-7.proximalTubules/AI_ratio_Gene_6793-AF-7.proximalTubules.csv' % (root, method),
            '%s/%s/V11M25-279/6793-AF-8.proximalTubules/AI_ratio_Gene_6793-AF-8.proximalTubules.csv' % (root, method),
            '%s/%s/V11M25-279/6793-AF-9.proximalTubules/AI_ratio_Gene_6793-AF-9.proximalTubules.csv' % (root, method),
            '%s/%s/V11M25-279/6793-AF-10.proximalTubules/AI_ratio_Gene_6793-AF-10.proximalTubules.csv' % (root, method)]


    data_sep = (pd.read_csv(f) for f in OmniSeg_list)
    data = pd.concat(data_sep, ignore_index=True)

    if glom == 1:
        data1 = (data['topic_17'] + data['topic_10']).tolist()
        data2 = data['AI_capsule'].tolist()
    else:
        data1 = (data['topic_11'] + data['topic_9'] + data['topic_7']).tolist()
        data2 = data['AI_pt'].tolist()


    print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
    print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))


    corr1, _ = spearmanr(data2, data1)


    # others
    corr2, _ = pearsonr(data2, data1)

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

    corr1, _ = spearmanr(data2, data1)


    # others
    corr2, _ = pearsonr(data1, data2)
    #ax3.set_title('Corr. = %02f' % corr)
    ax1.scatter(data1, data2, s = dot, c = [color] * len(data1), edgecolor="black")
    #ax3.plot(x, y,  color='r')
    #ax3.set_xlabel('Manual')
    #ax3.set_ylabel('Our method')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])

    m, b = np.polyfit(data1, data2, 1)
    ax1.plot(x, m*x + b, color='r')
    fig.show()

    fig = plt.figure(figsize=(size, size))
    ax2 = fig.add_subplot(1,1,1)
    bland_altman_plot(ax2, data2, data1, color, dot)
    # ax.set_title('slice')
    # ax6.set_ylabel('difference')
    # ax6.set_xlabel('Our method')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-1, 1])
    # ax6.set_yticks([])
    # ax6.set_xticks([])
    fig.show()

    #ax3.set_yticks([])
    #ax3.set_xticks([])


    #
    # fig = plt.figure(figsize=(1, 2))
    # ax1 = fig.add_subplot(1,1,1)
    # ax2 = fig.add_subplot(1,1,2)
    #
    # x = np.linspace(0, 1)
    # #y = x
    #
    # corr2, _ = spearmanr(data2, data1)
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
    # bland_altman_plot(ax2, data2, data1)
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

    corr2, _ = spearmanr(data1, data3)
    #ax2.set_title('Corr. = %02f' % corr)
    #ax2.set_ylabel('MPA')
    ax2.scatter(data3, data1)
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


