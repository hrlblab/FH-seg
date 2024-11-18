import cv2 as cv2
import numpy as np
from PIL import Image
import os
import SimpleITK as sitk

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from skimage.transform import resize
import glob
import openslide
import matplotlib.pyplot as plt
import xmltodict
import pandas as pd
import tifffile
from PIL import Image, ImageDraw



def get_contour_detection(img, contour, cnt_Big, down_rate, shift):
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((4,1,2))

    cnt[0, 0, 0] = vertices[1]['@X']
    cnt[0, 0, 1] = vertices[0]['@Y']
    cnt[1, 0, 0] = vertices[1]['@X']
    cnt[1, 0, 1] = vertices[1]['@Y']
    cnt[2, 0, 0] = vertices[0]['@X']
    cnt[2, 0, 1] = vertices[1]['@Y']
    cnt[3, 0, 0] = vertices[0]['@X']
    cnt[3, 0, 1] = vertices[0]['@Y']

    cnt = cnt / down_rate

    # Big_x = (cnt_Big[0, 0, 0] + cnt_Big[1, 0, 0] + cnt_Big[2, 0, 0] + cnt_Big[3, 0, 0]) / 4
    # Big_y = (cnt_Big[0, 0, 1] + cnt_Big[1, 0, 1] + cnt_Big[2, 0, 1] + cnt_Big[3, 0, 1]) / 4
    x_min = cnt_Big[0, 0, 0]
    y_min = cnt_Big[0, 0, 1]

    cnt[..., 0] = cnt[..., 0] - x_min
    cnt[..., 1] = cnt[..., 1] - y_min

    cnt[cnt < 0] = 0

    glom = img[int(cnt[0, 0, 1]):int(cnt[3, 0, 1]), int(cnt[0, 0, 0]):int(cnt[1, 0, 0])]

    return glom, cnt

def get_annotation_contour(img, contour, down_rate, shift, lv, start_x, start_y, end_x, end_y):
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((4,1,2))

    now_id = int(contour['@Id'])

    cnt[0, 0, 0] = vertices[0]['@Y']
    cnt[0, 0, 1] = vertices[0]['@X']
    cnt[1, 0, 0] = vertices[1]['@Y']
    cnt[1, 0, 1] = vertices[1]['@X']
    cnt[2, 0, 0] = vertices[2]['@Y']
    cnt[2, 0, 1] = vertices[2]['@X']
    cnt[3, 0, 0] = vertices[3]['@Y']
    cnt[3, 0, 1] = vertices[3]['@X']

    cnt[0, 0, 0] = cnt[0, 0, 0] - shift
    cnt[1, 0, 0] = cnt[1, 0, 0] - shift
    cnt[2, 0, 0] = cnt[2, 0, 0] - shift
    cnt[3, 0, 0] = cnt[3, 0, 0] - shift

    cnt = cnt.astype(int)

    patch_size_x = int((cnt[2, 0, 0] - cnt[1, 0, 0]) / down_rate)
    patch_size_y = int((cnt[1, 0, 1] - cnt[0, 0, 1]) / down_rate)

    patch_start_x = cnt[0, 0, 0] + start_x  # remember the 90 degree rotation
    patch_start_y = end_y - cnt[1, 0, 1]
    patch = np.array(img.read_region((patch_start_x, patch_start_y), lv, (patch_size_x, patch_size_y)).convert('RGB'))

    # patch_resize = resize(patch, (int(patch.shape[0] / 2), int(patch.shape[1] / 2)))

    return patch, cnt, now_id

def scan_nonblack_end(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_end+offset_x, py_end), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end+offset_x, py_end), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_end, py_end+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end, py_end+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_end+(offset_x-1)
    y = py_end+(offset_y-1)
    return x,y

def get_none_zero(black_arr):

    nonzeros = black_arr.nonzero()
    starting_y = nonzeros[0].min()
    ending_y = nonzeros[0].max()
    starting_x = nonzeros[1].min()
    ending_x = nonzeros[1].max()

    return starting_x, starting_y, ending_x, ending_y

def get_nonblack_starting_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x + 1) * multiples
    py3 = (ending_y + 1) * multiples

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack(simg, px2, py2, px3, py3)

    return xx,yy

def get_nonblack_ending_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x - 1) * (multiples-1)
    py3 = (ending_y - 1) * (multiples-1)

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack_end(simg, px2, py2, px3, py3)

    return xx,yy

def scan_nonblack(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_start+offset_x, py_start), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start+offset_x, py_start), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_start, py_start+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start, py_start+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_start+offset_x-1
    y = py_start+offset_y-1
    return x,y

def area_calc(pred, radius):
    circle_area = 0
    pred_area = 0
    a = int(pred.shape[0] / 2)
    b = int(pred.shape[1] / 2)

    for xi in range(pred.shape[0]):
        for yi in range(pred.shape[1]):
            if (xi - a)**2 + (yi - b)**2 <= radius ** 2:
                circle_area += 1
                pred_area += pred[xi,yi]

    ratio = pred_area/circle_area
    return ratio

def scn_to_png(csv_list, Gene_list, output_folder):
    for ci in range(len(csv_list)):
        # now_csv = pd.read_csv(csv_list[ci], names=['Gene', 'class', 'row_ind', 'col_ind', 'imagerow', 'imagecol'])
        now_csv = pd.read_csv(Gene_list[ci])
        'get each boundary of slice and match the detection results'

        #patch_output_folder = os.path.join(output_folder, os.path.basename(csv_list[ci]).replace('.csv', '.proximalTubules'))
        patch_output_folder = os.path.join(output_folder, 'V11M25-279', os.path.basename(csv_list[ci]).replace('.csv', ''))

        now_csv['AI_pt'] = pd.NaT
        now_csv['AI_capsule'] = pd.NaT

        if not os.path.exists(patch_output_folder):
            os.makedirs(patch_output_folder)

        # AI_ratio = pd.DataFrame(columns = ['index', 'ratio'])

        for ki in range(len(now_csv)):
        #for ki in range(10):
            print(ki)
            # ind = ki + 1# "%04d"
            Gene = now_csv.iloc[ki][now_csv.keys()[0]]

            now_seg_name = glob.glob(os.path.join(patch_output_folder, '*%s*_preds.png' %(Gene)))[0]

            # x = int(now_csv.iloc[ki]['imagerow']) * 2           # cv2.circle has shifted the coords
            # y = int(now_csv.iloc[ki]['imagecol']) * 2
            #
            # now_name = '%04d_%s_%s_%s.png' % (ind, Gene, x, y)
            # now_seg_name = '%04d_%s_%s_%s.png_preds.png' % (ind, Gene, x, y)

            # now_seg = plt.imread(os.path.join(patch_output_folder, now_seg_name))[:,:,:3]
            now_seg = plt.imread(now_seg_name)[:,:,:3]

            size = 150 # (1/4 micron per pixel)
            fiducial_radius = int(105 / 2)
            # spot_radius = int(55 * 2 / 2)

            spot_radius_40X = int(55)

            ratio = area_calc(now_seg[:,:,0], spot_radius_40X)

            # row = len(AI_ratio)
            # AI_ratio.loc[row] = [ind, ratio]

            if ci < 4:
                now_csv.at[ki, 'AI_pt'] = ratio
            else:
                now_csv.at[ki, 'AI_capsule'] = ratio

            # Blue color in BGR
            color1 = (1., 0, 0)
            color2 = (0, 1., 0)

            # Line thickness of 2 px
            thickness = 5
            now_seg_name_circle = now_seg_name.replace('_preds.png', '_preds_circle.png')
            now_seg = cv2.circle(now_seg.copy(), (int(now_seg.shape[0] / 2), int(now_seg.shape[1] / 2)), spot_radius_40X + 5, color2, thickness)
            plt.imsave(os.path.join(patch_output_folder, now_seg_name_circle), now_seg)

        patch_output_ratio_folder = patch_output_folder
        now_csv.to_csv(os.path.join(patch_output_folder,'AI_ratio_Gene_%s.csv' % (os.path.basename(patch_output_ratio_folder))), index = False)

if __name__ == "__main__":

    lv = 1
    data_dir = '/Data2/HumanKidney/Mouse_Atubular/HE_scn'
    svs_folder = '/Data2/HumanKidney/Mouse_Atubular/HE_scn'
    output_dir = '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_OmniSeg_scale_#7_pt1024to512'

    csv_list = ['/Data2/HumanKidney/Omni-Seg_revision/MouseGeneCoords/6793-AF-7.proximalTubules.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/MouseGeneCoords/6793-AF-8.proximalTubules.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/MouseGeneCoords/6793-AF-9.proximalTubules.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/MouseGeneCoords/6793-AF-10.proximalTubules.csv',]


    Gene_list = ['/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_Gene/6793-AF-7.SPOTlightResults.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_Gene/6793-AF-8.SPOTlightResults.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_Gene/6793-AF-9.SPOTlightResults.csv',
                '/Data2/HumanKidney/Omni-Seg_revision/Patch_ratio_all_Gene/6793-AF-10.SPOTlightResults.csv',]




    sections = glob.glob(os.path.join(data_dir, '*.scn'))

    for si in range(len(sections)):
        now_section = sections[si]
        section_name = os.path.basename(sections[si]).replace('.scn', '')
        # now_circlenet_xml = os.path.join(circlenet_folder, section_name, '%s.xml' % (section_name))
        now_annotation_xml = sections[si].replace('.scn', '.xml')
        output_folder = now_section.replace(data_dir, output_dir).replace('.scn','')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        scn_to_png(csv_list, Gene_list, output_dir)
