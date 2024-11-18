import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math

# from batchgenerators.transforms import Compose
# from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
# from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
#     BrightnessTransform, ContrastAugmentationTransform
# from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
# from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import glob
from torch.utils.data import DataLoader, random_split
import scipy

class MOTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        task_list = []
        scale_list = []
        image_path_list = []
        label_path_list = []
        tasks = glob.glob(os.path.join(self.root,'*'))
        for ki in range(len(tasks)):
            tasks_id = os.path.basename(tasks[ki]).split('_')[0]
            scale_id = os.path.basename(tasks[ki]).split('_')[1]
            stain_folders = glob.glob(os.path.join(tasks[ki],'*'))

            for si in range(len(stain_folders)):
                images = glob.glob(os.path.join(stain_folders[si],'*'))
                for ri in range(len(images)):
                    if 'mask' in images[ri]:
                        continue
                    else:
                        image_root = images[ri]
                        _, ext = os.path.splitext(images[ri])
                        mask_root = glob.glob(os.path.join(stain_folders[si],os.path.basename(image_root).replace(ext,'_mask*')))[0]
                        task_list.append(int(tasks_id))
                        scale_list.append(int(scale_id))
                        image_path_list.append(image_root)
                        label_path_list.append(mask_root)

        self.task_cnt = np.zeros(len(tasks))
        # self.task_cnt = np.zeros(1)

        for ti in range(len(tasks)):
            self.task_cnt[ti] = task_list.count(ti)

        #self.task_cnt = np.array([335., 337., 377., 402., 961., 116.])

        self.task_cnt = self.task_cnt / self.task_cnt.max()
        #self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]

#        if not max_iters == None:
#            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        print("Start preprocessing....")
        for i in range(len(image_path_list)):
            #print(image_path_list[i] + ', ' + str(task_list[i]) + ', ' + str(scale_list[i]))
            image_path = image_path_list[i]
            label_path = label_path_list[i]
            task_id = task_list[i]
            # if (task_id == 0) or (task_id == 2) or (task_id == 3):
            #     continue
            scale_id = scale_list[i]
            # if task_id != 1:
            #     name = osp.splitext(osp.basename(label_path))[0]
            # else:
            name = osp.basename(label_path)
            img_file = image_path
            label_file = label_path
            #label = plt.imread(label_file)
            label = np.ones((512,512))
            #label = resize(label, (512,512), anti_aliasing=True)
            #if task_id == 1:
            #    label = label.transpose((1, 2, 0))
            boud_h, boud_w = np.where(label >= 1)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name,
                "task_id": task_id,
                "scale_id": scale_id,
                "bbx": [boud_h, boud_w]
            })
        print('{} images are loaded!'.format(len(image_path_list)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT, task_id):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, scaler, bbx):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        # boud_h, boud_w, boud_d = np.where(label >= 1)
        boud_h, boud_w, boud_d = bbx
        margin = 32  # pixels

        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
            bbx_h_mint = bbx_h_min - math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
            if bbx_h_mint < 0:
                bbx_h_maxt -= bbx_h_mint
                bbx_h_mint = 0
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
            bbx_w_mint = bbx_w_min - math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
            if bbx_w_mint < 0:
                bbx_w_maxt -= bbx_w_mint
                bbx_w_mint = 0
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
            bbx_d_mint = bbx_d_min - math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
            if bbx_d_mint < 0:
                bbx_d_maxt -= bbx_d_mint
                bbx_d_mint = 0
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, np.max([bbx_d_max - scale_d, bbx_d_min]))
            h0 = random.randint(bbx_h_min, np.max([bbx_h_max - scale_h, bbx_h_min]))
            w0 = random.randint(bbx_w_min, np.max([bbx_w_max - scale_w, bbx_w_min]))
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        print(datafiles["image"])

        patch_num = 1

        # read png file
        image = plt.imread(datafiles["image"])
        label = plt.imread(datafiles["label"])

        weight = scipy.ndimage.morphology.binary_dilation(label.astype(np.uint8) == 1, iterations=2) & ~ label.astype(np.uint8)
        weight = weight.astype(np.float32)

        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]
        # if (task_id == 0) or (task_id == 2) or (task_id == 3):
        #     return image.copy(), label.copy(), name, task_id, scale_id

        # Multi-scale input

        if scale_id == 2:
            for si in range(int(1)):
                boud_h, boud_w = np.where(label >= 1)
                n = np.random.rand(1)
                if n > 0:
                    # method 2
                    length = 256
                    length_max = image.shape[0] - length
                    length_x = int(np.random.rand(1) * length_max)
                    length_y = int(np.random.rand(1) * length_max)
                    # while label[length_x:length_x+length,length_y:length_y+length].sum()  == 0:
                    cnt = 0
                    if image[length_x:length_x+length,length_y:length_y+length,:].mean() > 1.:
                        th = 0.8 * 255
                    else:
                        th = 0.8

                    while image[length_x:length_x + length, length_y:length_y + length, :].mean() <= th:
                        cnt = cnt + 1
                        length_max = image.shape[0] - length
                        length_x = int(np.random.rand(1) * length_max)
                        length_y = int(np.random.rand(1) * length_max)
                        if cnt == 50:
                            return np.ones((256, 256, 3)), np.zeros((256, 256)), name, task_id, scale_id

                image_c = image[length_x:length_x+length,length_y:length_y+length,:]
                label_c = label[length_x:length_x+length,length_y:length_y+length]
                weight_c = weight[length_x:length_x+length,length_y:length_y+length]
                label_c = np.stack([label_c,label_c,label_c],2)
                label_c[label_c > 0.] = 1.

                assert (image_c.shape == (256, 256, 3))

                old_folder = os.path.dirname(datafiles["image"])
                new_folder = os.path.dirname(datafiles["image"]).replace(self.root, self.root + '_patch').replace(
                        '_2_', '_3_')

                if not os.path.exists(new_folder):
                    # os.makedirs(new_folder)

                    try:
                        os.makedirs(new_folder)
                    except:
                        print('folder')

                new_root_image = os.path.join(new_folder, os.path.basename(datafiles["image"]).replace('.png',
                                                                                                       '_%s.png' % str(
                                                                                                           si)).replace(
                    '.tif', '_%s.png' % str(si)))
                new_root_label = os.path.join(new_folder, os.path.basename(datafiles["label"]).replace('_mask',
                                                                                                       '_%s_mask' % str(
                                                                                                           si)))
                plt.imsave(new_root_image, image_c)
                plt.imsave(new_root_label, label_c)


        elif scale_id == 1:
            image = resize(image, (3000/4, 3000/4, 3), anti_aliasing=False)
            label = resize(label, (3000/4, 3000/4), anti_aliasing=False)

            for si in range(int(1)):

                boud_h, boud_w = np.where(label >= 1)
                n = np.random.rand(1)
                if n > 0:

                    # method 2
                    length = 256
                    length_max = image.shape[0] - length
                    length_x = int(np.random.rand(1) * length_max)
                    length_y = int(np.random.rand(1) * length_max)
                    # while label[length_x:length_x+length,length_y:length_y+length].sum()  == 0:
                    cnt = 0
                    if image[length_x:length_x+length,length_y:length_y+length,:].mean() > 1.:
                        th = 0.8 * 255
                    else:
                        th = 0.8

                    while image[length_x:length_x + length, length_y:length_y + length, :].mean() <= th:
                        cnt = cnt + 1
                        length_max = image.shape[0] - length
                        length_x = int(np.random.rand(1) * length_max)
                        length_y = int(np.random.rand(1) * length_max)
                        if cnt == 50:
                            return np.ones((256, 256, 3)), np.zeros((256, 256)), name, task_id, scale_id

                image_c = image[length_x:length_x+length,length_y:length_y+length,:]
                label_c = label[length_x:length_x+length,length_y:length_y+length]
                weight_c = weight[length_x:length_x + length, length_y:length_y + length]
                label_c = np.stack([label_c,label_c,label_c],2)
                label_c[label_c > 0.] = 1.

                assert (image_c.shape == (256, 256, 3))

                # do scale augmentation
                n = np.random.rand(1)
                if 0: #n > 0.50:
                    '''rescale 40X'''
                    length = 64
                    length_max = image_c.shape[0] - length
                    length_x = int(np.random.rand(1) * length_max)
                    length_y = int(np.random.rand(1) * length_max)
                    cnt = 0
                    while label_c[length_x:length_x + length, length_y:length_y + length].sum() == 0 and cnt <= 100:
                        cnt += 1
                        length_max = image_c.shape[0] - length
                        length_x = int(np.random.rand(1) * length_max)
                        length_y = int(np.random.rand(1) * length_max)

                    mapping_mask = np.zeros((label_c.shape))
                    mapping_mask[length_x:length_x+length,length_y:length_y+length,:] = 1.
                    image_cc = image_c[length_x:length_x+length,length_y:length_y+length,:]

                    label_cc = label_c[length_x:length_x+length,length_y:length_y+length,:]
                    # label_cc = np.stack([label_cc,label_cc,label_cc],2)
                    # label_cc[label_cc > 0.] = 1.

                    assert (image_cc.shape == (64, 64, 3))

                    image_cc = resize(image_cc, (256, 256, 3),anti_aliasing=False)
                    label_cc = resize(label_cc, (256, 256, 3),anti_aliasing=False)

                    label_cc[label_cc > 0.5] = 1.
                    label_cc[label_cc < 0.5] = 0.

                    old_folder = os.path.dirname(datafiles["image"])
                    new_folder = os.path.dirname(datafiles["image"]).replace(self.root, self.root + '_patch').replace(
                        '_1_', '_2_')

                    if not os.path.exists(new_folder):
                        # os.makedirs(new_folder)

                        try:
                            os.makedirs(new_folder)
                        except:
                            print('folder')

                    new_root_image = os.path.join(new_folder, os.path.basename(datafiles["image"]).replace('.png',
                                                                                                           '_%s.png' % str(
                                                                                                               si)).replace(
                        '.tif', '_%s.png' % str(si)))
                    new_root_label = os.path.join(new_folder, os.path.basename(datafiles["label"]).replace('_mask',
                                                                                                           '_%s_mask' % str(
                                                                                                               si)))

                    plt.imsave(new_root_image, image_cc)
                    plt.imsave(new_root_label, label_cc)

                else:
                    old_folder = os.path.dirname(datafiles["image"])
                    new_folder = os.path.dirname(datafiles["image"]).replace(self.root, self.root + '_patch')

                    if not os.path.exists(new_folder):
                        # os.makedirs(new_folder)

                        try:
                            os.makedirs(new_folder)
                        except:
                            print('folder')

                    new_root_image = os.path.join(new_folder, os.path.basename(datafiles["image"]).replace('.png','_%s.png' % str(si)).replace('.tif','_%s.png' % str(si)))
                    new_root_label = os.path.join(new_folder, os.path.basename(datafiles["label"]).replace('_mask','_%s_mask' % str(si)))


                    plt.imsave(new_root_image, image_c)
                    plt.imsave(new_root_label, label_c)

        else:
            image = resize(image,(3000/8, 3000/8, 3), anti_aliasing=False)
            label = resize(label,(3000/8, 3000/8), anti_aliasing=False)

            for si in range(int(1)):

                boud_h, boud_w = np.where(label >= 1)
                n = np.random.rand(1)
                if n > 0:
                    # method 2
                    length = 256
                    length_max = image.shape[0] - length
                    length_x = int(np.random.rand(1) * length_max)
                    length_y = int(np.random.rand(1) * length_max)
                    # while label[length_x:length_x+length,length_y:length_y+length].sum()  == 0:
                    cnt = 0
                    if image[length_x:length_x+length,length_y:length_y+length,:].mean() > 1.:
                        th = 0.8 * 255
                    else:
                        th = 0.8

                    while image[length_x:length_x + length, length_y:length_y + length, :].mean() <= th:
                        cnt = cnt + 1
                        length_max = image.shape[0] - length
                        length_x = int(np.random.rand(1) * length_max)
                        length_y = int(np.random.rand(1) * length_max)
                        if cnt == 50:
                            return np.ones((256, 256, 3)), np.zeros((256, 256)), name, task_id, scale_id

                image_c = image[length_x:length_x+length,length_y:length_y+length,:]
                label_c = label[length_x:length_x+length,length_y:length_y+length]
                label_c = np.stack([label_c,label_c,label_c],2)
                label_c[label_c > 0.] = 1.

                assert (image_c.shape == (256, 256, 3))

                # do scale augmentation
                n = np.random.rand(1)
                if 0: #n > 0.66:
                    '''rescale 40X'''
                    length = 32
                    length_max = image_c.shape[0] - length
                    length_x = int(np.random.rand(1) * length_max)
                    length_y = int(np.random.rand(1) * length_max)
                    cnt = 0
                    while label_c[length_x:length_x + length, length_y:length_y + length].sum() == 0 and cnt <= 100:
                        cnt += 1
                        length_max = image_c.shape[0] - length
                        length_x = int(np.random.rand(1) * length_max)
                        length_y = int(np.random.rand(1) * length_max)

                    mapping_mask = np.zeros((label_c.shape))
                    mapping_mask[length_x:length_x + length, length_y:length_y + length, :] = 1.
                    image_cc = image_c[length_x:length_x + length, length_y:length_y + length, :]

                    label_cc = label_c[length_x:length_x + length, length_y:length_y + length, :]
                    # label_cc = np.stack([label_cc, label_cc, label_cc], 2)
                    # label_cc[label_cc > 0.] = 1.

                    assert (image_cc.shape == (32, 32, 3))

                    image_cc = resize(image_cc, (256, 256, 3),anti_aliasing=False)
                    label_cc = resize(label_cc, (256, 256, 3),anti_aliasing=False)

                    label_cc[label_cc > 0.5] = 1.
                    label_cc[label_cc < 0.5] = 0.

                    old_folder = os.path.dirname(datafiles["image"])
                    new_folder = os.path.dirname(datafiles["image"]).replace(self.root, self.root + '_patch').replace(
                        '_0_', '_2_')

                    if not os.path.exists(new_folder):
                        # os.makedirs(new_folder)

                        try:
                            os.makedirs(new_folder)
                        except:
                            print('folder')

                    new_root_image = os.path.join(new_folder, os.path.basename(datafiles["image"]).replace('.png',
                                                                                                           '_%s.png' % str(
                                                                                                               si)).replace(
                        '.tif', '_%s.png' % str(si)))
                    new_root_label = os.path.join(new_folder, os.path.basename(datafiles["label"]).replace('_mask',
                                                                                                           '_%s_mask' % str(
                                                                                                               si)))

                    plt.imsave(new_root_image, image_cc)
                    plt.imsave(new_root_label, label_cc)

                elif 0: # n > 0.33 and n < 0.66:
                    '''rescale 10X'''
                    length = 128
                    length_max = image_c.shape[0] - length
                    length_x = int(np.random.rand(1) * length_max)
                    length_y = int(np.random.rand(1) * length_max)
                    cnt = 0
                    while label_c[length_x:length_x + length, length_y:length_y + length].sum() == 0 and cnt <= 100:
                        cnt += 1
                        length_max = image_c.shape[0] - length
                        length_x = int(np.random.rand(1) * length_max)
                        length_y = int(np.random.rand(1) * length_max)

                    mapping_mask = np.zeros((label_c.shape))
                    mapping_mask[length_x:length_x+length,length_y:length_y+length,:] = 1.
                    image_cc = image_c[length_x:length_x+length,length_y:length_y+length,:]

                    label_cc = label_c[length_x:length_x+length,length_y:length_y+length,:]
                    #label_cc = np.stack([label_cc,label_cc,label_cc],2)

                    assert (image_cc.shape == (128, 128, 3))

                    image_cc = resize(image_cc, (256, 256, 3),anti_aliasing=False)
                    label_cc = resize(label_cc, (256, 256, 3),anti_aliasing=False)
                    label_cc[label_cc > 0.5] = 1.
                    label_cc[label_cc < 0.5] = 0.

                    old_folder = os.path.dirname(datafiles["image"])
                    new_folder = os.path.dirname(datafiles["image"]).replace(self.root, self.root + '_patch').replace(
                        '_0_', '_1_')

                    if not os.path.exists(new_folder):
                        # os.makedirs(new_folder)

                        try:
                            os.makedirs(new_folder)
                        except:
                            print('folder')

                    new_root_image = os.path.join(new_folder, os.path.basename(datafiles["image"]).replace('.png',
                                                                                                           '_%s.png' % str(
                                                                                                               si)).replace(
                        '.tif', '_%s.png' % str(si)))
                    new_root_label = os.path.join(new_folder, os.path.basename(datafiles["label"]).replace('_mask',
                                                                                                           '_%s_mask' % str(
                                                                                                               si)))

                    plt.imsave(new_root_image, image_cc)
                    plt.imsave(new_root_label, label_cc)

                else:
                    old_folder = os.path.dirname(datafiles["image"])
                    new_folder = os.path.dirname(datafiles["image"]).replace(self.root, self.root + '_patch')

                    if not os.path.exists(new_folder):
                        # os.makedirs(new_folder)
                        try:
                            os.makedirs(new_folder)
                        except:
                            print('folder')

                    new_root_image = os.path.join(new_folder, os.path.basename(datafiles["image"]).replace('.png','_%s.png' % str(si)).replace('.tif','_%s.png' % str(si)))
                    new_root_label = os.path.join(new_folder, os.path.basename(datafiles["label"]).replace('_mask','_%s_mask' % str(si)))

                    plt.imsave(new_root_image, image_c)
                    plt.imsave(new_root_label, label_c)

        return image.copy(), label.copy(), name, task_id, scale_id

class MOTSValDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 256, 256), mean=(128, 128, 128), scale=False,
                 mirror=False, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror

        task_list = []
        scale_list = []
        image_path_list = []
        label_path_list = []
        tasks = glob.glob(os.path.join(self.root,'*'))
        for ki in range(len(tasks)):
            tasks_id = os.path.basename(tasks[ki]).split('_')[0]
            scale_id = os.path.basename(tasks[ki]).split('_')[1]
            stain_folders = glob.glob(os.path.join(tasks[ki],'*'))
            for si in range(len(stain_folders)):
                images = glob.glob(os.path.join(stain_folders[si],'*'))
                for ri in range(len(images)):
                    if 'mask' in images[ri]:
                        continue
                    else:
                        image_root = images[ri]
                        _, ext = os.path.splitext(images[ri])
                        mask_root = glob.glob(os.path.join(stain_folders[si],os.path.basename(image_root).replace(ext,'_mask*')))[0]
                        task_list.append(int(tasks_id))
                        scale_list.append(int(scale_id))
                        image_path_list.append(image_root)
                        label_path_list.append(mask_root)


        self.files = []

        for i in range(len(image_path_list)):
            print(image_path_list[i] + ', ' + label_path_list[i] )
            image_path = image_path_list[i]
            label_path = label_path_list[i]
            task_id = task_list[i]
            scale_id = scale_list[i]
            # if task_id != 1:
            #     name = osp.splitext(osp.basename(label_path))[0]
            # else:
            name = osp.basename(label_path)
            img_file = image_path
            label_file = label_path

            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name,
                "task_id": task_id,
                "scale_id": scale_id,
            })
        print('{} images are loaded!'.format(len(image_path_list)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT, task_id):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.
        
        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, scaler):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        boud_h, boud_w, boud_d = np.where(label >= 1)
        margin = 32  # pixels
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_mint = bbx_h_min - (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_mint = bbx_w_min - (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_mint = bbx_d_min - (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, bbx_d_max - scale_d)
            h0 = random.randint(bbx_h_min, bbx_h_max - scale_h)
            w0 = random.randint(bbx_w_min, bbx_w_max - scale_w)
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read png data
        image = plt.imread(datafiles["image"])
        label = plt.imread(datafiles["label"])
        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        #image = resize(image, (512, 512, 3), anti_aliasing=True)
        image = np.expand_dims(image, axis=3)

        #label = resize(label, (512, 512), anti_aliasing=True)
        label = np.expand_dims(label, axis=2)

        image = image.transpose((2, 0, 1, 3))  # Channel x Depth x H x W
        #label = label.transpose((2, 0, 1))  # Depth x H x W

        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return image.copy(), label.copy(), name, task_id, scale_id #labelNII.affine


def get_train_transform():
    tr_transforms = []

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
    tr_transforms.append(
        GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
                              p_per_sample=0.2, data_key="image"))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
    tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
    tr_transforms.append(
        SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
                                       order_upsample=3, p_per_sample=0.25,
                                       ignore_axes=None, data_key="image"))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
                                        p_per_sample=0.15, data_key="image"))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def my_collate(batch):
    image, label, name, task_id, scale_id = zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    task_id = np.stack(task_id, 0)
    scale_id = np.stack(scale_id, 0)
    data_dict = {'image': image, 'label': label, 'name': name, 'task_id': task_id, 'scale_id': scale_id}
    # tr_transforms = get_train_transform()
    # data_dict = tr_transforms(**data_dict)
    return data_dict

if __name__ == '__main__':

    trainset_dir = '/Data2/KI_data_train_scale_normal_onestep'
    train_list = '/Data2/KI_data_train_scale_normal_onestep'
    itrs_each_epoch = 250
    batch_size = 1
    input_size = (256,256)
    random_scale = False
    random_mirror = False

    save_img = '/Data2/KI_data_valid_patches_scale_aug'
    save_mask = '/Data2/KI_data_valid_patches_scale_aug'

    img_scale = 0.5

    trainloader = DataLoader(
        MOTSDataSet(trainset_dir, train_list, max_iters=itrs_each_epoch * batch_size,
                    crop_size=input_size, scale=random_scale, mirror=random_mirror),batch_size = 1, shuffle = False, num_workers =8)

    for iter, batch in enumerate(trainloader):
        print(iter)
        # imgs = torch.from_numpy(batch['image']).cuda()
        # lbls = torch.from_numpy(batch['label']).cuda()
        # volumeName = batch['name']
        # t_ids = torch.from_numpy(batch['task_id']).cuda()
        # s_ids = torch.from_numpy(batch['scale_id']).cuda()