from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ImageOps
import torchvision
import imgaug.augmenters as iaa


from torchvision import transforms

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        self.image_mask_aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(rotate=(-180, 180)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
        ])
        self.image_aug = iaa.Sequential([
            # iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            # iaa.GammaContrast((0.5, 2.0)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.1))
        ])

        self.image_aug_256 = iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)
        ])


    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))
        # pil_img = pil_img.resize((512, 512))

        # pil_img = torch.from_numpy(np.array(pil_img).transpose(([2,0,1]))) / 255
        #
        # transform_norm = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225],
        # )
        # #
        # pil_img = transform_norm(pil_img)

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # # Rotation
        # angle = np.random.rand(1) * 180 - 90
        # img = img.rotate(angle)
        # mask = mask.rotate(angle)
        #
        # flip_flag = np.random.rand(1)
        # if flip_flag >= 0.5:
        #     img = ImageOps.flip(img)
        #     mask = ImageOps.flip(mask)


        img = np.array(img)
        mask = np.array(mask)
        # # Center crop
        # center = int(img.shape[0] / 2)
        # size = img.shape[0]
        # min = center - size
        # max = center + size
        # img = img[min:max,min:max,:]
        # mask = mask[min:max,min:max,:]

        # rescale to (0,1)
        # img = img.transpose((2, 0, 1)) / 255
        # mask = mask.transpose((2, 0, 1)) / 255

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # img = self.image_aug_256(images = img)

        img = (img / 255).astype(np.float32)
        mask = (mask / 255).astype(np.float32)

        img, mask = self.image_mask_aug(images=img, heatmaps=mask)

        img = self.image_aug(images = img)

        #img = self.preprocess(img, self.scale)
        #mask = self.preprocess(mask, self.scale)

        # # Do data augmentation
        # image_transforms = transforms.Compose([
        #     transforms.RandomRotation(45),
        #     transforms.CenterCrop(size=700),
        #     transforms.RandomResizedCrop(512),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     # transforms.Normalize([0.485, 0.456, 0.406],
        #     #                     [0.229, 0.224, 0.225])  # Imagenet standards
        # ])
        #
        # img = image_transforms(img)
        # mask = image_transforms(mask)

        #mask = mask[0,:,:].unsqueeze(0)

        # return {
        #     'image': img,
        #     'mask': mask
        # }
        #Random crop
        # size = img.shape[1]
        # length = 512
        # length_max = size - length
        # length_x = int(np.random.rand(1) * length_max)
        # length_y = int(np.random.rand(1) * length_max)
        # img = img[0,length_x:length_x+length,length_y:length_y+length,:]
        # mask = mask[0,length_x:length_x+length,length_y:length_y+length,:]

        img = img[0,...]
        mask = mask[0,...]

        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))


        #Normalize the  image
        transform_norm = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)[0,:,:].unsqueeze(0)

        normalized_img = transform_norm(img)

        return {
            'image': normalized_img,
            'mask': mask
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
