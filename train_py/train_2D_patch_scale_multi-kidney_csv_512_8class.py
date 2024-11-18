import argparse
import os, sys
import pandas as pd

#sys.path.append("..")
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
import matplotlib.pyplot as plt

import torch.nn as nn

# from unet2D_Dodnet_scale import UNet2D as UNet2D_scale
# from unet2D_Dodnet_ns import UNet2D as UNet2D_ns
import imgaug.augmenters as iaa
import kornia

from torchvision import transforms

from PIL import Image, ImageOps
from unet import UNet

import os.path as osp

from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSDataSet as MOTSDataSet
from MOTSDataset_2D_Patch_supervise_csv_512 import MOTSValDataSet as MOTSValDataSet

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool

from model import Custom

def one_hot_3D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot

    # parser.add_argument("--valset_dir", type=str, defau


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")
    # parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_trainingset_patch/data_list.csv')
    parser.add_argument("--trainset_dir", type=str, default='./data/omniseg-sampled/train')

    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/data_list.csv')
    parser.add_argument("--valset_dir", type=str, default='./data/omniseg-sampled/val')


    parser.add_argument("--train_list", type=str, default='./data/omniseg-sampled/train/data_list.csv')
    parser.add_argument("--val_list", type=str, default='./data/omniseg-sampled/val/data_list.csv')
    parser.add_argument("--edge_weight", type=float, default=1.0)

    parser.add_argument("--scale", type=str2bool, default=False)
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/Multi-kidney_1019_8class_batch4/')
    parser.add_argument("--reload_path", type=str, default='')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=101)
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


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))

    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        try:
            rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
            cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]
        except:
            rmin[ki], rmax[ki] = 0, 255
            cmin[ki], cmax[ki] = 0, 255

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)

def get_scale_tensor(pred, rmin, rmax, cmin, cmax):
    if len(pred.shape) == 3:
        return pred[:,rmin:rmax,cmin:cmax].unsqueeze(0)
    else:
        return pred[rmin:rmax, cmin:cmax].unsqueeze(0)

def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        # pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        # label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        pred = preds[ki,:,rmin:rmax,cmin:cmax]
        label = labels[ki,:,rmin:rmax,cmin:cmax]

        # preds1 = sum(preds,[])
        # labels1 = sum(labels,[])
        #try:
        Val_DICE += dice_score(pred, label)
        # preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1, ...].flatten().detach().cpu().numpy()
        # labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1, ...].detach().flatten().detach().cpu().numpy()

        cnf_matrix = confusion_matrix(preds1, labels1)

        # FP = cnf_matrix[1,0]
        # FN = cnf_matrix[0,1]
        # TP = cnf_matrix[1,1]
        # TN = cnf_matrix[0,0]

        try:
            FP = cnf_matrix[1,0]
            FN = cnf_matrix[0,1]
            TP = cnf_matrix[1,1]
            TN = cnf_matrix[0,0]
        except:
            FP = np.array(1)
            FN = np.array(1)
            TP = np.array(1)
            TN = np.array(1)

        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)

        Val_TPR += TP / (TP + FN)
        Val_PPV += TP / (TP + FP)

        Val_F1 += f1_score(preds1, labels1, average='macro')

        # except:
        #
        #     Val_DICE += 1.
        #     Val_F1 += 1.
        #     Val_TPR += 1.
        #     Val_PPV += 1.

    return Val_F1/cnt, Val_DICE/cnt, Val_TPR/cnt, Val_PPV/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()


def get_loss(images, preds, labels, weight, loss_seg_DICE, loss_seg_CE):

    term_seg_Dice = 0
    term_seg_BCE = 0
    term_all = 0

    term_seg_Dice += loss_seg_DICE.forward(preds, labels, weight)
    term_seg_BCE += loss_seg_CE.forward(preds, labels, weight)
    term_all += (term_seg_Dice + term_seg_BCE)

    return term_seg_Dice, term_seg_BCE, term_all

def supervise_learning(images, labels, batch_size, scales, model, now_task, weight, loss_seg_DICE, loss_seg_CE):

    preds,_ = model(images, torch.ones(batch_size).cuda() * now_task, scales)
    labels = one_hot_3D(labels.long())

    term_seg_Dice, term_seg_BCE, term_all = get_loss(images, preds[:,now_task].unsqueeze(1), labels[:,1].unsqueeze(1), weight, loss_seg_DICE, loss_seg_CE)

    return term_seg_Dice, term_seg_BCE, term_all


def semi_learning(images, batch_size, scales, model, now_task, loss_KL, loss_MSE, data_aug, data_return):

    affine_images1 = data_aug(images)
    affine_images2 = data_aug(images)

    _, features_map1 = model(affine_images1.cuda(), torch.ones(batch_size).cuda() * now_task, scales)
    _, features_map2 = model(affine_images2.cuda(), torch.ones(batch_size).cuda() * now_task, scales)

    softmax = nn.Softmax()
    term_KL = loss_KL(softmax(features_map1), softmax(features_map2))

    term_MSE = loss_MSE(features_map1, features_map2)
    term_all = term_KL + term_MSE

    return term_KL, term_MSE, term_all


def  data_aug(images):

    'Tensor version data augmentation'

    color_transform = transforms.Compose([
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.GaussianBlur(kernel_size=5)
        ]
    )

    affine_transform = kornia.augmentation.AugmentationSequential(
        kornia.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3),
        kornia.augmentation.RandomAffine(p=0.75, degrees=(-180, 180), translate=(0.2, 0.2), shear=(-16, 16),
                                         scale=(0.75, 1.5)),
        kornia.augmentation.RandomHorizontalFlip(return_transform=True, p=0.5),
        kornia.augmentation.RandomVerticalFlip(return_transform=True, p=0.5),
        return_transform=True,
        same_on_batch=False,
    )

    weight = torch.ones((images.shape)).cuda()

    color_images = color_transform(images)
    affine_images, transforms_matrix = affine_transform(color_images)
    invert_matrix = kornia.geometry.transform.invert_affine_transform(transforms_matrix[:, :2, :])

    return_weight = kornia.geometry.transform.affine(kornia.geometry.transform.affine(weight, transforms_matrix[:, :2, :]), invert_matrix)
    # return_image = kornia.geometry.transform.affine(affine_images, invert_matrix)

    return affine_images.cuda(), invert_matrix.cuda(), return_weight


def  data_aug_noreturn(images):

    'Tensor version data augmentation'

    image_size = 256
    imagenet_norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    self_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0),
                                     interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                               p=0.1),
        #transforms.RandomApply([Solarization()], p=0.2),
        transforms.Normalize(*imagenet_norm)
    ])

    aug_images = self_transform(images)

    return aug_images.cuda()

def data_return(images, invert_matrix):

    return_images = kornia.geometry.transform.affine(images, invert_matrix)

    return return_images


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        model = Custom(input_ch = 3, output_ch = 8, modelDim=2)

        check_wo_gpu = 0

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        #optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...a
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not check_wo_gpu:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=1).to(device)                     #only for multi-head
            loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=1, ignore_index=255).to(device)
            loss_KL = nn.KLDivLoss().to(device)
            loss_MSE = nn.MSELoss().to(device)


        else:
            weights = [1., 1.]
            class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
            loss_seg_CE = loss.CELoss4MOTS(weight = weights, num_classes=args.num_classes, ignore_index=255)
            loss_KL = nn.KLDivLoss()
            loss_MSE = nn.MSELoss()


        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        trainloader = DataLoader(
            MOTSDataSet(args.trainset_dir, args.train_list, max_iters=args.itrs_each_epoch * args.batch_size,
                        crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                        edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=num_worker)

        valloader = DataLoader(
            MOTSValDataSet(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=4,shuffle=True,num_workers=num_worker)


        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999

        layer_num = [0,2,6]

        for epoch in range(0,args.num_epochs):
            model.train()

            task0_pool_image = ImagePool(8)
            task0_pool_mask = ImagePool(8)
            task0_pool_weight = ImagePool(8)
            task0_scale = []
            task0_layer = []

            task1_pool_image = ImagePool(8)
            task1_pool_mask = ImagePool(8)
            task1_pool_weight = ImagePool(8)
            task1_scale = []
            task1_layer = []

            task2_pool_image = ImagePool(8)
            task2_pool_mask = ImagePool(8)
            task2_pool_weight = ImagePool(8)
            task2_scale = []
            task2_layer = []

            task3_pool_image = ImagePool(8)
            task3_pool_mask = ImagePool(8)
            task3_pool_weight = ImagePool(8)
            task3_scale = []
            task3_layer = []

            task4_pool_image = ImagePool(8)
            task4_pool_mask = ImagePool(8)
            task4_pool_weight = ImagePool(8)
            task4_scale = []
            task4_layer = []

            task5_pool_image = ImagePool(8)
            task5_pool_mask = ImagePool(8)
            task5_pool_weight = ImagePool(8)
            task5_scale = []
            task5_layer = []

            task6_pool_image = ImagePool(8)
            task6_pool_mask = ImagePool(8)
            task6_pool_weight = ImagePool(8)
            task6_scale = []
            task6_layer = []

            task7_pool_image = ImagePool(8)
            task7_pool_mask = ImagePool(8)
            task7_pool_weight = ImagePool(8)
            task7_scale = []
            task7_layer = []


            if epoch < args.start_epoch:
                continue

            if engine.distributed:
                train_sampler.set_epoch(epoch)

            epoch_loss = []
            adjust_learning_rate(optimizer, epoch, args.learning_rate, args.num_epochs, args.power)

            batch_size = args.batch_size
            task_num = 8
            each_loss = torch.zeros((task_num)).cuda()
            count_batch = torch.zeros((task_num)).cuda()
            loss_weight = torch.ones((task_num)).cuda()


            for iter, batch in enumerate(trainloader):

                # if iter > 10:
                #     break

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3]
                l_ids = batch[4].cuda()
                t_ids = batch[5].cuda()
                s_ids = batch[6].cuda()

                sum_loss = 0

                for ki in range(len(imgs)):
                    now_task = layer_num[l_ids[ki]] + t_ids[ki]

                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))
                        task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_pool_weight.add(wt[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                        task0_layer.append((l_ids[ki]))
                    elif now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_pool_weight.add(wt[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                        task1_layer.append((l_ids[ki]))
                    elif now_task == 2:
                        task2_pool_image.add(imgs[ki].unsqueeze(0))
                        task2_pool_mask.add(lbls[ki].unsqueeze(0))
                        task2_pool_weight.add(wt[ki].unsqueeze(0))
                        task2_scale.append((s_ids[ki]))
                        task2_layer.append((l_ids[ki]))
                    elif now_task == 3:
                        task3_pool_image.add(imgs[ki].unsqueeze(0))
                        task3_pool_mask.add(lbls[ki].unsqueeze(0))
                        task3_pool_weight.add(wt[ki].unsqueeze(0))
                        task3_scale.append((s_ids[ki]))
                        task3_layer.append((l_ids[ki]))
                    elif now_task == 4:
                        task4_pool_image.add(imgs[ki].unsqueeze(0))
                        task4_pool_mask.add(lbls[ki].unsqueeze(0))
                        task4_pool_weight.add(wt[ki].unsqueeze(0))
                        task4_scale.append((s_ids[ki]))
                        task4_layer.append((l_ids[ki]))
                    elif now_task == 5:
                        task5_pool_image.add(imgs[ki].unsqueeze(0))
                        task5_pool_mask.add(lbls[ki].unsqueeze(0))
                        task5_pool_weight.add(wt[ki].unsqueeze(0))
                        task5_scale.append((s_ids[ki]))
                        task5_layer.append((l_ids[ki]))
                    elif now_task == 6:
                        task6_pool_image.add(imgs[ki].unsqueeze(0))
                        task6_pool_mask.add(lbls[ki].unsqueeze(0))
                        task6_pool_weight.add(wt[ki].unsqueeze(0))
                        task6_scale.append((s_ids[ki]))
                        task6_layer.append((l_ids[ki]))
                    elif now_task == 7:
                        task7_pool_image.add(imgs[ki].unsqueeze(0))
                        task7_pool_mask.add(lbls[ki].unsqueeze(0))
                        task7_pool_weight.add(wt[ki].unsqueeze(0))
                        task7_scale.append((s_ids[ki]))
                        task7_layer.append((l_ids[ki]))



                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    labels = task0_pool_mask.query(batch_size)
                    wts = task0_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task0_scale.pop(0)
                        layers[bi] = task0_layer.pop(0)

                    now_task = 0
                    weight = edge_weight ** wts
                    
                    # if seed > semi_ratio:
                    
                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all
                    #sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))


                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    wts = task1_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task1_scale.pop(0)
                        layers[bi] = task1_layer.pop(0)

                        
                    now_task = 1
                    weight = edge_weight ** wts

                    seed = np.random.rand(1)

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

                if task2_pool_image.num_imgs >= batch_size:
                    images = task2_pool_image.query(batch_size)
                    labels = task2_pool_mask.query(batch_size)
                    wts = task2_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task2_scale.pop(0)
                        layers[bi] = task2_layer.pop(0)

                    now_task = 2
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

                if task3_pool_image.num_imgs >= batch_size:
                    images = task3_pool_image.query(batch_size)
                    labels = task3_pool_mask.query(batch_size)
                    wts = task3_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task3_scale.pop(0)
                        layers[bi] = task3_layer.pop(0)

                    now_task = 3
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

                if task4_pool_image.num_imgs >= batch_size:
                    images = task4_pool_image.query(batch_size)
                    labels = task4_pool_mask.query(batch_size)
                    wts = task4_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task4_scale.pop(0)
                        layers[bi] = task4_layer.pop(0)

                    now_task = 4
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

                if task5_pool_image.num_imgs >= batch_size:
                    images = task5_pool_image.query(batch_size)
                    labels = task5_pool_mask.query(batch_size)
                    wts = task5_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task5_scale.pop(0)
                        layers[bi] = task5_layer.pop(0)

                    now_task = 5
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size, scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))


                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

                if task6_pool_image.num_imgs >= batch_size:
                    images = task6_pool_image.query(batch_size)
                    labels = task6_pool_mask.query(batch_size)
                    wts = task6_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task6_scale.pop(0)
                        layers[bi] = task6_layer.pop(0)

                    now_task = 6
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
                                                                                   scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))

                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))

                if task7_pool_image.num_imgs >= batch_size:
                    images = task7_pool_image.query(batch_size)
                    labels = task7_pool_mask.query(batch_size)
                    wts = task7_pool_weight.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    layers = torch.ones(batch_size).cuda()
                    for bi in range(len(scales)):
                        scales[bi] = task7_scale.pop(0)
                        layers[bi] = task7_layer.pop(0)

                    now_task = 7
                    weight = edge_weight ** wts

                    'supervise_learning'
                    term_seg_Dice, term_seg_BCE, Sup_term_all = supervise_learning(images, labels, batch_size,
                                                                                   scales,
                                                                                   model, now_task, weight,
                                                                                   loss_seg_DICE, loss_seg_CE)

                    reduce_Dice = engine.all_reduce_tensor(term_seg_Dice)
                    reduce_BCE = engine.all_reduce_tensor(term_seg_BCE)
                    reduce_all = engine.all_reduce_tensor(Sup_term_all)

                    optimizer.zero_grad()
                    reduce_all.backward()
                    optimizer.step()

                    print(
                        'Epoch {}: {}/{}, lr = {:.4}, Dice = {:.4}, BCE = {:.4}, loss_Sum = {:.4}'.format( \
                            epoch, iter, len(trainloader), optimizer.param_groups[0]['lr'], reduce_Dice.item(),
                            reduce_BCE.item(), reduce_all.item()))

                    term_all = reduce_all

                    # sum_loss += term_all
                    each_loss[now_task] += term_all
                    count_batch[now_task] += 1

                    epoch_loss.append(float(term_all))


            epoch_loss = np.mean(epoch_loss)

            all_tr_loss.append(epoch_loss)

            if (args.local_rank == 0):
                print('Epoch_sum {}: lr = {:.4}, loss_Sum = {:.4}'.format(epoch, optimizer.param_groups[0]['lr'],
                                                                          epoch_loss.item()))
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Train_loss', epoch_loss.item(), epoch)

            if (epoch >= 0) and (args.local_rank == 0) and (((epoch % 10 == 0) and (epoch >= 800)) or (epoch % 1 == 0)):
                print('save validation image ...')

                model.eval()
                # semi_pool_image = ImagePool(8 * 6)
                task0_pool_image = ImagePool(8)
                task0_pool_mask = ImagePool(8)
                task0_scale = []
                task0_layer = []

                task1_pool_image = ImagePool(8)
                task1_pool_mask = ImagePool(8)
                task1_scale = []
                task1_layer = []

                task2_pool_image = ImagePool(8)
                task2_pool_mask = ImagePool(8)
                task2_scale = []
                task2_layer = []

                task3_pool_image = ImagePool(8)
                task3_pool_mask = ImagePool(8)
                task3_scale = []
                task3_layer = []

                task4_pool_image = ImagePool(8)
                task4_pool_mask = ImagePool(8)
                task4_scale = []
                task4_layer = []

                task5_pool_image = ImagePool(8)
                task5_pool_mask = ImagePool(8)
                task5_scale = []
                task5_layer = []

                task6_pool_image = ImagePool(8)
                task6_pool_mask = ImagePool(8)
                task6_scale = []
                task6_layer = []

                task7_pool_image = ImagePool(8)
                task7_pool_mask = ImagePool(8)
                task7_scale = []
                task7_layer = []


                val_loss = np.zeros((task_num))
                val_F1 = np.zeros((task_num))
                val_Dice = np.zeros((task_num))
                val_TPR = np.zeros((task_num))
                val_PPV = np.zeros((task_num))
                cnt = np.zeros((task_num))

                # for iter, batch1, batch2 in enumerate(zip(valloaderloader, semi_valloaderloader)):
                with torch.no_grad():
                    for iter, batch1 in enumerate(valloader):

                        'dataloader'
                        imgs = batch1[0].cuda()
                        lbls = batch1[1].cuda()
                        wt = batch1[2].cuda().float()
                        volumeName = batch1[3]
                        l_ids = batch1[4].cuda()
                        t_ids = batch1[5].cuda()
                        s_ids = batch1[6].cuda()

                        # semi_img = batch2[0]

                        for ki in range(len(imgs)):
                            now_task = layer_num[l_ids[ki]] + t_ids[ki]

                            # if now_task <= 9 :
                            #     continue

                            if now_task == 0:
                                task0_pool_image.add(imgs[ki].unsqueeze(0))
                                task0_pool_mask.add(lbls[ki].unsqueeze(0))
                                task0_scale.append((s_ids[ki]))
                                task0_layer.append((l_ids[ki]))
                            elif now_task == 1:
                                task1_pool_image.add(imgs[ki].unsqueeze(0))
                                task1_pool_mask.add(lbls[ki].unsqueeze(0))
                                task1_scale.append((s_ids[ki]))
                                task1_layer.append((l_ids[ki]))
                            elif now_task == 2:
                                task2_pool_image.add(imgs[ki].unsqueeze(0))
                                task2_pool_mask.add(lbls[ki].unsqueeze(0))
                                task2_scale.append((s_ids[ki]))
                                task2_layer.append((l_ids[ki]))
                            elif now_task == 3:
                                task3_pool_image.add(imgs[ki].unsqueeze(0))
                                task3_pool_mask.add(lbls[ki].unsqueeze(0))
                                task3_scale.append((s_ids[ki]))
                                task3_layer.append((l_ids[ki]))
                            elif now_task == 4:
                                task4_pool_image.add(imgs[ki].unsqueeze(0))
                                task4_pool_mask.add(lbls[ki].unsqueeze(0))
                                task4_scale.append((s_ids[ki]))
                                task4_layer.append((l_ids[ki]))
                            elif now_task == 5:
                                task5_pool_image.add(imgs[ki].unsqueeze(0))
                                task5_pool_mask.add(lbls[ki].unsqueeze(0))
                                task5_scale.append((s_ids[ki]))
                                task5_layer.append((l_ids[ki]))
                            elif now_task == 6:
                                task6_pool_image.add(imgs[ki].unsqueeze(0))
                                task6_pool_mask.add(lbls[ki].unsqueeze(0))
                                task6_scale.append((s_ids[ki]))
                                task6_layer.append((l_ids[ki]))
                            elif now_task == 7:
                                task7_pool_image.add(imgs[ki].unsqueeze(0))
                                task7_pool_mask.add(lbls[ki].unsqueeze(0))
                                task7_scale.append((s_ids[ki]))
                                task7_layer.append((l_ids[ki]))


                        output_folder = os.path.join(args.snapshot_dir.replace('snapshots_2D/','/Data/HATs/validation_'), str(epoch))
                        #output_folder = os.path.join('/Data/DoDNet/a_DynConv/validation_noscale_0829', str(epoch))
                        if not os.path.exists(output_folder):
                            os.makedirs(output_folder)
                        optimizer.zero_grad()


                        'medulla'
                        if task0_pool_image.num_imgs >= batch_size:
                            images = task0_pool_image.query(batch_size)
                            labels = task0_pool_mask.query(batch_size)
                            now_task = torch.tensor(0)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task0_scale.pop(0)
                                layers[bi] = task0_layer.pop(0)

                            preds = torch.zeros((batch_size, task_num, 1024, 1024)).cuda()
                            preds[:,:,:512,:512],_ = model(images[:,:,:512,:512], torch.ones(batch_size).cuda()*0, scales)
                            preds[:,:,:512,512:],_ = model(images[:,:,:512,512:], torch.ones(batch_size).cuda()*0, scales)
                            preds[:,:,512:,512:],_ = model(images[:,:,512:,512:], torch.ones(batch_size).cuda()*0, scales)
                            preds[:,:,512:,:512],_ = model(images[:,:,512:,:512], torch.ones(batch_size).cuda()*0, scales)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())

                            rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())

                        'cortex'
                        if task1_pool_image.num_imgs >= batch_size:
                            images = task1_pool_image.query(batch_size)
                            labels = task1_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task1_scale.pop(0)
                                layers[bi] = task1_layer.pop(0)

                            preds = torch.zeros((batch_size, task_num, 1024, 1024)).cuda()
                            preds[:, :, :512, :512], _ = model(images[:, :, :512, :512],
                                                               torch.ones(batch_size).cuda() * 1, scales)
                            preds[:, :, :512, 512:], _ = model(images[:, :, :512, 512:],
                                                               torch.ones(batch_size).cuda() * 1, scales)
                            preds[:, :, 512:, 512:], _ = model(images[:, :, 512:, 512:],
                                                               torch.ones(batch_size).cuda() * 1, scales)
                            preds[:, :, 512:, :512], _ = model(images[:, :, 512:, :512],
                                                               torch.ones(batch_size).cuda() * 1, scales)

                            now_task = torch.tensor(1)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels.long())
                            rmin, rmax, cmin, cmax = 0, 1024, 0, 1024
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, ...].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' %(now_task.item())),
                                           prediction.detach().cpu().numpy())


                        'dt'
                        if task2_pool_image.num_imgs >= batch_size:
                            images = task2_pool_image.query(batch_size)
                            labels = task2_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task2_scale.pop(0)
                                layers[bi] = task2_layer.pop(0)
                            preds, _ = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*2, scales)

                            now_task = torch.tensor(2)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
                            rmin, rmax, cmin, cmax = 128, 384, 128, 384
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi,128:384,128:384]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi,384:640,384:640].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        'pt'
                        if task3_pool_image.num_imgs >= batch_size:
                            images = task3_pool_image.query(batch_size)
                            labels = task3_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task3_scale.pop(0)
                                layers[bi] = task3_layer.pop(0)

                            preds, _ = model(images[:, :, 256:768, 256:768], torch.ones(batch_size).cuda() * 3, scales)

                            now_task = torch.tensor(3)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:, 256:768, 256:768].long())
                            rmin, rmax, cmin, cmax = 128, 384, 128, 384
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi, 128:384, 128:384]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi, :, 384:640, 384:640].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi, 384:640, 384:640].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        'cap'
                        if task4_pool_image.num_imgs >= batch_size:
                            images = task4_pool_image.query(batch_size)
                            labels = task4_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task4_scale.pop(0)
                                layers[bi] = task4_layer.pop(0)

                            preds, _ = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*4, scales)

                            now_task = torch.tensor(4)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
                            rmin, rmax, cmin, cmax = 128, 384, 128, 384
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi,128:384,128:384]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi,384:640,384:640].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        'tuft'
                        if task5_pool_image.num_imgs >= batch_size:
                            images = task5_pool_image.query(batch_size)
                            labels = task5_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task5_scale.pop(0)
                                layers[bi] = task5_layer.pop(0)

                            preds, _ = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda()*5, scales)

                            now_task = torch.tensor(5)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
                            rmin, rmax, cmin, cmax = 128, 384, 128, 384
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin, cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi,128:384,128:384]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi,:,384:640,384:640].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi,384:640,384:640].detach().cpu().numpy())
                                plt.imsave(os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                           prediction.detach().cpu().numpy())

                        'pod'
                        if task6_pool_image.num_imgs >= batch_size:
                            images = task6_pool_image.query(batch_size)
                            labels = task6_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task6_scale.pop(0)
                                layers[bi] = task6_layer.pop(0)
                            preds, _ = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda() * 6, scales)

                            now_task = torch.tensor(6)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
                            rmin, rmax, cmin, cmax = 0, 512, 0, 512
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
                                                             cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi,:,256:768,256:768].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi,256:768,256:768].detach().cpu().numpy())
                                plt.imsave(
                                    os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                    prediction.detach().cpu().numpy())

                        'mes'
                        if task7_pool_image.num_imgs >= batch_size:
                            images = task7_pool_image.query(batch_size)
                            labels = task7_pool_mask.query(batch_size)
                            scales = torch.ones(batch_size).cuda()
                            layers = torch.ones(batch_size).cuda()
                            for bi in range(len(scales)):
                                scales[bi] = task7_scale.pop(0)
                                layers[bi] = task7_layer.pop(0)
                            preds, _ = model(images[:,:,256:768,256:768], torch.ones(batch_size).cuda() * 7, scales)

                            now_task = torch.tensor(7)

                            now_preds = torch.argmax(preds, 1) == now_task
                            now_preds_onehot = one_hot_3D(now_preds.long())

                            labels_onehot = one_hot_3D(labels[:,256:768,256:768].long())
                            rmin, rmax, cmin, cmax = 0, 512, 0, 512
                            F1, DICE, TPR, PPV = count_score(now_preds_onehot, labels_onehot, rmin, rmax, cmin,
                                                             cmax)

                            val_F1[now_task] += F1
                            val_Dice[now_task] += DICE
                            val_TPR[now_task] += TPR
                            val_PPV[now_task] += PPV
                            cnt[now_task] += 1

                            for pi in range(len(images)):
                                prediction = now_preds[pi]
                                num = len(glob.glob(os.path.join(output_folder, '*')))
                                out_image = images[pi,:,256:768,256:768].permute([1, 2, 0]).detach().cpu().numpy()
                                img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                                plt.imsave(os.path.join(output_folder, str(num) + '_image.png'),
                                           img)
                                plt.imsave(os.path.join(output_folder, str(num) + '_mask.png'),
                                           labels[pi,256:768,256:768].detach().cpu().numpy())
                                plt.imsave(
                                    os.path.join(output_folder, str(num) + '_preds_%s.png' % (now_task.item())),
                                    prediction.detach().cpu().numpy())


                    avg_val_F1 = val_F1 / cnt
                    avg_val_Dice = val_Dice / cnt
                    avg_val_TPR = val_TPR / cnt
                    avg_val_PPV = val_PPV / cnt

                    print('Validate \n 0medulla_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}'
                          ' \n 1cortex_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          ' \n 2dt_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          ' \n 3pt_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          ' \n 4cap_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          ' \n 5tuft_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          ' \n 6pod_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          ' \n 7mes_f1={:.4} dsc={:.4} tpr={:.4} ppv={:.4}\n'
                          .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item(),
                                  avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item(),
                                  avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item(),
                                  avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item(),
                                  avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item(),
                                  avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item(),
                                  avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_TPR[6].item(), avg_val_PPV[6].item(),
                                  avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_TPR[7].item(), avg_val_PPV[7].item()))

                    df = pd.DataFrame(columns = ['task','F1','Dice','TPR','PPV'])
                    df.loc[0] = ['0medulla', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_TPR[0].item(), avg_val_PPV[0].item()]
                    df.loc[1] = ['1cortex', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_TPR[1].item(), avg_val_PPV[1].item()]
                    df.loc[2] = ['2dt', avg_val_F1[2].item(), avg_val_Dice[2].item(), avg_val_TPR[2].item(), avg_val_PPV[2].item()]
                    df.loc[3] = ['3pt', avg_val_F1[3].item(), avg_val_Dice[3].item(), avg_val_TPR[3].item(), avg_val_PPV[3].item()]
                    df.loc[4] = ['4cap', avg_val_F1[4].item(), avg_val_Dice[4].item(), avg_val_TPR[4].item(), avg_val_PPV[4].item()]
                    df.loc[5] = ['5tuft', avg_val_F1[5].item(), avg_val_Dice[5].item(), avg_val_TPR[5].item(), avg_val_PPV[5].item()]
                    df.loc[6] = ['6pod', avg_val_F1[6].item(), avg_val_Dice[6].item(), avg_val_TPR[6].item(), avg_val_PPV[6].item()]
                    df.loc[7] = ['7mes', avg_val_F1[7].item(), avg_val_Dice[7].item(), avg_val_TPR[7].item(), avg_val_PPV[7].item()]

                    df.to_csv(os.path.join(output_folder,'validation_result.csv'))


                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_e' + str(epoch) + '.pth'))

            if (epoch >= args.num_epochs - 1) and (args.local_rank == 0):
                print('save model ...')
                if args.FP16:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'amp': amp.state_dict()
                    }
                    torch.save(checkpoint, osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                else:
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir, 'MOTS_DynConv_' + args.snapshot_dir.split('/')[-2] + '_final_e' + str(epoch) + '.pth'))
                break

        end = timeit.default_timer()
        print(end - start, 'seconds')


if __name__ == '__main__':
    main()
