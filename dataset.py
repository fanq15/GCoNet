import os
from PIL import Image
import torch
import random
import numpy as np
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import random
import pandas as pd


class CoData(data.Dataset):
    def __init__(self, img_root, gt_root, img_size, transform, max_num, is_train):

        class_list = os.listdir(img_root)
        self.size = [img_size, img_size]
        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))
        self.gt_dirs = list(
            map(lambda x: os.path.join(gt_root, x), class_list))
        self.transform = transform
        self.max_num = max_num
        self.is_train = is_train

    def __getitem__(self, item):
        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))
        gt_paths = list(
            map(lambda x: os.path.join(self.gt_dirs[item], x[:-4]+'.png'), names))

        if self.is_train:
            # random pick one category
            other_cls_ls = list(range(len(self.img_dirs)))
            other_cls_ls.remove(item)
            other_item = random.sample(set(other_cls_ls), 1)[0]

            other_names = os.listdir(self.img_dirs[other_item])
            other_num = len(other_names)
            other_img_paths = list(
                map(lambda x: os.path.join(self.img_dirs[other_item], x), other_names))
            other_gt_paths = list(
                map(lambda x: os.path.join(self.gt_dirs[other_item], x[:-4]+'.png'), other_names))

            final_num = min(num, other_num, self.max_num)

            sampled_list = random.sample(range(num), final_num)
            new_img_paths = [img_paths[i] for i in sampled_list]
            #img_paths = new_img_paths
            new_gt_paths = [gt_paths[i] for i in sampled_list]
            #gt_paths = new_gt_paths
            #num =self.max_num

            other_sampled_list = random.sample(range(other_num), final_num)
            new_img_paths = new_img_paths + [other_img_paths[i] for i in other_sampled_list]
            img_paths = new_img_paths
            new_gt_paths = new_gt_paths + [other_gt_paths[i] for i in other_sampled_list]
            gt_paths = new_gt_paths
            #final_num =self.max_num

            final_num = final_num * 2
        else:
            final_num = num

        imgs = torch.Tensor(final_num, 3, self.size[0], self.size[1])
        gts = torch.Tensor(final_num, 1, self.size[0], self.size[1])

        subpaths = []
        ori_sizes = []
        for idx in range(final_num):
            # print(idx)
            img = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')

            subpaths.append(os.path.join(img_paths[idx].split('/')[-2], img_paths[idx].split('/')[-1][:-4]+'.png'))
            ori_sizes.append((img.size[1], img.size[0]))
            # ori_sizes += ((img.size[1], img.size[0]))

            [img, gt] = self.transform(img, gt)

            imgs[idx] = img
            gts[idx] = gt
        if self.is_train:
            cls_ls = [item] * int(final_num / 2) + [other_item] * int(final_num / 2)
            return imgs, gts, subpaths, ori_sizes, cls_ls
        else:
            return imgs, gts, subpaths, ori_sizes
            # return imgs, gts, class_names, [1, 2]

    def __len__(self):
        return len(self.img_dirs)


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, img, gt):
        # assert img.size == gt.size

        img = img.resize(self.size, Image.BILINEAR)
        gt = gt.resize(self.size, Image.NEAREST)

        return img, gt


class ToTensor(object):
    def __call__(self, img, gt):

        return F.to_tensor(img), F.to_tensor(gt)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, img, gt):
        img = F.normalize(img, self.mean, self.std)

        return img, gt


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        return img, gt




class RandomRotation(object):
    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, gt):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, Image.BILINEAR, self.expand, self.center), F.rotate(gt, angle, Image.NEAREST, self.expand, self.center)



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt):
        for t in self.transforms:
            img, gt = t(img, gt)
        return img, gt

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


# get the dataloader (Note: without data augmentation)
def get_loader(img_root, gt_root, img_size, batch_size, max_num = float('inf'), istrain=True, shuffle=False, num_workers=0, pin=False):
    if istrain:
        transform = Compose([
            FixedResize(img_size),
            RandomHorizontalFlip(),
            RandomRotation((-90, 90)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = Compose([
            FixedResize(img_size),
            # RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = CoData(img_root, gt_root, img_size, transform, max_num, is_train=istrain)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                  pin_memory=pin)
    return data_loader


if __name__ == '__main__':
    img_root = '/disk2TB/co-saliency/Dataset/CoSal2015/Image'
    gt_root = '/disk2TB/co-saliency/Dataset/CoSal2015/GT'
    loader = get_loader(img_root, gt_root, 224, 1)
    for img, gt, subpaths, ori_sizes in loader:
        # print(img.size())
        # print(gt.size())
        print(subpaths)
        # print(ori_sizes)
        print(ori_sizes[0][0].item())
        break
