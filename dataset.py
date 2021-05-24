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
#random.seed(666)

class CoData(data.Dataset):
    def __init__(self, img_root, gt_root, img_size, transform, max_num, group_dict, is_train):

        class_list = os.listdir(img_root)
        self.size = [img_size, img_size]
        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))
        self.gt_dirs = list(
            map(lambda x: os.path.join(gt_root, x), class_list))
        self.transform = transform
        self.max_num = max_num
        self.is_train = is_train
        if self.is_train:
            self.group_dict = group_dict

    def __getitem__(self, item):
        if self.is_train:
            img_ls = self.group_dict[item]

        subpaths = []
        ori_sizes = []
        if self.is_train:
            _, other_item, img_ls = img_ls.strip().split(':')
            other_item = int(other_item)
            img_ls = img_ls.strip().split(',')
            if img_ls[-1] == '':
                img_ls = img_ls[:-1]

            final_num = len(img_ls)

            imgs = torch.Tensor(final_num, 3, self.size[0], self.size[1])
            gts = torch.Tensor(final_num, 1, self.size[0], self.size[1])

            img_paths = img_ls
            gt_paths = []
            old_cls_ls = []
            cls_ls = []
            for idx, img_item in enumerate(img_paths):
                img_item, rotation, flip = img_item.split(';')
                angle = float(rotation)
                flip = float(flip)
                #print(angle, flip)
                gt_item = img_item.replace('images', 'gts').replace('.jpg', '.png')
                cls_id = int(img_item.split('/')[-2])
                old_cls_ls.append(cls_id)

                img = Image.open(img_item).convert('RGB')
                gt = Image.open(gt_item).convert('L')

                img = img.resize((224, 224), Image.BILINEAR)
                gt = gt.resize((224, 224), Image.NEAREST)

                if flip == 1:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
                img, gt = F.rotate(img, angle, Image.BILINEAR, False, None), F.rotate(gt, angle, Image.NEAREST, False, None)
                
                subpaths.append(os.path.join(img_item.split('/')[-2], img_item.split('/')[-1][:-4]+'.png'))
                ori_sizes.append((img.size[1], img.size[0]))

                [img, gt] = self.transform(img, gt)

                imgs[idx] = img
                gts[idx] = gt

            for cls_item in old_cls_ls:
                if cls_item == old_cls_ls[0]:
                    cls_ls.append(item)
                else:
                    cls_ls.append(int(other_item))
        else:
            names = os.listdir(self.img_dirs[item])
            num = len(names)
            img_paths = list(
                map(lambda x: os.path.join(self.img_dirs[item], x), names))
            gt_paths = list(
                map(lambda x: os.path.join(self.gt_dirs[item], x[:-4]+'.png'), names))
            final_num = num
        
            imgs = torch.Tensor(final_num, 3, self.size[0], self.size[1])
            gts = torch.Tensor(final_num, 1, self.size[0], self.size[1])

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
            return imgs, gts, subpaths, ori_sizes, cls_ls
        else:
            return imgs, gts, subpaths, ori_sizes

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
        #print(random.random())
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

        #print(angle)
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
def get_loader(img_root, gt_root, img_size, batch_size, max_num = float('inf'), istrain=True, shuffle=False, num_workers=0, epoch=None, pin=False):
    if istrain:
        transform = Compose([
            #FixedResize(img_size),
            #RandomHorizontalFlip(),
            #RandomRotation((-90, 90)),
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

    
    if istrain:
            group_dict = {}
            df = pd.read_csv('./df_dir/df_group_' + str(epoch) +  '.csv')

            cnt = 0
            with open('./group_dir/group_' + str(epoch) + '.txt', 'r') as f:
                for item in f.readlines():
                    epoch_cur, iter_cur, groups, img_ls = item.strip().split(':')
                    epoch_cur = int(epoch_cur)
                    iter_cur = int(iter_cur)
                    img_ls = img_ls.split(',')[:-1]
                    new_img_ls = ''
                    group_1, group_2 = groups.split(',')
                    group_1 = int(group_1)
                    group_2 = int(group_2)

                    for cnt_cur, img_cur in enumerate(img_ls):
                        df_cur = df[(df['img_path']==img_cur) & (df['item']==group_1)]
                        rotation_cur = df_cur['angle'].values[0]
                        flip_cur = df_cur['flip'].values[0]
                        new_img_ls += img_cur + ';' + str(rotation_cur) + ';' + str(flip_cur) + ','
                    group_dict[group_1] = str(cnt) + ':' + str(group_2) + ':' +  new_img_ls #group_2
    else:
        group_dict = {} 
    dataset = CoData(img_root, gt_root, img_size, transform, max_num, group_dict, is_train=istrain)
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
