# 对Co-Saliency任务的评价代码
from torch.utils import data
import torch
import os
from PIL import Image
from torchvision import transforms


class EvalDataset(data.Dataset):
    def __init__(self, img_root, gt_root):

        class_list = os.listdir(img_root)
        self.img_dirs = list(
            map(lambda x: os.path.join(img_root, x), class_list))
        self.gt_dirs = list(
            map(lambda x: os.path.join(gt_root, x), class_list))
        self.transform = transforms.ToTensor()

    def __getitem__(self, item):
        names = os.listdir(self.img_dirs[item])
        num = len(names)
        img_paths = list(
            map(lambda x: os.path.join(self.img_dirs[item], x), names))
        gt_paths = list(
            map(lambda x: os.path.join(self.gt_dirs[item], x[:-4]+'.png'), names))
        
        imgs = []
        gts = []
        for idx in range(num):
            img = Image.open(img_paths[idx]).convert('RGB')
            gt = Image.open(gt_paths[idx]).convert('L')
            imgs.append(self.transform(img))
            gts.append(self.transform(gt))
        return imgs, gts

    def __len__(self):
        return len(self.img_dirs)


class Eval():
    def __init__(self, img_root, label_root):
        self.loader = EvalDataset(img_root, label_root)

    def eval_mae(self):
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            for preds, gts in self.loader:
                for pred, gt in zip(preds, gts):
                    pred = pred.cuda()
                    gt = gt.cuda()
                    mae = torch.abs(pred - gt).mean()
                    if mae == mae:  # for Nan
                        avg_mae += mae
                        img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def eval_fmeasure(self):
        # beta2 = 0.3
        # avg_p, avg_r, img_num = 0.0, 0.0, 0.0
        # with torch.no_grad():
        #     for preds, gts in self.loader:
        #         for pred, gt in zip(preds, gts):
        #             pred = pred.cuda()
        #             gt = gt.cuda()
        #             prec, recall = self._eval_pr(pred, gt, 255)
        #             avg_p += prec
        #             avg_r += recall
        #             img_num += 1.0
        #     avg_p /= img_num
        #     avg_r /= img_num
        #     score = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
        #     score[score != score] = 0  # for Nan

            # return score.max().item()
            return 0

    def _eval_pr(self, y_pred, y, num):
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / \
                (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall


if __name__ == "__main__":

    eval_dataset = EvalDataset(img_root='./tmp/run1/Salmaps/', gt_root='./Dataset/iCoSeg/GroundTruth/')
    # preds, gts = eval_dataset.__getitem__(0)
    # print(eval_dataset.__len__())
        
    # for i, outs in enumerate(eval_dataset):
    #     print('--',i)
        



    evaler = Eval(img_root='./tmp/run1/Salmaps/', label_root='./Dataset/iCoSeg/GroundTruth')

    mae = evaler.eval_mae()
    fm = evaler.eval_fmeasure()
