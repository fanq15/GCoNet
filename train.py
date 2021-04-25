import torch
import torch.nn as nn
import torch.optim as optim
from util import Logger, AverageMeter, save_checkpoint, save_tensor_img, set_seed
import os
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse
from tqdm import tqdm
from dataset import get_loader
from criterion import Eval
import torchvision.utils as vutils

import torch.nn.functional as F
import pytorch_toolbelt.losses as PTL

# Parameter from command line
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model',
                    default='CoSalNet',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--loss',
                    default='DSLoss_IoU',
                    type=str,
                    help="Options: '', ''")
parser.add_argument('--bs', '--batch_size', default=1, type=int)
parser.add_argument('--lr',
                    '--learning_rate',
                    default=1e-4,
                    type=float,
                    help='Initial learning rate')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    help='path to latest checkpoint')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--trainset',
                    default='Jigsaw2_DUTS',
                    type=str,
                    help="Options: 'Jigsaw2_DUTS', 'DUTS_class'")
parser.add_argument('--valset',
                    default='CoSal15',
                    type=str,
                    help="Options: 'CoSal15', 'CoCA'")
parser.add_argument('--size',
                    default=224,
                    type=int,
                    help='input size')
parser.add_argument('--tmp', default=None, help='Temporary folder')

args = parser.parse_args()


# Prepare dataset
if args.trainset == 'Jigsaw2_DUTS':
    train_img_path = '../Dataset/Jigsaw2_DUTS/img/'
    train_gt_path = '../Dataset/Jigsaw2_DUTS/gt/'
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=20,
                              istrain=True,
                              shuffle=False,
                              num_workers=4,
                              pin=True)
elif args.trainset == 'DUTS_class':
    train_img_path = '/home/fanqi/data/sod/DUTS_class/img/'
    train_gt_path = '/home/fanqi/data/sod/DUTS_class/gt/'
    train_loader = get_loader(train_img_path,
                              train_gt_path,
                              args.size,
                              args.bs,
                              max_num=16, #20,
                              istrain=True,
                              shuffle=False,
                              num_workers=8, #4,
                              pin=True)

else:
    print('Unkonwn train dataset')
    print(args.dataset)


if args.valset == 'CoSal15':
    val_img_path = '/home/fanqi/data/sod/Cosal2015/Image/'
    val_gt_path = '/home/fanqi/data/sod/Cosal2015/GroundTruth/'
    val_loader = get_loader(val_img_path,
                            val_gt_path,
                            args.size,
                            1,
                            istrain=False,
                            shuffle=False,
                            num_workers=4,
                            pin=True)
elif args.valset == 'CoCA':
    val_img_path = '/home/fanqi/data/sod/CoCA/image/'
    val_gt_path = '/home/fanqi/data/sod/CoCA/binary/'
    val_loader = get_loader(val_img_path,
                            val_gt_path,
                            args.size,
                            1,
                            istrain=False,
                            shuffle=False,
                            num_workers=4,
                            pin=True)
else:
    print('Unkonwn val dataset')
    print(args.dataset)

# make dir for tmp
os.makedirs(args.tmp, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.tmp, "log.txt"))

set_seed(1996)

# Init model
device = torch.device("cuda")

exec('from models import ' + args.model)
model = eval(args.model+'()')
model = model.to(device)

backbone_params = list(map(id, model.ginet.backbone.parameters()))
base_params = filter(lambda p: id(p) not in backbone_params,
                     model.ginet.parameters())

all_params = [{'params': base_params}, {'params': model.ginet.backbone.parameters(), 'lr': args.lr * 0.01}]

# Setting optimizer
optimizer = optim.Adam(params=all_params, lr=args.lr, betas=[0.9, 0.99])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma = 0.1)

for key, value in model.named_parameters():
    if 'ginet.backbone' in key and 'ginet.backbone.conv5.conv5_3' not in key:
        value.requires_grad = False

for key, value in model.named_parameters():
    print(key,  value.requires_grad)

# log model and optimizer pars
logger.info("Model details:")
logger.info(model)
logger.info("Optimizer details:")
logger.info(optimizer)
logger.info("Scheduler details:")
logger.info(scheduler)
logger.info("Other hyperparameters:")
logger.info(args)

# Setting Loss
exec('from loss import ' + args.loss)
dsloss = eval(args.loss+'()')


def main():
    val_mae_record = []
    val_fm_record = []

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.ginet.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    print(args.epochs)
    #best_mae = 100.0
    for epoch in range(args.start_epoch, args.epochs):


        train_loss = train(epoch)
        #if epoch <= 3:
        #    val_mae = validate(epoch)
        #val_mae_record.append(val_mae)

        # Save checkpoint
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.ginet.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            path=args.tmp)
        '''
        if val_mae <= best_mae:
            best_mae = val_mae
            logger.info('best epoch: ' + str(args.epochs))

            ginet_dict = model.ginet.state_dict()
            torch.save(ginet_dict, os.path.join(args.tmp, 'best_gicd_ginet.pth'))
        '''
    ginet_dict = model.ginet.state_dict()
    torch.save(ginet_dict, os.path.join(args.tmp, 'final_gicd_ginet.pth'))

def train(epoch):
    loss_log = AverageMeter()

    # Switch to train mode
    model.train()
    model.set_mode('train')
    #CE = torch.nn.BCEWithLogitsLoss()
    FL = PTL.BinaryFocalLoss()

    for batch_idx, batch in enumerate(train_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        cls_gts = torch.LongTensor(batch[-1]).to(device)
        
        gts_neg = torch.full_like(gts, 0.0)
        gts_cat = torch.cat([gts, gts_neg], dim=0)
        #print(cls_gts, gts.shape)
        scaled_preds, pred_cls, pred_x5 = model(inputs)

        loss_sal = dsloss(scaled_preds, gts)
        loss_cls = F.cross_entropy(pred_cls, cls_gts) * 3.0
        loss_x5 = FL(pred_x5, gts_cat) * 250.0
        loss = loss_sal + loss_cls + loss_x5

        loss_log.update(loss, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            # NOTE: Top2Down; [0] is the grobal slamap and [5] is the final output
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]  '
                        'Train Loss: loss_sal: {4:.3f}, loss_cls: {5:.3f}, loss_x5: {6:.3f} '
                        'Loss_total: {loss.val:.3f} ({loss.avg:.3f})  '.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            len(train_loader),
                            loss_sal,
                            loss_cls,
                            loss_x5,
                            loss=loss_log,
                        ))
    scheduler.step()
    logger.info('@==Final== Epoch[{0}/{1}]  '
                'Train Loss: {loss.avg:.3f}  '.format(epoch,
                                                      args.epochs,
                                                      loss=loss_log))

    return loss_log.avg


def validate(epoch):

    # Switch to evaluate mode
    model.eval()
    model.set_mode('test')
    showbags = []

    saved_root = os.path.join(args.tmp, 'Salmaps')
    # make dir for saving results
    os.makedirs(saved_root, exist_ok=True)

    for batch in tqdm(val_loader):
        inputs = batch[0].to(device).squeeze(0)
        gts = batch[1].to(device).squeeze(0)
        subpaths = batch[2]
        ori_sizes = batch[3]

        scaled_preds = model(inputs)[-1]

        num = len(scaled_preds)

        os.makedirs(os.path.join(saved_root, subpaths[0][0].split('/')[0]),
                    exist_ok=True)

        for inum in range(num):
            subpath = subpaths[inum][0]
            ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
            res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0),
                                            size=ori_size,
                                            mode='bilinear',
                                            align_corners=True)
            save_tensor_img(res, os.path.join(saved_root, subpath))
    
    evaler = Eval(img_root=saved_root, label_root=val_gt_path)

    mae = evaler.eval_mae()

    logger.info('@==Final== Epoch[{0}/{1}]  '
                'MAE: {mae:.3f}  '
                .format(epoch, args.epochs, mae=mae))

    return mae


if __name__ == '__main__':
    main()