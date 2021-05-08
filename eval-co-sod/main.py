import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset
# from concurrent.futures import ThreadPoolExecutor
def main(cfg):
    root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
    #gt_dir = osp.join(root_dir, 'gt')
    #pred_dir = osp.join(root_dir, 'pred')
    gt_dir = cfg.gt_dir
    pred_dir = cfg.pred_dir
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')

    method_names = ['']
    save_method = pred_dir.split('/')[-1]
    dataset_names = ['CoCA', 'CoSOD3k', 'Cosal2015']
    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, save_method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        print(thread.run())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default=None)

    parser.add_argument('--pred_dir', type=str, default=None)
    parser.add_argument('--gt_dir', type=str, default=None)
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
