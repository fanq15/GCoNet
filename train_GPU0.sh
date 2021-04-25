CUDA_VISIBLE_DEVICES=0 python train.py --model GCoNet --loss DSLoss_IoU_noCAM --trainset DUTS_class --size 224 --tmp tmp/GCoNet_run1 --lr 1e-4 --bs 1 --epochs 50
