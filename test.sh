ID=3001
CUDA_VISIBLE_DEVICES=0 python test.py --model GICD --param_root tmp/GICD_run1 --save_root /home/fanqi/data/SalMaps/$ID
CUDA_VISIBLE_DEVICES=0 python eval-co-sod/main.py --pred_dir /home/fanqi/data/SalMaps/$ID --gt_dir /home/fanqi/data/sod/cosod/gt/
