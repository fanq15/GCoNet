CUDA_VISIBLE_DEVICES=0 python test.py --model GCoNet --param_root tmp/GCoNet_run1 --save_root ./data/SalMaps/
CUDA_VISIBLE_DEVICES=0 python eval-co-sod/main.py --pred_dir ./data/SalMaps/ --gt_dir ./data/sod/cosod/gt/
