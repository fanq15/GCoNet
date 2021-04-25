# GCoNet
The official repo of the CVPR 2021 paper [Group Collaborative Learning for Co-Salient Object Detection ](https://arxiv.org/abs/2104.01108).


## Trained model
Download `final_gconet.pth` ([Google Drive](https://drive.google.com/file/d/1y1UxatK033mQz1GIA_tdElIHK-peVzz4/view?usp=sharing)).

## Data Format

  Put the [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015]() datasets to `GCoNet/data` as the following structure:
  ```
  GCoNet
     ├── other codes
     ├── ...
     │ 
     └── data
           ├──── images
           |       ├── CoCA (CoCA's image files)
           |       ├── CoSOD3k (CoSOD3k's image files)
           │       └── Cosal2015 (Cosal2015's image files)
           │ 
           └────── gts
                    ├── CoCA (CoCA's Groundtruth files)
                    ├── CoSOD3k (CoSOD3k's Groundtruth files)
                    └── Cosal2015 (Cosal2015's Groundtruth files)
  ```  
  
<!-- USAGE EXAMPLES -->
## Usage
1. Put the trained model to the data dir: `GCoNet/data/final_gconet.pth`.

2. Run `sh test.sh`

## Prediction results
The co-saliency maps of GCoNet can be found at [Google Drive](https://drive.google.com/file/d/17LgbcwGNK1DFl9jRAoMxF2796YlQYR4a/view?usp=sharing).




## Citation
  ```
@inproceedings{fan2020fsod,
  title={Group Collaborative Learning for Co-Salient Object Detection},
  author={Fan, Qi and Fan, Deng-Ping and Fu, Huazhu and Tang, Chi-Keung and Shao, Ling and Tai, Yu-Wing},
  booktitle={CVPR},
  year={2021}
}
  ```

## Acknowledgements
[Zhao Zhang](https://github.com/zzhanghub) and his [GICD](https://github.com/zzhanghub/gicd/edit/master/README.md) give us lots of helps!
