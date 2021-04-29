# GCoNet
The official repo of the CVPR 2021 paper [Group Collaborative Learning for Co-Salient Object Detection ](https://arxiv.org/abs/2104.01108).


## Trained model
Download `final_gconet.pth` ([Google Drive](https://drive.google.com/file/d/1y1UxatK033mQz1GIA_tdElIHK-peVzz4/view?usp=sharing)).

## Data Format

  Put the [DUTS_class (training dataset from GICD)](https://drive.google.com/file/d/1Ej6FKifpRi1bx09I0r7D6MO-GI8SDu_M/view?usp=sharing), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015]() datasets to `GCoNet/data` as the following structure:
  ```
  GCoNet
     ├── other codes
     ├── ...
     │ 
     └── data
           ├──── images
           |       ├── DUTS_class (DUTS_class's image files)
           |       ├── CoCA (CoCA's image files)
           |       ├── CoSOD3k (CoSOD3k's image files)
           │       └── Cosal2015 (Cosal2015's image files)
           │ 
           └────── gts
                    ├── DUTS_class (DUTS_class's Groundtruth files)
                    ├── CoCA (CoCA's Groundtruth files)
                    ├── CoSOD3k (CoSOD3k's Groundtruth files)
                    └── Cosal2015 (Cosal2015's Groundtruth files)
  ```  
  
<!-- USAGE EXAMPLES -->
## Usage

Run `sh all.sh` for training (`train_GPU0.sh`) and testing (`test.sh`).

## Note

We observe that the results on CoCA dataset are unstable when train the model multiple times, especifically for the `F-measure` metric. It may be caused by the challenging images of CoCA dataset where the target objects are relative small and there are many non-target objects in a complex environment. [This paper](https://arxiv.org/abs/2007.03380) also discussed the stability problem in CoSOD. It is a potential research direction to obtain stable results on such challenging real-world images. We follow other CoSOD methods to report the best performance of our model. You need to train the model multiple times to obtain the best result on CoCA dataset. If you want more discussion about it, you can concat me (qfanaa@connect.ust.hk).

## Prediction results
The co-saliency maps of GCoNet can be found at [Google Drive](https://drive.google.com/file/d/17LgbcwGNK1DFl9jRAoMxF2796YlQYR4a/view?usp=sharing).




## Citation
  ```
@inproceedings{fan2021gconet,
  title={Group Collaborative Learning for Co-Salient Object Detection},
  author={Fan, Qi and Fan, Deng-Ping and Fu, Huazhu and Tang, Chi-Keung and Shao, Ling and Tai, Yu-Wing},
  booktitle={CVPR},
  year={2021}
}
  ```

## Acknowledgements
[Zhao Zhang](https://github.com/zzhanghub) gives us lots of helps! Our framework is built on his [GICD](https://github.com/zzhanghub/gicd/edit/master/README.md).
