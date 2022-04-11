# GCoNet
The official repo of the CVPR 2021 paper [Group Collaborative Learning for Co-Salient Object Detection ](https://arxiv.org/abs/2104.01108).


## Trained model
Download `final_gconet.pth` ([Google Drive](https://drive.google.com/file/d/1y1UxatK033mQz1GIA_tdElIHK-peVzz4/view?usp=sharing)). And it is the [training log](https://drive.google.com/file/d/1BBeRIEKjoewMrfwramoxxzJotBzRpuMb/view?usp=sharing).

Put `final_gconet.pth` at `GCoNet/tmp/GCoNet_run1`.

Run `test.sh` for evaluation.

## Data Format

  Put the [DUTS_class (training dataset from GICD)](https://drive.google.com/file/d/1Ej6FKifpRi1bx09I0r7D6MO-GI8SDu_M/view?usp=sharing), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](https://pan.baidu.com/s/1qx1YpuNnqODSl53egDHz5w) (password: cvtt) and [Cosal2015](https://pan.baidu.com/s/191v0XyaCw-Ay7_hUWRWCBQ) (password: qb4g) datasets to `GCoNet/data` as the following structure:
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
           │       └── Cosal2015 (Cosal2015's image files)
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

## Prediction results
The co-saliency maps of GCoNet can be found at [Google Drive](https://drive.google.com/file/d/17LgbcwGNK1DFl9jRAoMxF2796YlQYR4a/view?usp=sharing).


## Note and Discussion

***In your training, you can usually obtain slightly worse performance on CoCA dataset and slightly better perofmance on Cosal2015 and CoSOD3k datasets. The performance fluctuation is around 1.0 point for Cosal2015 and CoSOD3k datasets and around 2.0 points for CoCA dataset.***

We observed that the results on CoCA dataset are unstable when train the model multiple times, and the performance fluctuation can reach around 1.5 ponits (But our performance are still much better than other methods in the worst case).  
***Therefore, we provide our used training pairs and sequences with deterministic data augmentation to help you to reproduce our results on CoCA. (In different machines, these inputs and data augmentation are different but deterministic.) However, there is still randomness in the training stage, and you can obtain different performance on CoCA.***

There are three possible reasons:

1.	It may be caused by the challenging images of CoCA dataset where the target objects are relative small and there are many non-target objects in a complex environment.
2.	The imperfect training dataset. We use the training dataset in GICD, whose labels are produced by the classification model. There are some noisy labels in the training dataset.
3.	The randomness of training groups. In our training, two groups are randomly picked for training. Different collaborative training groups have different training difficulty.

Possible research directions for performance stability:

1.	Reduce label noise. If you want to use the training dataset in GICD to train your model. It is better to use multiple powerful classification models (ensemble) to obtain better class labels.
2.	Deterministic training groups. For two collaborative image groups, you can explore different ways to pick the suitable groups, e.g., pick two most similar groups for hard example mining.

It is a potential research direction to obtain stable results on such challenging real-world images. We follow other CoSOD methods to report the best performance of our model. You need to train the model multiple times to obtain the best result on CoCA dataset. If you want more discussion about it, you can contact me (qfanaa@connect.ust.hk).





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
