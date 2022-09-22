# FST

# Overview

The Pytorch implementation of _Learning from Future: A Novel Self-Training Framework for Semantic Segmentation._

[[arXiv]](https://arxiv.org/pdf/2209.06993.pdf)

> Self-training has shown great potential in semi-supervised learning. Its core idea is to use the model learned on labeled data to generate pseudo-labels for unlabeled samples, and in turn teach itself. To obtain valid supervision, active attempts typically employ a momentum teacher for pseudo-label prediction yet observe the confirmation bias issue, where the incorrect predictions may provide wrong supervision signals and get accumulated in the training process. The primary cause of such a drawback is that the prevailing self-training framework acts as guiding the current state with previous knowledge, because the teacher is updated with the past student only. To alleviate this problem, we propose a novel self-training strategy, which allows the model to learn from the future. Concretely, at each training step, we first virtually optimize the student (i.e., caching the gradients without applying them to the model weights), then update the teacher with the virtual future student, and finally ask the teacher to produce pseudo-labels for the current student as the guidance. In this way, we manage to improve the quality of pseudo-labels and thus boost the performance. We also develop two variants of our future-self-training (FST) framework through peeping at the future both deeply (FST-D) and widely (FST-W). Taking the tasks of unsupervised domain adaptive semantic segmentation and semi-supervised semantic segmentation as the instances, we experimentally demonstrate the effectiveness and superiority of our approach under a wide range of settings.


<img width="1046" alt="image" src="https://user-images.githubusercontent.com/83934424/190574312-20421c04-1aa5-48a9-ac63-afffaeb83bce.png">


## Preparation

### Envs

For this project, we used python 3.8.5. We recommend setting up a new virtual environment.

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0
```

### Backbones

Download the pre-trained weights using the following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```

### Datasets

Prepare datasets follow the instructions below:

##### Cityscapes

For UDA, download leftImg8bit_trainvaltest.zip and gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to `data/cityscapes`.
For SSL, Next, unzip the files to folder ```data``` and make the dictionary structures as follows:
```angular2html
  data/cityscapes
  ├── gtFine
  │   ├── test
  │   ├── train
  │   └── val
  └── leftImg8bit
      ├── test
      ├── train
      └── val
```

##### GTA5
Download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.

##### SYNTHIA
Download SYNTHIA-RAND-CITYSCAPES from [here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`. The UDA data folder structure should look like this:
```angular2html
  .
  ├── ...
  ├── data
  │   ├── cityscapes
  │   │   ├── leftImg8bit
  │   │   │   ├── train
  │   │   │   ├── val
  │   │   ├── gtFine
  │   │   │   ├── train
  │   │   │   ├── val
  │   ├── gta
  │   │   ├── images
  │   │   ├── labels
  │   ├── synthia (optional)
  │   │   ├── RGB
  │   │   ├── GT
  │   │   │   ├── LABELS
  ├── ...
```

##### VOC 2012
Refer to [this link](https://github.com/zhixuanli/segmentation-paper-reading-notes/blob/master/others/Summary%20of%20the%20semantic%20segmentation%20datasets.md) and download `PASCAL VOC 2012 augmented with SBD` dataset.
Then unzip the files to folder ```data``` and make the dictionary structures as follows:
```angular2html
  data/VOC2012
  ├── Annotations
  ├── ImageSets
  ├── JPEGImages
  ├── SegmentationClass
  ├── SegmentationClassAug
  └── SegmentationObject
```



## Model Zoo

### UDA

| Dataset            | Method  | Backbone                                       |                          checkpoint                          | mIoU  |
| ------------------ | ------- | ---------------------------------------------- | :----------------------------------------------------------: | :---: |
| GTA-Cityscapes     | **FST** | [ResNet-101](https://arxiv.org/abs/1512.03385) | [GoogleDrive](https://drive.google.com/drive/folders/1CxjiTBSr6nwy_nHewK9UkybJZekvd16Q?usp=sharing) | 59.94 |
| GTA-Cityscapes     | **FST** | [Swin-B](https://arxiv.org/abs/2103.14030)     | [GoogleDrive](https://drive.google.com/drive/folders/1M1HBG32I6rDEkTyK5sq-J0VeHU20E8SH?usp=sharing) | 66.80 |
| GTA-Cityscapes     | **FST** | [MiT-B5](https://arxiv.org/abs/2105.15203)     | [GoogleDrive](https://drive.google.com/drive/folders/1dxB5ell6_IGGt_sO0Jrh7DN-iWJ-MQ3W?usp=sharing) | 69.56 |
| SYNTHIA-Cityscapes | **FST** | [MiT-B5](https://arxiv.org/abs/2105.15203)     | [GoogleDrive](https://drive.google.com/drive/folders/1mTi7w2H5OquJMrd96o8M9lDtKtlXqauE?usp=sharing) | 62.26 |

### SSL

| Dataset    | Method  | SegModel                                       |                          checkpoint                          | mIoU(1/16) | mIoU(1/8) | mIoU(1/4) |
| ---------- | ------- | ---------------------------------------------- | :----------------------------------------------------------: | :--------: | :-------: | :-------: |
| VOC2012    | **FST** | [PSPNet](https://arxiv.org/abs/1612.01105)     | [OneDrive](https://1drv.ms/u/s!AgGL9MGcRHv0m3yaDFSDVBFONGXf?e=IL2fte) |   68.35    |   72.77   |   75.90   |
| VOC2012    | **FST** | [DeepLabV2](https://arxiv.org/abs/1606.00915)  | [OneDrive](https://1drv.ms/u/s!AgGL9MGcRHv0m3vgJwWMNz6lrHGI?e=2riJaP) |   69.43    |   73.18   |   76.32   |
| VOC2012    | **FST** | [DeepLabV3+](https://arxiv.org/abs/1802.02611) | [OneDrive](https://1drv.ms/u/s!AgGL9MGcRHv0m3oI_7lsG7vFbs_g?e=huVbHp) |   73.88    |   76.07   |   76.32   |
| Cityscapes | **FST** | [DeepLabV3+](https://arxiv.org/abs/1802.02611) | [OneDrive](https://1drv.ms/u/s!AgGL9MGcRHv0m3c8G1BRN2FoFAiu?e=lRlNDa) |   71.03    |   75.36   |   76.61   |



## Usage


### UDA
#### Training

Run the `./train.sh` command or
```shell
python run_experiments.py --config $CONFIG_FILE
```

#### Testing

To test the trained models, download the trained weights, then run

```shell
sh test.sh path/to/checkpoint_directory
```

The trained models can be found in Model Zoo.

### SSL

#### Training
For example, to train FST on Cityscapes:
```shell
cd semi_seg/experiments/cityscapes/372/mt_fst
sh train.sh
```

#### Testing
```shell
cd semi_seg/experiments/cityscapes/372/mt_fst
sh eval.sh
```




## Citation

```
@misc{https://doi.org/10.48550/arxiv.2209.06993,
  author = {Du, Ye and Shen, Yujun and Wang, Haochen and Fei, Jingjing and Li, Wei and Wu, Liwei and Zhao, Rui and Fu, Zehua and Liu, Qingjie},
  title = {Learning from Future: A Novel Self-Training Framework for Semantic Segmentation},
  publisher = {arXiv},
  year = {2022}
  }
```



## Acknowledgements

This work is based on [DAFormer](https://github.com/lhoyer/DAFormer), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [DACS](https://github.com/vikolss/DACS). We sincerely thank these respositories and their authors for their great work and open source spirit.

