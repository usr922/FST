# FST

# Overview

The Pytorch implementation of _Learning from Future: A Novel Self-Training Framework for Semantic Segmentation._

[[arXiv]](https://arxiv.org/pdf/2209.06993.pdf)

> Self-training has shown great potential in semi-supervised learning. Its core idea is to use the model learned on labeled data to generate pseudo-labels for unlabeled samples, and in turn teach itself. To obtain valid supervision, active attempts typically employ a momentum teacher for pseudo-label prediction yet observe the confirmation bias issue, where the incorrect predictions may provide wrong supervision signals and get accumulated in the training process. The primary cause of such a drawback is that the prevailing self-training framework acts as guiding the current state with previous knowledge, because the teacher is updated with the past student only. To alleviate this problem, we propose a novel self-training strategy, which allows the model to learn from the future. Concretely, at each training step, we first virtually optimize the student (i.e., caching the gradients without applying them to the model weights), then update the teacher with the virtual future student, and finally ask the teacher to produce pseudo-labels for the current student as the guidance. In this way, we manage to improve the quality of pseudo-labels and thus boost the performance. We also develop two variants of our future-self-training (FST) framework through peeping at the future both deeply (FST-D) and widely (FST-W). Taking the tasks of unsupervised domain adaptive semantic segmentation and semi-supervised semantic segmentation as the instances, we experimentally demonstrate the effectiveness and superiority of our approach under a wide range of settings.


<img width="1046" alt="image" src="https://user-images.githubusercontent.com/83934424/190574312-20421c04-1aa5-48a9-ac63-afffaeb83bce.png">




## Preparation

For this project, we used python 3.8.5. We recommend setting up a new virtual environment.

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0
```

Download the pre-trained weights using the following script. If problems occur with the automatic download, please follow
the instructions for a manual download within the script.

```shell
sh tools/download_checkpoints.sh
```



## Model Zoo

To do.



## Usage

### Training

Run the `./train.sh` command or

```shell
python run_experiments.py --config $CONFIG_FILE
```

### Testing

To test the trained models, download the trained weights, then run

```shell
sh test.sh path/to/checkpoint_directory
```

The trained models will be uploaded later.





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

This work is based on DAFormer, MMSegmentation and DACS. We sincerely thank these respositories and their authors.

