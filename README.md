# ISL

This is the official pytorch implementation for the paper: [*Instance Similarity Learning for Unsupervised Feature Representation*](https://arxiv.org/abs/2108.02721), which is accepted to ICCV2021. The code contains training and testing several network architecture (ResNet18, ResNet50 and AlexNet) on four datasets (SVHN, CIFAR10, CIFAR100 and ImageNet) using our proposed ISL method.

## Quick Start

### Prerequisites

- python 3.5+
- pytorch 1.0.0+
- torchvision 0.2.1 (not compatible with higher version)
- other packages like numpy and PIL

### Dataset Preparation

Please follow the instruction in [this](https://github.com/zhirongw/lemniscate.pytorch) to download the ImageNet dataset. For small datasets like SVHN, you can either download them manually or set the **download** parameter in torchvision.dataset **True** to download them automatically.

After downloading them, please put them in data/, an SSD is highly recommended for training on ImageNet.

## Training and Testing

### Small Datasets

For training on SVHN, CIFAR10 or CIFAR100, please run:

```shell
python small_main.py --data='data/' --arch='resnet18/resnet50/alexnet' --dataset='svhn/cifar10/cifar100'
```

The training code contains testing the weighted $k$NN on features with $k=200$ every 5 epochs. For testing an existing weight file, just run:

```shell
python small_main.py --data='data/' --arch='resnet18/resnet50/alexnet' --dataset='svhn/cifar10/cifar100' --test-only=True --recompute=True --resume='weight_file'
```

### ImageNet

For training on ImageNet, just run:

```shell
python imagenet_main.py --data='data/' --arch='resnet18/resnet50/alexnet'
```

During training, we monitor the weighted $k$NN with $k=1$ every two epochs, that's because using $k=200$ will be slow on big dataset like ImageNet.

For testing using $k$NN with $k=200$, you can run:

```shell
python imagenet_main.py --data='data/' --arch='resnet18/resnet50/alexnet' --test-only=True --recompute=True --resume='weight_file'
```

To reproduce the ResNet ImageNet result in our paper, you need to run the code on a 16GB memory GPU like NVIDIA Tesla V100 (AlexNet can run on a 11 GB memory GPU like RTX 2080Ti). The performance will drop slightly if trained on two GPUs as observed in our experiments. Also, you may need to switch the training stage manually because sometimes the program just fails to identify the end of training GANs and it might not be able to use the best G for neighborhood mining. The total training time lasts for around 4 days in our experiments using a single GPU and batch size equals to 256.

## Citation
Please cite our paper if you find it useful in your research:

```
@article{wang2021instance,
  title={Instance Similarity Learning for Unsupervised Feature Representation},
  author={Wang, Ziwei and Wang, Yunsong and Wu, Ziyi and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:2108.02721},
  year={2021}
}
```

## License

ISL is released under the MIT License. See the LICENSE file for more details.

## Acknowledgements

We thank the authors of following works for opening source their excellent codes.

- [Instance Discrimination](https://github.com/zhirongw/lemniscate.pytorch)
- [AND](https://github.com/Raymond-sci/AND)
- [Local Aggregation](https://github.com/neuroailab/LocalAggregation-Pytorch)
- [GANS in PyTorch](https://github.com/eriklindernoren/PyTorch-GAN)

## Contact

If you have any questions about the code, please contact Ziyi Wu (dazitu616@gmail.com)
