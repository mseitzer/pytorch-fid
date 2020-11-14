# Metrics for Evaluating GANs (Pytorch)

The following GAN metrics are implemented:

1. Fréchet Inception Distance (FID)
2. Kernel Inception Distance (KID)


## Usage

Requirements:
- python3
- pytorch
- torchvision
- numpy
- scipy
- scikit-learn
- Pillow

To compute the FID or KID score between two datasets with features extracted from inception net:

* Ensure that you have saved both datasets as numpy files (`.npy`) in channels-first format, i.e. `(no. of images, channels, height, width)`

```
python fid_score.py --true path/to/real/images.npy --fake path/to/gan/generated.npy --gpu ID
```
```
python kid_score.py --true path/to/real/images.npy --fake path/to/gan/generated.npy --gpu ID
```

### Using different layers for feature maps

In difference to the official implementation, you can choose to use a different feature layer of the Inception network instead of the default `pool3` layer. 
As the lower layer features still have spatial extent, the features are first global average pooled to a vector before estimating mean and covariance.

This might be useful if the datasets you want to compare have less than the otherwise required 2048 images. 
Note that this changes the magnitude of the FID score and you can not compare them against scores calculated on another dimensionality. 
The resulting scores might also no longer correlate with visual quality.

You can select the dimensionality of features to use with the flag `--dims N`, where N is the dimensionality of features. 
The choices are:
- 64:   first max pooling features
- 192:  second max pooling featurs
- 768:  pre-aux classifier features
- 2048: final average pooling features (this is the default)

### MNIST

The repo also contains a LeNet (modified from [activatedgeek/LeNet-5](https://github.com/activatedgeek/LeNet-5)) pretrained on MNIST which can be used for evaluating MNIST samples. Just set the model to LeNet using `--model_type lenet`.

```
python fid_score.py --true path/to/real/images.npy --fake path/to/gan/generated.npy --gpu ID --model_type lenet
```
```
python kid_score.py --true path/to/real/images.npy --fake path/to/gan/generated.npy --gpu ID --model_type lenet
```

## License for [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).
