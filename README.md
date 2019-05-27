# Fréchet Inception Distance (FID score) in PyTorch

This is a port of the official implementation of [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500) to PyTorch. 
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) for the original implementation using Tensorflow.

FID is a measure of similarity between two datasets of images. 
It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network. 

Further insights and an independent evaluation of the FID score can be found in [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337).

**Note that the official implementation gives slightly different scores.** If you report FID scores in your paper, and you want them to be exactly comparable to FID scores reported in other papers, you should use [the official Tensorflow implementation](https://github.com/bioinf-jku/TTUR).
You can still use this version if you want a quick FID estimate without installing Tensorflow.

**Update:** The weights and the model are now exactly the same as in the official Tensorflow implementation, and I verified them to give the same results (around `1e-8` mean absolute error) on single inputs on my platform. However, due to differences in the image interpolation implementation and library backends, FID results might still differ slightly from the original implementation. A test I ran (details are to come) resulted in `.08` absolute error and `0.0009` relative error. 

## Usage

Requirements:
- python3
- pytorch
- torchvision
- numpy
- scipy

To compute the FID score between two datasets, where images of each dataset are contained in an individual folder:
```
./fid_score.py path/to/dataset1 path/to/dataset2
```

To run the evaluation on GPU, use the flag `--gpu N`, where `N` is the index of the GPU to use. 

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

## License

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).
