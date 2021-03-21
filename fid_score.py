#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
#from models import lenet
from models.inception import InceptionV3
from models.lenet import LeNet5


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    is_numpy = True if type(files[0]) == np.ndarray else False

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        if is_numpy:
            images = np.copy(files[start:end]) + 1
            images /= 2.
        else:
            images = [np.array(Image.open(str(f))) for f in files[start:end]]
            images = np.stack(images).astype(np.float32) / 255.
            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print('done', np.min(images))

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def extract_lenet_features(imgs, net, cuda):
    net.eval()
    feats = []
    imgs = imgs.reshape([-1, 100] + list(imgs.shape[1:]))
    if imgs[0].min() < -0.001:
      imgs = (imgs + 1)/2.0
    print(imgs.shape, imgs.min(), imgs.max())
    if cuda:
        imgs = torch.from_numpy(imgs).cuda()
    else:
        imgs = torch.from_numpy(imgs)
    for i, images in enumerate(imgs):
        feats.append(net.extract_features(images).detach().cpu().numpy())
    feats = np.vstack(feats)
    return feats


def _compute_activations(path, model, batch_size, dims, cuda, model_type):
    if not type(path) == np.ndarray:
        import glob
        jpg = os.path.join(path, '*.jpg')
        png = os.path.join(path, '*.png')
        path = glob.glob(jpg) + glob.glob(png)
        if len(path) > 25000:
            import random
            random.shuffle(path)
            path = path[:25000]
    if model_type == 'inception':
        act = get_activations(path, model, batch_size, dims, cuda)
    elif model_type == 'lenet':
        act = extract_lenet_features(path, model, cuda)

    return act


def calculate_fid_given_paths(paths, batch_size, cuda, dims, bootstrap=True, n_bootstraps=10, model_type='inception'):
    """Calculates the FID of two paths"""
    pths = []
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
        if os.path.isdir(p):
            pths.append(p)
        elif p.endswith('.npy'):
            np_imgs = np.load(p)
            if np_imgs.shape[0] > 25000:
                np_imgs = np_imgs[:50000]
            pths.append(np_imgs)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    if model_type == 'inception':
        model = InceptionV3([block_idx])
    elif model_type == 'lenet':
        model = LeNet5()
        model.load_state_dict(torch.load('./models/lenet.pth'))
    if cuda:
       model.cuda()

    act_true = _compute_activations(pths[0], model, batch_size, dims, cuda, model_type)
    n_bootstraps = n_bootstraps if bootstrap else 1
    pths = pths[1:]
    results = []
    for j, pth in enumerate(pths):
        print(paths[j+1])
        actj = _compute_activations(pth, model, batch_size, dims, cuda, model_type)
        fid_values = np.zeros((n_bootstraps))
        with tqdm(range(n_bootstraps), desc='FID') as bar:
            for i in bar:
                act1_bs = act_true[np.random.choice(act_true.shape[0], act_true.shape[0], replace=True)]
                act2_bs = actj[np.random.choice(actj.shape[0], actj.shape[0], replace=True)]
                m1, s1 = calculate_activation_statistics(act1_bs)
                m2, s2 = calculate_activation_statistics(act2_bs)
                fid_values[i] = calculate_frechet_distance(m1, s1, m2, s2)
                bar.set_postfix({'mean': fid_values[:i+1].mean()})
        results.append((paths[j+1], fid_values.mean(), fid_values.std()))
    return results


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--true', type=str, required=True,
                        help=('Path to the true images'))
    parser.add_argument('--fake', type=str, nargs='+', required=True,
                        help=('Path to the generated images'))
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--model', default='inception', type=str,
                        help='inception or lenet')
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    paths = [args.true] + args.fake

    results = calculate_fid_given_paths(paths, args.batch_size, args.gpu != '', args.dims, model_type=args.model)
    for p, m, s in results:
        print('FID (%s): %.2f (%.3f)' % (p, m, s))
