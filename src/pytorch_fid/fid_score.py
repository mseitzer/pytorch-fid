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
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3

IMAGE_EXTENSIONS = (
    'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'
)


class ImagePathDataset(data.Dataset):
    def __init__(self, root, transform=None, ext=IMAGE_EXTENSIONS):
        super().__init__()
        self.files = [f for f in Path(root).iterdir()
                      if f.is_file() and f.suffix.lower()[1:] in ext]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img = Image.open(filename).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


class FrechetInceptionDistance:
    def __init__(self,
                 model,
                 dims=2048,
                 batch_size=32,
                 num_workers=None,
                 progressbar=False):
        self.model = model
        self.dims = dims
        self.batch_size = batch_size
        self.num_workers = cpu_count() if num_workers is None else num_workers
        self.progressbar = progressbar

    @staticmethod
    def get_inception_model(dims=2048):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        return InceptionV3([block_idx])

    def _get_device(self):
        return next(self.model.parameters()).device

    def get_activations(self, batches):
        """
        Calculates the activations of the pool_3 layer for all images.

        Args:
            batches: Iterator returning image batches in pytorch tensor format.

        Returns:
            A numpy array of dimension (num images, dims) containing
            feature activations for given images.
        """

        self.model.eval()

        device = self._get_device()

        activations = []

        if self.progressbar:
            batches = tqdm(batches)

        for batch in batches:
            batch = batch.to(device)

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average
            # pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            activations.append(
                pred.cpu().data.numpy().reshape(pred.size(0), -1)
            )

        return np.concatenate(activations)

    @staticmethod
    def calculate_activation_statistics(activations):
        """
        Calculates statistics used for FID by given feature activations.

        Args:
            activations: Numpy array of dimension (num images, dims)
            containing feature activations.

        Returns:
            Mean and covariance matrix over the given activations.
        """
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def get_activation_statistics(self, batches):
        activations = self.get_activations(batches)
        mu, sigma = self.calculate_activation_statistics(activations)
        return mu, sigma

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate
        Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Args:
            mu1: Mean over first set of activations.
            sigma1: Covariance matrix over first set of activations.
            mu1: Mean over second set of activations.
            sigma1: Covariance matrix over second set of activations.

        Returns:
            The Frechet Inception Distance.
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

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def get_batches_from_image_folder(self, path):
        transformations = [
            transforms.Resize((299, 299)),
            transforms.ToTensor()
        ]

        images = ImagePathDataset(
            path, transform=transforms.Compose(transformations)
        )

        if not len(images):
            raise AssertionError(f'No images found in path {path}')

        batches = DataLoader(
            images,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        return batches

    def get_statistics_for_path(self, path, cache=True):
        path = Path(path)
        if path.is_file() and path.suffix in ['.np', '.npz']:
            cached = path
            cache = True
        else:
            cached = path / f'inception_statistics_{self.dims}.npz'

        if cached.is_file() and cache:
            with np.load(cached) as fp:
                m, s = fp['mu'][:], fp['sigma'][:]
                return m, s

        batches = self.get_batches_from_image_folder(path)

        activations = self.get_activations(batches)
        m, s = self.calculate_activation_statistics(activations)

        if cache:
            np.savez(cached, mu=m, sigma=s)

        return m, s

    def calculate_fid_given_paths(self, path1, path2, cache=True):
        m1, s1 = self.get_statistics_for_path(path1, cache=cache)
        m2, s2 = self.get_statistics_for_path(path2, cache=cache)

        fid_score = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_score


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--device',
                        help='Device to use. Defaults to CPU if not provided.')
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths to the generated images or '
                              'to .npz statistic files'))
    parser.add_argument('--cache', action='store_true',
                        help='Whether to look for cached statistics or cache '
                             'computed statistics in the given image folders.')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model = FrechetInceptionDistance.get_inception_model(args.dims).to(device)
    fid = FrechetInceptionDistance(
        model, args.dims, args.batch_size, args.num_workers, progressbar=True
    )
    fid_score = fid.calculate_fid_given_paths(
        args.path[0], args.path[1], cache=args.cache
    )

    print('FID:', fid_score)


if __name__ == '__main__':
    main()
