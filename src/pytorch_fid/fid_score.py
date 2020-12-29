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
import PIL
import torch
import numpy as np

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy import linalg
from pathlib import Path
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3


class FlatImageFolder(data.Dataset):
    def __init__(self, root, transform=None, ext=('.png', '.jpg', '.jpeg', 'bmp')):
        super().__init__()

        self.files = [f for f in Path(root).iterdir() if f.is_file() and f.suffix.lower() in ext]
        self.transform = transform

    def __len__(self):
        return min(len(self.files), 1000)

    def __getitem__(self, idx):
        filename = self.files[idx]
        X = PIL.Image.open(filename)
        if self.transform:
            X = self.transform(X)
        return X


class FrechetInceptionDistance:
    def __init__(self, model, dims=2048, batch_size=32, num_workers=0, progressbar=False):
        self.model = model
        self.dims = dims
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.progressbar = progressbar

    @staticmethod
    def get_inception_model(dims=2048):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        return InceptionV3([block_idx])

    def get_activations(self, batches):
        self.model.eval()

        device = next(self.model.parameters()).device

        activations = []

        if self.progressbar:
            batches = tqdm(batches)

        for i, batch in enumerate(batches):
            batch = batch.to(device)

            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            activations.append(pred.cpu().data.numpy().reshape(pred.size(0), -1))

        return np.concatenate(activations)

    @staticmethod
    def calculate_activation_statistics(activations):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU

        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    @staticmethod
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

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)

    def get_batches_from_image_folder(self, path):
        transformations = [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
        ]

        images = FlatImageFolder(path, transform=transforms.Compose(transformations))
        batches = DataLoader(
            images,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        return batches

    def get_statistics_for_images(self, path, cache=True):
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

    def calculate_fid_for_image_directories(self, path1, path2, cache=True):
        m1, s1 = self.get_statistics_for_images(path1, cache=cache)
        m2, s2 = self.get_statistics_for_images(path2, cache=cache)

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
                        help='Whether to look for cached statistics or cache computed statistics in the given image '
                             'folders.')
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model = FrechetInceptionDistance.get_inception_model(args.dims).to(args.device)
    fid = FrechetInceptionDistance(model, args.dims, args.batch_size, args.num_workers, progressbar=True)
    fid_score = fid.calculate_fid_for_image_directories(args.path[0], args.path[1])

    print('FID:', fid_score)


if __name__ == '__main__':
    main()
