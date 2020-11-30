import numpy as np
import pytest
import torch

from pytorch_fid import fid_score


@pytest.fixture
def device():
    return torch.device('cpu')


def test_calculate_fid_given_statistics(mocker, tmp_path, device):
    dim = 2048
    m1, m2 = np.zeros((dim,)), np.ones((dim,))
    sigma = np.eye(dim)

    def dummy_statistics(path, model, batch_size, dims, device):
        if path.endswith('1'):
            return m1, sigma
        elif path.endswith('2'):
            return m2, sigma
        else:
            raise ValueError()

    mocker.patch('pytorch_fid.fid_score._compute_statistics_of_path',
                 side_effect=dummy_statistics)

    dir_names = ['1', '2']
    paths = []
    for name in dir_names:
        path = tmp_path / name
        path.mkdir()
        paths.append(str(path))

    fid_value = fid_score.calculate_fid_given_paths(paths,
                                                    batch_size=dim,
                                                    device=device,
                                                    dims=dim)

    assert fid_value == np.sum((m1 - m2)**2)
