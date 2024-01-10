import numpy as np
import itertools
import pytest
from mysampler.mcmcsampler import MCMCsampler


@pytest.fixture
def valid_stochastic_matrix():
    return np.array([[0.2, 0.8], [0.6, 0.4]])


def test_invalid_stochastic_matrix():
    with pytest.raises(ValueError, match="square"):
        MCMCsampler(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
    with pytest.raises(ValueError, match="positive"):
        MCMCsampler(np.array([[-0.1, 0.2], [0.4, 0.5]]))
    with pytest.raises(ValueError, match="sum to 1"):
        MCMCsampler(np.array([[0.2, 0.8], [0.4, 0.5]]))


def test_valid_stochastic_matrix(valid_stochastic_matrix):
    sampler = MCMCsampler(valid_stochastic_matrix)
    assert np.array_equal(sampler.stochastic_matrix, valid_stochastic_matrix)
    assert np.array_equal(sampler._cumulative_stmatrix, np.array([[0.2, 1], [0.6, 1]]))
    assert sampler._n_states == sampler.stochastic_matrix.shape[0]


def test_sample_chain(valid_stochastic_matrix):
    sampler = MCMCsampler(valid_stochastic_matrix)
    chain_length = 100
    samples, counts = sampler.sample_chain(chain_length)

    assert len(samples) == chain_length
    assert len(counts) == 2
    assert np.sum(counts) == chain_length


def test_sample(valid_stochastic_matrix):
    sampler = MCMCsampler(valid_stochastic_matrix, n_iter=500, n_chains=2)
    result = sampler.sample()
    chains = result["chains"]

    assert len(chains) == sampler._n_states
    assert len(list(itertools.chain(*chains))) == sampler.n_iter
    assert len(result["frequencies"]) == 2
    assert np.isclose(np.sum(result["frequencies"]), 1.0)
