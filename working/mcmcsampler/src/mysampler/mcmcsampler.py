import numpy as np
from numba import njit
from multiprocessing.dummy import Pool
import random


class MCMCsampler:
    """Sample from a discrete states Markov process."""

    def __init__(self, stmatrix: np.ndarray, n_iter: int = 10000, n_chains=4) -> None:
        """Initialize stochastic matrix and chain lenghts."""
        self.stochastic_matrix = stmatrix
        self.n_iter = n_iter
        self.n_chains = n_chains

    @property
    def stochastic_matrix(self):
        """Stochastic matrix of the Markov process."""
        return self._stochastic_matrix

    @stochastic_matrix.setter
    def stochastic_matrix(self, stmatrix: np.ndarray) -> None:
        sh = stmatrix.shape
        if not sh[0] == sh[1]:
            raise ValueError("The stochastic matrix must be square.")
        elif not (stmatrix >= 0).all():
            raise ValueError("All elements must be zero or positive.")
        elif not (np.isclose(np.sum(stmatrix, axis=1), 1)).all():
            raise ValueError("All rows must sum to 1.")
        self._stochastic_matrix = stmatrix
        self._cumulative_stmatrix = np.array(
            [[sum(p[: i + 1]) for i in range(sh[0])] for p in stmatrix]
        )
        self._n_states = sh[0]

    def sample(self) -> dict:
        """Sample from the Markov process, return chains and long-term frequencies."""
        samples_per_chain = [int(self.n_iter / self.n_chains)] * self.n_chains
        samples_per_chain[-1] += self.n_iter % self.n_chains

        pool = Pool(processes=self.n_chains)
        results = pool.map(self.sample_chain, samples_per_chain)
        sampling_results = {"chains": [], "frequencies": np.zeros(self._n_states)}
        for chain, counts in results:
            sampling_results["chains"].append(chain)
            sampling_results["frequencies"] += counts
        sampling_results["frequencies"] /= self.n_iter
        return sampling_results

    def sample_chain(self, n_samples: int):
        """Sample a single markov chain."""
        return _sample_chain(n_samples, self._n_states, self._cumulative_stmatrix)


@njit
def _sample_chain(
    n_samples: int, n_states: int, cumulative_stmatrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a single markov chain."""
    current_state = random.randint(0, n_states - 1)
    samples = np.zeros(n_samples)
    counts = np.zeros(n_states)

    for i in range(n_samples):
        counts[current_state] += 1
        samples[i] = current_state
        nrand = random.random()
        current_state = np.searchsorted(cumulative_stmatrix[current_state], nrand)
    return samples, counts
