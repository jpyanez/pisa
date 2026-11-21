"""
Reconstruction stage for binned (distribution-mode) pipelines: maps expected
counts per true bin onto reco bins with a response matrix
P(reco bin | true bin) read from a file.
"""

import numpy as np

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning
from pisa.core.stage import Stage
from pisa.utils.binned_response import load_response_file
from pisa.utils.log import logging

__all__ = ['response_matrix', 'init_test']

__author__ = 'J. P. Yanez'

__license__ = '''Copyright (c) 2014-2026, The IceCube Collaboration
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.'''


class response_matrix(Stage):  # pylint: disable=invalid-name
    """
    Apply a tabulated detector response, one per channel, read from a binned
    response file (see `pisa.utils.binned_response`).

    For each container, the weights on the true binning (`calc_mode`) are
    expected counts per true bin. They are mapped onto the reco binning
    (`apply_mode`) with

        counts_reco[j] = sum_i counts_true[i] * R[i, j],

    where R[i, j] = P(reco bin j | true bin i). This is the same matrix product
    that `utils.hist` uses in binned mode, but with the matrix supplied instead
    of histogrammed from MC events. Event topology is one of the reco
    dimensions (e.g. `pid`).

    Parameters
    ----------
    response_file : str
        Path or resource of the binned response file. Its true binning must be
        `calc_mode` and its reco binning `apply_mode`.

    Notes
    -----
    Like `utils.hist`, this stage changes the representation of the
    containers from the true to the reco binning. Only binned stages should
    follow it.

    If a container has "astro_weights" (written by `flux.astrophysical`),
    they are added to "weights" before the mapping, as in `utils.hist`.

    Expected container keys are::

        "weights"
    """

    def __init__(
        self,
        response_file,
        **std_kwargs,
    ):
        self.response_file = response_file

        supported_reps = {
            'calc_mode': MultiDimBinning,
            'apply_mode': MultiDimBinning,
        }
        super().__init__(
            expected_params=(),
            expected_container_keys=('weights',),
            supported_reps=supported_reps,
            **std_kwargs,
        )

        loaded = load_response_file(self.response_file)
        self.true_binning = loaded['true_binning']
        self.reco_binning = loaded['reco_binning']

        # Matrices of shape (n_true, n_reco), flattened in binning order
        self.matrices = {}
        n_true = self.true_binning.size
        n_reco = self.reco_binning.size
        for name, full in loaded['response'].items():
            self.matrices[name] = full.reshape(n_true, n_reco).astype(FTYPE)

    @staticmethod
    def _same_binning(binning_a, binning_b):
        same_names = list(binning_a.names) == list(binning_b.names)
        return same_names and binning_a.edges_hash == binning_b.edges_hash

    def setup_function(self):
        # The file must describe exactly the binnings this stage connects
        if not self._same_binning(self.calc_mode, self.true_binning):
            raise ValueError(
                'The true binning of the response file does not match '
                f'calc_mode:\n{self.true_binning}\nvs\n{self.calc_mode}'
            )
        if not self._same_binning(self.apply_mode, self.reco_binning):
            raise ValueError(
                'The reco binning of the response file does not match '
                f'apply_mode:\n{self.reco_binning}\nvs\n{self.apply_mode}'
            )
        for container in self.data:
            if container.name not in self.matrices:
                raise KeyError(
                    f'No response for container "{container.name}" in '
                    f'{self.response_file}'
                )
        logging.debug(
            'reco.response_matrix: %d true bins -> %d reco bins',
            self.true_binning.size, self.reco_binning.size
        )

    def apply_function(self):
        for container in self.data:
            # Read the expected counts per true bin
            container.representation = self.calc_mode
            weights = container['weights']
            if 'astro_weights' in container.keys:
                weights = weights + container['astro_weights']

            counts_reco = weights @ self.matrices[container.name]

            # Switch the container to the reco binning and store the result
            container.representation = self.apply_mode
            container['weights'] = counts_reco


def init_test(**param_kwargs):
    """Instantiation example, with input containers from `data.binned_grid` on
    the true binning of the toy response file"""
    # pylint: disable=import-outside-toplevel
    from pisa.core.container import ContainerSet
    from pisa.stages.data.binned_grid import binned_grid
    response_file = 'reco/toy_binned_response.json'
    loaded = load_response_file(response_file)

    grid = binned_grid(output_names=['nue_cc', 'numubar_nc'],
                       calc_mode=loaded['true_binning'],
                       apply_mode=loaded['true_binning'])
    grid.data = ContainerSet('data')
    grid.setup()

    stage = response_matrix(
        response_file=response_file,
        calc_mode=loaded['true_binning'],
        apply_mode=loaded['reco_binning'],
    )
    stage.data = grid.data
    return stage
