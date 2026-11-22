"""
Reconstruction stage for binned (distribution-mode) pipelines: maps expected
counts per true bin onto reco bins with a response matrix
P(reco bin | true bin) read from a file.
"""

import numpy as np

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.binned_response import (
    apply_map_along_axis, apply_response_matrix, gaussian_smearing_map,
    load_response_file, shift_map
)
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

    Response systematics modify the tabulated response along its reco axes,
    treating the content of each reco bin as uniform within the bin:

    * `reco_energy_scale` multiplies the reco energy (a shift of
      log10(scale) along the log10 reco-energy axis);
    * `reco_energy_extra_smear` adds a Gaussian smearing of this width in
      log10(E_reco);
    * `reco_coszen_extra_smear` adds a Gaussian smearing of this width in
      cos(zenith)_reco, reflected at -1 and +1.

    Content moved outside the reco binning is lost, and content that was
    outside it in the file cannot move in. Store the response with a reco
    energy range somewhat wider than the analysis needs if the energy scale
    is varied.

    Parameters
    ----------
    response_file : str
        Path or resource of the binned response file. Its true binning must be
        `calc_mode` and its reco binning `apply_mode`.

    energy_name, coszen_name : str
        Names of the reco energy and cos(zenith) dimensions.

    params : ParamSet
        Must have parameters::

            reco_energy_scale : quantity (dimensionless), nominal 1
            reco_energy_extra_smear : quantity (dimensionless), nominal 0
            reco_coszen_extra_smear : quantity (dimensionless), nominal 0

    Notes
    -----
    Like `utils.hist`, this stage changes the representation of the
    containers from the true to the reco binning. Only binned stages should
    follow it.

    Expected container keys are::

        "weights"
    """

    def __init__(
        self,
        response_file,
        energy_name='reco_energy',
        coszen_name='reco_coszen',
        **std_kwargs,
    ):
        self.response_file = response_file
        self.energy_name = energy_name
        self.coszen_name = coszen_name

        supported_reps = {
            'calc_mode': MultiDimBinning,
            'apply_mode': MultiDimBinning,
        }
        super().__init__(
            expected_params=(
                'reco_energy_scale',
                'reco_energy_extra_smear',
                'reco_coszen_extra_smear',
            ),
            expected_container_keys=('weights',),
            supported_reps=supported_reps,
            **std_kwargs,
        )

        loaded = load_response_file(self.response_file)
        self.true_binning = loaded['true_binning']
        self.reco_binning = loaded['reco_binning']

        # Nominal response tensors (true dims + reco dims) as read from the file
        self.nominal_responses = loaded['response']
        # Matrices of shape (n_true, n_reco), flattened in binning order,
        # including the current systematics; filled in compute
        self.matrices = {}

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
            if container.name not in self.nominal_responses:
                raise KeyError(
                    f'No response for container "{container.name}" in '
                    f'{self.response_file}'
                )
        logging.debug(
            'reco.response_matrix: %d true bins -> %d reco bins',
            self.true_binning.size, self.reco_binning.size
        )

    def compute_function(self):
        """Apply the response systematics to the nominal tensors."""
        energy_scale = self.params.reco_energy_scale.m_as('dimensionless')
        energy_smear = self.params.reco_energy_extra_smear.m_as('dimensionless')
        coszen_smear = self.params.reco_coszen_extra_smear.m_as('dimensionless')
        if energy_scale <= 0 or energy_smear < 0 or coszen_smear < 0:
            raise ValueError('Need reco_energy_scale > 0 and extra smearings >= 0')

        n_true_dims = self.true_binning.num_dims
        reco_names = list(self.reco_binning.names)

        # Bin-to-bin maps along the reco axes (identity at nominal values)
        bin_maps = []
        if self.energy_name in reco_names:
            log10_edges = np.log10(self.reco_binning[self.energy_name].bin_edges.m_as('GeV'))
            energy_map = shift_map(log10_edges, np.log10(energy_scale))
            energy_map = energy_map @ gaussian_smearing_map(log10_edges, energy_smear)
            bin_maps.append((n_true_dims + reco_names.index(self.energy_name), energy_map))
        elif energy_scale != 1 or energy_smear != 0:
            raise ValueError(f'No "{self.energy_name}" dimension for the energy systematics')
        if self.coszen_name in reco_names:
            cz_edges = self.reco_binning[self.coszen_name].bin_edges.m_as('dimensionless')
            coszen_map = gaussian_smearing_map(cz_edges, coszen_smear, reflect_range=(-1., 1.))
            bin_maps.append((n_true_dims + reco_names.index(self.coszen_name), coszen_map))
        elif coszen_smear != 0:
            raise ValueError(f'No "{self.coszen_name}" dimension for the coszen systematic')

        n_true = self.true_binning.size
        n_reco = self.reco_binning.size
        for container in self.data:
            tensor = self.nominal_responses[container.name]
            for axis, one_map in bin_maps:
                tensor = apply_map_along_axis(tensor, one_map, axis)
            self.matrices[container.name] = tensor.reshape(n_true, n_reco).astype(FTYPE)

    def apply_function(self):
        for container in self.data:
            apply_response_matrix(container, self.matrices[container.name],
                                  self.calc_mode, self.apply_mode)


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

    param_set = ParamSet([
        Param(name='reco_energy_scale', value=1.05, **param_kwargs),
        Param(name='reco_energy_extra_smear', value=0.1, **param_kwargs),
        Param(name='reco_coszen_extra_smear', value=0.05, **param_kwargs),
    ])
    stage = response_matrix(
        response_file=response_file,
        calc_mode=loaded['true_binning'],
        apply_mode=loaded['reco_binning'],
        params=param_set,
    )
    stage.data = grid.data
    return stage
