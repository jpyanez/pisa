"""
Earth-transmission stage for binned (distribution-mode) pipelines: applies
precomputed transfer matrices (absorption, NC feed-down, tau regeneration)
that mix energies and channels within each zenith bin.
"""

import numpy as np

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning
from pisa.core.param import ParamSet
from pisa.core.stage import Stage
from pisa.stages.data.binned_grid import parse_flavint_name
from pisa.utils.transfer_matrix import (
    CHANNELS, channel_index, load_transfer_matrix_file, make_toy_transfer_matrix,
    write_transfer_matrix_file
)

__all__ = ['transfer_matrix', 'init_test']

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

# Relative agreement required between containers that must carry the same flux
FLUX_MATCH_RTOL = 1e-10


class transfer_matrix(Stage):  # pylint: disable=invalid-name
    """
    Propagate the neutrino flux through the Earth with transfer matrices
    built by `pisa.utils.transfer_matrix` (from nuSQuIDS, oscillations off).

    For each true cos(zenith) bin,

        Phi_det[a, E_out] = sum_{b, E_in} T[cz, a, E_out, b, E_in] * Phi_surf[b, E_in],

    with a, b over the six channels nue, ..., nutaubar. The flux of a channel
    is read from the weights of its containers as weights / initial_weights
    (the differential flux at the bin centre, after any oscillation stage);
    the CC and NC containers of a channel must carry the same flux. After the
    product the weights are initial_weights * Phi_det.

    Place it after the flux, oscillation and `flux.astrophysical` stages and
    before `aeff.table`.

    Parameters
    ----------
    transfer_matrix_file : str
        Path or resource of the HDF5 matrix file. Its true-energy and
        cos(zenith) edges must be those of the binning this stage runs on.

    params : ParamSet
        None.

    Notes
    -----
    Only apply is implemented, on a binning with exactly the two dimensions
    named in the file (any order). All six channels must be present, since
    tau regeneration couples them; flux in a missing channel would be
    silently dropped.

    Expected container keys are::

        "weights"
        "initial_weights"
    """

    def __init__(
        self,
        transfer_matrix_file,
        **std_kwargs,
    ):
        self.transfer_matrix_file = transfer_matrix_file

        supported_reps = {
            'calc_mode': [None],
            'apply_mode': MultiDimBinning,
        }
        super().__init__(
            expected_params=(),
            expected_container_keys=('weights', 'initial_weights'),
            supported_reps=supported_reps,
            **std_kwargs,
        )

        loaded = load_transfer_matrix_file(self.transfer_matrix_file)
        if not loaded['computed'].all():
            raise ValueError(f'{self.transfer_matrix_file} is a partial build: '
                             f'{np.sum(~loaded["computed"])} columns missing')
        self.matrix = loaded['matrix'].astype(FTYPE)
        self.energy_binning = loaded['energy_binning']
        self.coszen_binning = loaded['coszen_binning']

        # Filled in setup: container names of each channel, and the axis order
        self.channel_containers = None
        self.cz_first = None

    def setup_function(self):
        # The binning must be the matrices' (energy, coszen) binning
        binning = self.apply_mode
        names = list(binning.names)
        expected_names = [self.energy_binning.name, self.coszen_binning.name]
        if sorted(names) != sorted(expected_names):
            raise ValueError(f'transfer_matrix needs a binning with dimensions {expected_names}, '
                             f'got {names}')
        same_energy = np.allclose(binning[self.energy_binning.name].bin_edges.m_as('GeV'),
                                  self.energy_binning.bin_edges.m_as('GeV'), rtol=1e-10)
        same_coszen = np.allclose(binning[self.coszen_binning.name].bin_edges.m,
                                  self.coszen_binning.bin_edges.m, rtol=1e-10)
        if not (same_energy and same_coszen):
            raise ValueError(
                'The binning of the transfer-matrix file does not match the apply_mode '
                f'binning:\n{self.energy_binning}\n{self.coszen_binning}\nvs\n{binning}'
            )
        self.cz_first = names[0] == self.coszen_binning.name

        # Group the containers by channel (flavour, nu/nubar)
        self.channel_containers = []
        for ichan in range(6):
            self.channel_containers.append([])
        for container in self.data:
            flav, nubar = parse_flavint_name(container.name)
            self.channel_containers[channel_index(flav, nubar)].append(container.name)
        missing = []
        for ichan, one_list in enumerate(self.channel_containers):
            if len(one_list) == 0:
                missing.append(CHANNELS[ichan])
        if missing:
            raise ValueError(f'transfer_matrix needs containers of all six channels; '
                             f'missing {missing}')

    def to_cz_e(self, flat):
        """
        Reshape a flattened per-bin array to (n_cz, n_E).

        Parameters
        ----------
        flat : np.ndarray
            Shape (n_bins,), in the flattening order of the apply binning.

        Returns
        -------
        np.ndarray
            Shape (n_cz, n_E).
        """
        grid = flat.reshape(self.apply_mode.shape)
        if self.cz_first:
            return grid
        return grid.T

    def from_cz_e(self, grid):
        """Inverse of `to_cz_e`: (n_cz, n_E) to the flattened binning order."""
        if self.cz_first:
            return grid.ravel()
        return grid.T.ravel()

    def apply_function(self):
        n_cz = self.coszen_binning.num_bins
        n_e = self.energy_binning.num_bins

        # Surface flux per channel, from the first container of each channel
        flux_in = np.zeros((n_cz, 6, n_e), dtype=FTYPE)
        for ichan, one_list in enumerate(self.channel_containers):
            first = self.data[one_list[0]]
            flux = first['weights'] / first['initial_weights']
            flux_in[:, ichan, :] = self.to_cz_e(flux)
            # CC and NC containers of a channel must agree, or the mixing is ill-defined
            for other_name in one_list[1:]:
                other = self.data[other_name]
                other_flux = other['weights'] / other['initial_weights']
                if not np.allclose(other_flux, flux, rtol=FLUX_MATCH_RTOL, atol=0.):
                    raise ValueError(f'Containers {one_list[0]} and {other_name} carry '
                                     'different fluxes; transfer_matrix cannot mix them')

        # Detector flux: matrix product in every zenith bin
        flux_out = np.einsum('zaibj,zbj->zai', self.matrix, flux_in)

        for ichan, one_list in enumerate(self.channel_containers):
            new_flux = self.from_cz_e(flux_out[:, ichan, :])
            for name in one_list:
                container = self.data[name]
                container['weights'] = container['initial_weights'] * new_flux
                container.mark_changed('weights')


def init_test(**param_kwargs):
    """Instantiation example, with an analytic toy matrix on a small binning
    and input containers of all six channels from `data.binned_grid`"""
    # pylint: disable=import-outside-toplevel
    import os
    import tempfile
    from pisa import ureg
    from pisa.core.binning import OneDimBinning
    from pisa.core.container import ContainerSet
    from pisa.stages.data.binned_grid import binned_grid

    true_binning = MultiDimBinning([
        OneDimBinning(name='true_energy', is_log=True, num_bins=6,
                      domain=[1e3, 1e8] * ureg.GeV),
        OneDimBinning(name='true_coszen', is_lin=True, num_bins=4, domain=[-1, 1]),
    ])
    energy_edges = true_binning['true_energy'].bin_edges.m_as('GeV')
    coszen_edges = true_binning['true_coszen'].bin_edges.m
    matrix_file = os.path.join(tempfile.mkdtemp(), 'toy_transfer_matrix.hdf5')
    write_transfer_matrix_file(matrix_file,
                               make_toy_transfer_matrix(energy_edges, coszen_edges),
                               energy_edges, coszen_edges,
                               description='analytic toy, not physics')

    channel_names = ['nue_cc', 'numu_cc', 'nutau_cc', 'nuebar_cc', 'numubar_cc',
                     'nutaubar_cc', 'numu_nc']
    grid = binned_grid(output_names=channel_names,
                       calc_mode=true_binning, apply_mode=true_binning)
    grid.data = ContainerSet('data')
    grid.setup()

    stage = transfer_matrix(transfer_matrix_file=matrix_file, params=ParamSet([]),
                            apply_mode=true_binning)
    stage.data = grid.data
    return stage
