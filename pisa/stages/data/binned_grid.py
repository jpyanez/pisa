"""
Stage that creates containers directly on a binning in true variables, without
any events. It is the entry point of a fully binned ("distribution mode")
pipeline: every bin of the true binning is one entry of the containers.
"""

import numpy as np

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.container import Container
from pisa.core.stage import Stage
from pisa.utils.log import logging

__all__ = ['binned_grid', 'parse_flavint_name', 'init_test']

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


FLAV_CODES = {'nue': 0, 'numu': 1, 'nutau': 2}


def parse_flavint_name(name):
    """
    Get flavour code and neutrino/antineutrino flag from a container name.

    Parameters
    ----------
    name : str
        Container name such as 'nue_cc', 'numubar_nc' or 'nutaubar_cc'.

    Returns
    -------
    flav : int
        0, 1 or 2 for e, mu, tau (the convention of `data.grid`).
    nubar : int
        +1 for neutrinos, -1 for antineutrinos.
    """
    flav_part = name.split('_')[0]
    nubar = 1
    if flav_part.endswith('bar'):
        nubar = -1
        flav_part = flav_part[:-len('bar')]
    if flav_part not in FLAV_CODES:
        raise ValueError(
            f'Cannot read a neutrino flavour from container name "{name}"'
        )
    return FLAV_CODES[flav_part], nubar


class binned_grid(Stage):  # pylint: disable=invalid-name
    """
    Create one container per neutrino channel, defined on the true binning
    given as `calc_mode` (and `apply_mode`). No events are made.

    The binning dimensions (e.g. `true_energy`, `true_coszen`) are available
    in each container as the weighted bin centres. The key `initial_weights`
    holds the phase-space volume of each bin,

        Delta E [GeV] * 2 pi * Delta cos(zenith) [sr],

    so that a differential flux in GeV^-1 m^-2 s^-1 sr^-1 (the PISA flux
    convention), multiplied by an effective area in m^2 and a livetime in s,
    gives the expected number of events in each bin. `weights` are reset to `initial_weights` on every
    apply.

    Parameters
    ----------
    output_names : sequence of str
        Container names, one per neutrino channel, e.g. 'nue_cc',
        'nuebar_cc', ..., 'nutaubar_nc'.

    energy_name : str
        Name of the true-energy dimension of the binning.

    coszen_name : str
        Name of the true cos(zenith) dimension of the binning.

    Notes
    -----
    Fluxes and other quantities are evaluated at the bin centres. For steep
    spectra the true-energy bins must be narrow enough that this is a good
    approximation of the bin average.

    Calc and apply modes must be the same binning: once quantities are binned,
    later stages must not translate them back to events.
    """

    def __init__(
        self,
        output_names,
        energy_name='true_energy',
        coszen_name='true_coszen',
        **std_kwargs,
    ):
        self.output_names = list(output_names)
        self.energy_name = energy_name
        self.coszen_name = coszen_name

        supported_reps = {
            'calc_mode': MultiDimBinning,
            'apply_mode': MultiDimBinning,
        }
        super().__init__(
            expected_params=(),
            expected_container_keys=(),
            supported_reps=supported_reps,
            **std_kwargs,
        )

    def setup_function(self):
        if self.calc_mode != self.apply_mode:
            raise ValueError(
                'binned_grid needs calc_mode and apply_mode to be the same '
                'binning, so that nothing is translated back to events.'
            )
        grid_binning = self.calc_mode
        for dim_name in [self.energy_name, self.coszen_name]:
            if dim_name not in grid_binning.names:
                raise ValueError(
                    f'Dimension "{dim_name}" missing from the binning '
                    f'{grid_binning.names}'
                )

        bin_volumes = self.get_bin_volumes(grid_binning)
        logging.debug(
            'binned_grid: %d channels on %d true bins',
            len(self.output_names), grid_binning.size
        )

        for name in self.output_names:
            container = Container(name, representation=grid_binning)

            flav, nubar = parse_flavint_name(name)
            container.set_aux_data('nubar', nubar)
            container.set_aux_data('flav', flav)

            container['initial_weights'] = np.copy(bin_volumes)
            container['weights'] = np.copy(bin_volumes)
            self.data.add_container(container)

    def get_bin_volumes(self, grid_binning):
        """
        Phase-space volume of every bin, flattened in binning order.

        Parameters
        ----------
        grid_binning : MultiDimBinning
            True binning with an energy and a cos(zenith) dimension. Any other
            dimension contributes its plain bin width.

        Returns
        -------
        np.ndarray
            Shape (grid_binning.size,), units GeV sr (times the units of any
            extra dimensions).
        """
        widths = []
        for dim in grid_binning:
            if dim.name == self.energy_name:
                one_width = dim.bin_widths.m_as('GeV')
            elif dim.name == self.coszen_name:
                # Solid angle of a cos(zenith) band, integrated over azimuth
                one_width = 2. * np.pi * dim.bin_widths.m_as('dimensionless')
            else:
                one_width = dim.bin_widths.m
            widths.append(one_width)

        mesh = np.meshgrid(*widths, indexing='ij')
        volumes = np.ones(grid_binning.shape)
        for one_mesh in mesh:
            volumes = volumes * one_mesh
        return volumes.ravel().astype(FTYPE)

    def apply_function(self):
        # Start every evaluation from the bin volumes
        for container in self.data:
            container['weights'] = np.copy(container['initial_weights'])


def init_test(**param_kwargs):
    """Instantiation example"""
    from pisa import ureg  # pylint: disable=import-outside-toplevel
    true_binning = MultiDimBinning([
        OneDimBinning(name='true_energy', is_log=True, num_bins=4,
                      domain=[1e2, 1e4] * ureg.GeV),
        OneDimBinning(name='true_coszen', is_lin=True, num_bins=3,
                      domain=[-1, 1]),
    ])
    return binned_grid(
        output_names=['nue_cc', 'numubar_nc'],
        calc_mode=true_binning,
        apply_mode=true_binning,
    )
