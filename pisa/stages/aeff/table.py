"""
Effective-area stage for binned (distribution-mode) pipelines: multiplies the
weights in each true bin by a tabulated effective area and the livetime.
"""

import numpy as np

from pisa import FTYPE, ureg
from pisa.core.binning import MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.binned_response import load_response_file

__all__ = ['table', 'init_test']

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


class table(Stage):  # pylint: disable=invalid-name
    """
    Tabulated effective area, one table per channel, read from a binned
    response file (see `pisa.utils.binned_response`).

    In each true bin the weights are multiplied by

        aeff_scale * A_eff [m^2] * livetime [s].

    With `data.binned_grid` upstream (weights = bin volume in GeV sr) and a
    flux stage in between (GeV^-1 m^-2 s^-1 sr^-1, the PISA flux convention,
    e.g. `flux.daemon_flux`), the result is the expected number of events per
    true bin.

    Parameters
    ----------
    response_file : str
        Path or resource of the binned response file. Its true binning must
        be the binning this stage runs on.

    params : ParamSet
        Must have parameters::

            livetime : Quantity [time]
            aeff_scale : Quantity [dimensionless]

    Notes
    -----
    Only apply is implemented, on a binning; no translation to events.

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
            'calc_mode': [None],
            'apply_mode': MultiDimBinning,
        }
        super().__init__(
            expected_params=('livetime', 'aeff_scale'),
            expected_container_keys=('weights',),
            supported_reps=supported_reps,
            **std_kwargs,
        )

        loaded = load_response_file(self.response_file)
        self.true_binning = loaded['true_binning']
        # Flattened in binning order, in m^2 to match PISA flux units (per m^2)
        self.aeff_m2 = {}
        for name, aeff_m2 in loaded['aeff'].items():
            self.aeff_m2[name] = aeff_m2.ravel().astype(FTYPE)

    def setup_function(self):
        # The file must describe exactly the binning the weights live on
        same_names = list(self.apply_mode.names) == list(self.true_binning.names)
        if not same_names or self.apply_mode.edges_hash != self.true_binning.edges_hash:
            raise ValueError(
                'The true binning of the response file does not match the '
                f'apply_mode binning of aeff.table:\n{self.true_binning}\n'
                f'vs\n{self.apply_mode}'
            )
        for container in self.data:
            if container.name not in self.aeff_m2:
                raise KeyError(
                    f'No effective area for container "{container.name}" '
                    f'in {self.response_file}'
                )

    def apply_function(self):
        aeff_scale = self.params.aeff_scale.m_as('dimensionless')
        livetime_s = self.params.livetime.m_as('sec')

        for container in self.data:
            scale = aeff_scale * livetime_s * self.aeff_m2[container.name]
            container['weights'] *= scale
            container.mark_changed('weights')


def init_test(**param_kwargs):
    """Instantiation example, with input containers from `data.binned_grid` on
    the true binning of the toy response file"""
    # pylint: disable=import-outside-toplevel
    from pisa.core.container import ContainerSet
    from pisa.stages.data.binned_grid import binned_grid
    response_file = 'reco/toy_binned_response.json'
    true_binning = load_response_file(response_file)['true_binning']

    grid = binned_grid(output_names=['nue_cc', 'numubar_nc'],
                       calc_mode=true_binning, apply_mode=true_binning)
    grid.data = ContainerSet('data')
    grid.setup()

    param_set = ParamSet([
        Param(name='livetime', value=1 * ureg.year, **param_kwargs),
        Param(name='aeff_scale', value=1.0, **param_kwargs),
    ])
    stage = table(response_file=response_file, params=param_set,
                  apply_mode=true_binning)
    stage.data = grid.data
    return stage
