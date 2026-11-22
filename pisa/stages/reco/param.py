"""
Reconstruction stage for binned (distribution-mode) pipelines with a
parametrised detector response: the response P(reco bin | true bin) is built
from analytic resolution functions and recomputed when its nuisance
parameters change.
"""

import numpy as np

from pisa import FTYPE
from pisa.core.binning import MultiDimBinning
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.binned_response import (
    apply_response_matrix, expand_factorised_response, load_param_spec,
    make_param_response
)
from pisa.utils.log import logging

__all__ = ['param', 'init_test']

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


class param(Stage):  # pylint: disable=invalid-name
    """
    Parametrised detector response for binned pipelines.

    The response is factorised as in the PISA paper: the event class depends
    on true energy; the reco energy on true energy and class (a Gaussian in
    log10(E_reco / E_true)); the reco cos(zenith) on true energy, true
    cos(zenith) and class (a Gaussian in cos(zenith), or a 2D Gaussian angular
    error converted to cos(zenith)). Functions are tabulated versus true
    energy in a specification file (see
    `pisa.utils.binned_response.load_param_spec`).

    The weights on the true binning (`calc_mode`) are mapped onto the reco
    binning (`apply_mode`) as in `reco.response_matrix`.

    Nuisance parameters act on the functions directly:

    * `reco_energy_scale` multiplies the reco energy;
    * `reco_energy_res_scale` multiplies the width in log10(E_reco / E_true);
    * `reco_coszen_res_scale` multiplies the cos(zenith) or angular width.

    Parameters
    ----------
    param_spec : str
        Path or resource of the parametrisation file.

    params : ParamSet
        Must have parameters::

            reco_energy_scale : quantity (dimensionless), nominal 1
            reco_energy_res_scale : quantity (dimensionless), nominal 1
            reco_coszen_res_scale : quantity (dimensionless), nominal 1

    Notes
    -----
    `calc_mode` must be a (true_energy, true_coszen) binning and `apply_mode`
    a (reco_energy, reco_coszen, pid) binning, with one pid bin per class of
    the specification. True quantities are evaluated at bin centres.

    Expected container keys are::

        "weights"
    """

    def __init__(
        self,
        param_spec,
        **std_kwargs,
    ):
        self.param_spec_source = param_spec
        self.spec = load_param_spec(param_spec)

        supported_reps = {
            'calc_mode': MultiDimBinning,
            'apply_mode': MultiDimBinning,
        }
        super().__init__(
            expected_params=(
                'reco_energy_scale',
                'reco_energy_res_scale',
                'reco_coszen_res_scale',
            ),
            expected_container_keys=('weights',),
            supported_reps=supported_reps,
            **std_kwargs,
        )
        self.matrices = {}

    def setup_function(self):
        if self.calc_mode.num_dims != 2 or self.apply_mode.num_dims != 3:
            raise ValueError(
                'reco.param needs a 2D true binning and a 3D reco binning '
                '(reco_energy, reco_coszen, pid)'
            )
        # Channels that share a parametrisation are computed once
        self.spec_key = {}
        for container in self.data:
            if container.name in self.spec['channels']:
                self.spec_key[container.name] = container.name
            elif 'default' in self.spec['channels']:
                self.spec_key[container.name] = 'default'
            else:
                raise KeyError(
                    f'No parametrisation for "{container.name}" and no "default" '
                    f'in {self.param_spec_source}'
                )

    def compute_function(self):
        energy_scale = self.params.reco_energy_scale.m_as('dimensionless')
        energy_res_scale = self.params.reco_energy_res_scale.m_as('dimensionless')
        coszen_res_scale = self.params.reco_coszen_res_scale.m_as('dimensionless')
        if min(energy_scale, energy_res_scale, coszen_res_scale) <= 0:
            raise ValueError('reco.param scale parameters must be > 0')

        n_true = self.calc_mode.size
        n_reco = self.apply_mode.size
        matrices_by_key = {}
        for key in sorted(set(self.spec_key.values())):
            # make_param_response looks the key up in spec['channels']
            parts = make_param_response(
                self.calc_mode, self.apply_mode, self.spec, [key],
                energy_scale=energy_scale,
                energy_res_scale=energy_res_scale,
                coszen_res_scale=coszen_res_scale,
            )[key]
            full = expand_factorised_response(parts['pid'], parts['energy'], parts['coszen'])
            matrices_by_key[key] = full.reshape(n_true, n_reco).astype(FTYPE)
        for name, key in self.spec_key.items():
            self.matrices[name] = matrices_by_key[key]
        logging.debug('reco.param: built %d response(s)', len(matrices_by_key))

    def apply_function(self):
        for container in self.data:
            apply_response_matrix(container, self.matrices[container.name],
                                  self.calc_mode, self.apply_mode)


def init_test(**param_kwargs):
    """Instantiation example, with input containers from `data.binned_grid`"""
    # pylint: disable=import-outside-toplevel
    from pisa import ureg
    from pisa.core.binning import OneDimBinning
    from pisa.core.container import ContainerSet
    from pisa.stages.data.binned_grid import binned_grid

    true_binning = MultiDimBinning([
        OneDimBinning(name='true_energy', is_log=True, num_bins=6,
                      domain=[1e2, 1e7] * ureg.GeV),
        OneDimBinning(name='true_coszen', is_lin=True, num_bins=4, domain=[-1, 1]),
    ])
    reco_binning = MultiDimBinning([
        OneDimBinning(name='reco_energy', is_log=True, num_bins=4,
                      domain=[1e3, 1e7] * ureg.GeV),
        OneDimBinning(name='reco_coszen', is_lin=True, num_bins=3, domain=[-1, 1]),
        OneDimBinning(name='pid', is_lin=True, num_bins=2, domain=[0, 2],
                      bin_names=['cascade', 'track']),
    ])
    grid = binned_grid(output_names=['nue_cc', 'numu_cc'],
                       calc_mode=true_binning, apply_mode=true_binning)
    grid.data = ContainerSet('data')
    grid.setup()

    param_set = ParamSet([
        Param(name='reco_energy_scale', value=1.05, **param_kwargs),
        Param(name='reco_energy_res_scale', value=1.1, **param_kwargs),
        Param(name='reco_coszen_res_scale', value=0.9, **param_kwargs),
    ])
    stage = param(
        param_spec='reco/toy_param_response.json',
        calc_mode=true_binning,
        apply_mode=reco_binning,
        params=param_set,
    )
    stage.data = grid.data
    return stage
