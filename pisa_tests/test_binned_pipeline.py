"""
Closure test of a fully binned (distribution-mode) pipeline: data.binned_grid ->
aeff.table -> reco.response_matrix, compared with the same calculation done
directly with numpy.
"""

from collections import OrderedDict
from copy import deepcopy
import os
import tempfile

import numpy as np

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.param import Param, ParamSet
from pisa.utils.binned_response import (
    expand_factorised_response, load_response_file, make_toy_response,
    write_response_file
)
from pisa.utils.log import Levels, logging, set_verbosity

__all__ = ['TRUE_BINNING', 'RECO_BINNING', 'CHANNELS', 'test_binned_pipeline',
           'test_binned_pipeline_astro', 'test_astro_adds_to_weights']

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


PARAM_DEFAULTS = {'prior': None, 'range': None, 'is_fixed': True}

TRUE_BINNING = MultiDimBinning([
    OneDimBinning(name='true_energy', is_log=True, num_bins=20,
                  domain=[1e2, 1e7] * ureg.GeV),
    OneDimBinning(name='true_coszen', is_lin=True, num_bins=10, domain=[-1, 1]),
])

RECO_BINNING = MultiDimBinning([
    OneDimBinning(name='reco_energy', is_log=True, num_bins=8,
                  domain=[1e3, 1e7] * ureg.GeV),
    OneDimBinning(name='reco_coszen', is_lin=True, num_bins=5, domain=[-1, 1]),
    OneDimBinning(name='pid', is_lin=True, num_bins=2, domain=[0, 2],
                  bin_names=['cascade', 'track']),
])

CHANNELS = ['nue_cc', 'numu_cc', 'numubar_nc']

LIVETIME = 3.15e7 * ureg.s


def make_test_config(response_file, astro=None):
    """
    Pipeline configuration for the closure test.

    Parameters
    ----------
    response_file : str
        Binned response file with TRUE_BINNING and RECO_BINNING.
    astro : dict or None
        If given, insert `flux.astrophysical` after the grid, with keys
        'astro_only' (bool), 'astro_norm' and 'astro_delta' (floats).

    Returns
    -------
    OrderedDict
        Configuration accepted by DistributionMaker.
    """
    cfg = OrderedDict()
    cfg['pipeline'] = {
        'name': 'binned_test',
        'output_binning': RECO_BINNING,
        'output_key': ('weights'),
        'detector_name': None,
    }
    cfg[('data', 'binned_grid')] = {
        'calc_mode': TRUE_BINNING,
        'apply_mode': TRUE_BINNING,
        'output_names': CHANNELS,
    }
    if astro is not None:
        cfg[('flux', 'astrophysical')] = {
            'calc_mode': TRUE_BINNING,
            'apply_mode': TRUE_BINNING,
            'astro_only': astro['astro_only'],
            'params': ParamSet([
                Param(name='astro_norm', value=astro['astro_norm'], **PARAM_DEFAULTS),
                Param(name='astro_delta', value=astro['astro_delta'], **PARAM_DEFAULTS),
            ]),
        }
    cfg[('aeff', 'table')] = {
        'calc_mode': None,
        'apply_mode': TRUE_BINNING,
        'response_file': response_file,
        'params': ParamSet([
            Param(name='livetime', value=LIVETIME, **PARAM_DEFAULTS),
            Param(name='aeff_scale', value=1.0, **PARAM_DEFAULTS),
        ]),
    }
    cfg[('reco', 'response_matrix')] = {
        'calc_mode': TRUE_BINNING,
        'apply_mode': RECO_BINNING,
        'response_file': response_file,
    }
    return cfg


def expected_counts(response_file, astro=None):
    """
    Reco counts summed over channels, computed directly with numpy.

    Parameters
    ----------
    response_file : str
    astro : dict or None
        As in `make_test_config`. Without it the flux is 1 per GeV cm^2 s sr.

    Returns
    -------
    np.ndarray
        Shape of RECO_BINNING.
    """
    loaded = load_response_file(response_file)

    # Bin volume dE * 2 pi dcosz, in GeV sr
    widths_e = TRUE_BINNING['true_energy'].bin_widths.m_as('GeV')
    widths_cz = TRUE_BINNING['true_coszen'].bin_widths.m
    volume = widths_e[:, None] * 2 * np.pi * widths_cz[None, :]

    # Flux per neutrino type, GeV^-1 cm^-2 s^-1 sr^-1, evaluated at bin centres
    flux = np.ones(TRUE_BINNING.shape)
    if astro is not None:
        e_centres = TRUE_BINNING['true_energy'].weighted_centers.m_as('GeV')
        astro_1d = (astro['astro_norm'] * 0.787e-18
                    * (e_centres / 1e5) ** (-2.5 + astro['astro_delta']))
        astro_flux = np.repeat(astro_1d[:, None], TRUE_BINNING.shape[1], axis=1)
        flux = astro_flux if astro['astro_only'] else flux + astro_flux

    total = np.zeros(RECO_BINNING.shape)
    for name in CHANNELS:
        counts_true = volume * flux * loaded['aeff'][name] * 1e4 * LIVETIME.m_as('s')
        # Sum over the true axes of counts_true[e, z] * R[e, z, ...]
        total += np.tensordot(counts_true, loaded['response'][name], axes=2)
    return total


def run_closure(astro=None):
    """
    Run the pipeline and the numpy reference on the same toy response.

    Parameters
    ----------
    astro : dict or None
        As in `make_test_config`.

    Returns
    -------
    output_hist, reference : np.ndarray
        Pipeline output and reference, both of the shape of RECO_BINNING.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        response_file = os.path.join(tmpdir, 'toy_response.json')
        aeff, response = make_toy_response(TRUE_BINNING, RECO_BINNING, CHANNELS)
        # Store one channel as a full tensor to exercise both kinds
        parts = response['numu_cc']
        response['numu_cc'] = expand_factorised_response(
            parts['pid'], parts['energy'], parts['coszen'])
        write_response_file(response_file, TRUE_BINNING, RECO_BINNING, aeff,
                            response, description='closure test')

        dmaker = DistributionMaker([make_test_config(response_file, astro)])
        output = dmaker.get_outputs(return_sum=True)[0]
        reference = expected_counts(response_file, astro)
    return output.nominal_values, reference


def check_closure(output_hist, reference, label):
    """
    Assert that pipeline output and reference agree, and are not empty.

    Parameters
    ----------
    output_hist, reference : np.ndarray
    label : str
        Name used in log messages.
    """
    rel_diff = np.abs(output_hist - reference) / np.maximum(reference, 1e-300)
    logging.info('%s: total %.4g, max relative difference %.3g',
                 label, reference.sum(), rel_diff.max())
    assert output_hist.shape == reference.shape
    # Guard against a trivial pass with empty maps
    assert reference.sum() > 0 and np.count_nonzero(reference) > reference.size // 2
    assert np.allclose(output_hist, reference, rtol=1e-5, atol=0), rel_diff.max()


def test_binned_pipeline(verbosity=Levels.WARN):
    """Pipeline output equals the direct numpy calculation (flux = 1)."""
    set_verbosity(verbosity)
    output_hist, reference = run_closure()
    check_closure(output_hist, reference, 'flux = 1')
    logging.info('<< PASS : test_binned_pipeline >>')


def test_binned_pipeline_astro(verbosity=Levels.WARN):
    """Same closure with flux.astrophysical, replacing and adding to the weights."""
    set_verbosity(verbosity)
    for astro_only in [True, False]:
        astro = {'astro_only': astro_only, 'astro_norm': 1.3, 'astro_delta': 0.2}
        output_hist, reference = run_closure(astro)
        check_closure(output_hist, reference, f'astro_only={astro_only}')
    logging.info('<< PASS : test_binned_pipeline_astro >>')


def test_astro_adds_to_weights(verbosity=Levels.WARN):
    """flux.astrophysical adds initial_weights * astro_flux to incoming weights.

    The incoming weights are set to a fake atmospheric term of the same order
    as the astrophysical one, so that a missing or doubled addition is visible.
    """
    # pylint: disable=import-outside-toplevel
    from pisa.core.container import ContainerSet
    from pisa.stages.data.binned_grid import binned_grid
    from pisa.stages.flux.astrophysical import astrophysical
    set_verbosity(verbosity)

    grid = binned_grid(output_names=CHANNELS, calc_mode=TRUE_BINNING,
                       apply_mode=TRUE_BINNING)
    grid.data = ContainerSet('data')
    grid.setup()
    grid.apply()

    e_centres = TRUE_BINNING['true_energy'].weighted_centers.m_as('GeV')
    fake_atm_flux_1d = 2e-18 * (e_centres / 1e5) ** -3.7   # GeV^-1 cm^-2 s^-1 sr^-1
    fake_atm_flux = np.repeat(fake_atm_flux_1d[:, None], TRUE_BINNING.shape[1], axis=1).ravel()
    for container in grid.data:
        container['weights'] = container['initial_weights'] * fake_atm_flux

    params = ParamSet([
        Param(name='astro_norm', value=1.3, **PARAM_DEFAULTS),
        Param(name='astro_delta', value=0.2, **PARAM_DEFAULTS),
    ])
    astro = astrophysical(params=params, calc_mode=TRUE_BINNING, apply_mode=TRUE_BINNING)
    astro.data = grid.data
    astro.setup()
    astro.compute()
    astro.apply()

    astro_flux_1d = 1.3 * 0.787e-18 * (e_centres / 1e5) ** (-2.5 + 0.2)
    astro_flux = np.repeat(astro_flux_1d[:, None], TRUE_BINNING.shape[1], axis=1).ravel()
    for container in astro.data:
        volume = container['initial_weights']
        expected = volume * (fake_atm_flux + astro_flux)
        assert np.allclose(container['weights'], expected, rtol=1e-10, atol=0)
        # The astrophysical share is large enough to be tested at high energy
        assert np.max(astro_flux / fake_atm_flux) > 1
    logging.info('<< PASS : test_astro_adds_to_weights >>')


if __name__ == '__main__':
    test_binned_pipeline(verbosity=Levels.INFO)
    test_binned_pipeline_astro(verbosity=Levels.INFO)
    test_astro_adds_to_weights(verbosity=Levels.INFO)
