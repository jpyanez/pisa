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
from pisa.core.pipeline import Pipeline
from pisa.core.param import Param, ParamSet
from pisa.utils.binned_response import (
    expand_factorised_response, load_response_file, make_toy_response,
    write_response_file
)
from pisa.utils.log import Levels, logging, set_verbosity

__all__ = ['TRUE_BINNING', 'RECO_BINNING', 'CHANNELS', 'test_binned_pipeline',
           'test_binned_pipeline_astro', 'test_astro_mixed_weights_refused',
           'test_astro_adds_to_weights',
           'test_binned_pipeline_atmospheric']

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


def nominal_response_systematics():
    """
    Response-systematics parameters of reco.response_matrix at nominal values.

    Returns
    -------
    ParamSet
    """
    return ParamSet([
        Param(name='reco_energy_scale', value=1.0, **PARAM_DEFAULTS),
        Param(name='reco_energy_extra_smear', value=0.0, **PARAM_DEFAULTS),
        Param(name='reco_coszen_extra_smear', value=0.0, **PARAM_DEFAULTS),
    ])


def make_test_config(response_file, astro=None):
    """
    Pipeline configuration for the closure test.

    Parameters
    ----------
    response_file : str
        Binned response file with TRUE_BINNING and RECO_BINNING.
    astro : dict or None
        If given, insert `flux.astrophysical` after the grid, with keys
        'astro_norm' and 'astro_delta' (floats). With no flux stage before it,
        the stage replaces the weights by the astrophysical term.

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
        'params': nominal_response_systematics(),
    }
    return cfg


def expected_counts(response_file, astro=None):
    """
    Reco counts summed over channels, computed directly with numpy.

    Parameters
    ----------
    response_file : str
    astro : dict or None
        As in `make_test_config`. Without it the flux is 1 per GeV m^2 s sr.

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

    # Flux per neutrino type, GeV^-1 m^-2 s^-1 sr^-1 (PISA convention), at bin centres
    flux = np.ones(TRUE_BINNING.shape)
    if astro is not None:
        e_centres = TRUE_BINNING['true_energy'].weighted_centers.m_as('GeV')
        astro_1d = (astro['astro_norm'] * 0.787e-14
                    * (e_centres / 1e5) ** (-2.5 + astro['astro_delta']))
        astro_flux = np.repeat(astro_1d[:, None], TRUE_BINNING.shape[1], axis=1)
        # Untouched weights are replaced by the astrophysical term
        flux = astro_flux

    total = np.zeros(RECO_BINNING.shape)
    for name in CHANNELS:
        counts_true = volume * flux * loaded['aeff'][name] * LIVETIME.m_as('s')
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
    """Same closure with flux.astrophysical as the only flux: it finds the
    weights untouched and replaces them."""
    set_verbosity(verbosity)
    astro = {'astro_norm': 1.3, 'astro_delta': 0.2}
    output_hist, reference = run_closure(astro)
    check_closure(output_hist, reference, 'astrophysical only')
    logging.info('<< PASS : test_binned_pipeline_astro >>')


def test_astro_mixed_weights_refused(verbosity=Levels.WARN):
    """flux.astrophysical stops if some containers carry a flux and others not."""
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
    # Only the first channel gets a (fake) flux
    first = grid.data[CHANNELS[0]]
    first['weights'] = first['initial_weights'] * 2e-14

    params = ParamSet([
        Param(name='astro_norm', value=1.0, **PARAM_DEFAULTS),
        Param(name='astro_delta', value=0.0, **PARAM_DEFAULTS),
    ])
    astro = astrophysical(params=params, calc_mode=TRUE_BINNING, apply_mode=TRUE_BINNING)
    astro.data = grid.data
    astro.setup()
    astro.compute()
    try:
        astro.apply()
    except ValueError as err:
        assert 'untouched' in str(err)
    else:
        raise AssertionError('mixed untouched/flux weights not detected')
    logging.info('<< PASS : test_astro_mixed_weights_refused >>')


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
    fake_atm_flux_1d = 2e-14 * (e_centres / 1e5) ** -3.7   # GeV^-1 m^-2 s^-1 sr^-1
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

    astro_flux_1d = 1.3 * 0.787e-14 * (e_centres / 1e5) ** (-2.5 + 0.2)
    astro_flux = np.repeat(astro_flux_1d[:, None], TRUE_BINNING.shape[1], axis=1).ravel()
    for container in astro.data:
        volume = container['initial_weights']
        expected = volume * (fake_atm_flux + astro_flux)
        assert np.allclose(container['weights'], expected, rtol=1e-10, atol=0)
        # The astrophysical share is large enough to be tested at high energy
        assert np.max(astro_flux / fake_atm_flux) > 1
    logging.info('<< PASS : test_astro_adds_to_weights >>')


def make_atmospheric_config(response_file, channels, with_detector=True):
    """
    Binned pipeline with daemonflux, prob3 and the astrophysical flux.

    Parameters
    ----------
    response_file : str
        Binned response file with TRUE_BINNING and RECO_BINNING.
    channels : list of str
        Container names.
    with_detector : bool
        If False, stop after the fluxes (output on the true binning), so that
        the folding can be redone by hand.

    Returns
    -------
    OrderedDict
    """
    # pylint: disable=import-outside-toplevel
    from pisa.stages.flux.daemon_flux import init_test as daemon_init_test
    daemon_params = daemon_init_test(**PARAM_DEFAULTS).params
    for one_param in daemon_params:
        one_param.value = 0.0 * one_param.value.units   # baseline daemonflux

    cfg = OrderedDict()
    cfg['pipeline'] = {
        'name': 'binned_atm_test',
        'output_binning': RECO_BINNING if with_detector else TRUE_BINNING,
        'output_key': ('weights'),
        'detector_name': None,
    }
    cfg[('data', 'binned_grid')] = {
        'calc_mode': TRUE_BINNING, 'apply_mode': TRUE_BINNING, 'output_names': channels,
    }
    cfg[('flux', 'daemon_flux')] = {
        'calc_mode': TRUE_BINNING, 'apply_mode': None,
        # The stage's default grid stops at 100 TeV; daemonflux goes much higher
        'energy_grid': np.logspace(1, 7.5, 400) * ureg.GeV,
        'params': daemon_params,
    }
    cfg[('osc', 'prob3')] = {
        'calc_mode': TRUE_BINNING, 'apply_mode': TRUE_BINNING,
        'params': ParamSet([
            Param(name='detector_depth', value=2.6 * ureg.km, **PARAM_DEFAULTS),
            Param(name='prop_height', value=18 * ureg.km, **PARAM_DEFAULTS),
            Param(name='earth_model', value='osc/PREM_12layer.dat', **PARAM_DEFAULTS),
            Param(name='YeI', value=0.4656, **PARAM_DEFAULTS),
            Param(name='YeO', value=0.4957, **PARAM_DEFAULTS),
            Param(name='YeM', value=0.4957, **PARAM_DEFAULTS),
            Param(name='theta12', value=33.4 * ureg.degree, **PARAM_DEFAULTS),
            Param(name='theta13', value=8.6 * ureg.degree, **PARAM_DEFAULTS),
            Param(name='theta23', value=45 * ureg.degree, **PARAM_DEFAULTS),
            Param(name='deltam21', value=7.4e-5 * ureg.eV**2, **PARAM_DEFAULTS),
            Param(name='deltam31', value=2.5e-3 * ureg.eV**2, **PARAM_DEFAULTS),
            Param(name='deltacp', value=0 * ureg.degree, **PARAM_DEFAULTS),
        ]),
    }
    cfg[('flux', 'astrophysical')] = {
        'calc_mode': TRUE_BINNING, 'apply_mode': TRUE_BINNING,
        'params': ParamSet([
            Param(name='astro_norm', value=1.0, **PARAM_DEFAULTS),
            Param(name='astro_delta', value=0.0, **PARAM_DEFAULTS),
        ]),
    }
    if with_detector:
        cfg[('aeff', 'table')] = {
            'calc_mode': None, 'apply_mode': TRUE_BINNING,
            'response_file': response_file,
            'params': ParamSet([
                Param(name='livetime', value=LIVETIME, **PARAM_DEFAULTS),
                Param(name='aeff_scale', value=1.0, **PARAM_DEFAULTS),
            ]),
        }
        cfg[('reco', 'response_matrix')] = {
            'calc_mode': TRUE_BINNING, 'apply_mode': RECO_BINNING,
            'response_file': response_file,
            'params': nominal_response_systematics(),
        }
    return cfg


def test_binned_pipeline_atmospheric(verbosity=Levels.WARN):
    """
    Binned pipeline with daemonflux + prob3 + astrophysical flux.

    Checks: (1) the detector stages fold the flux-weighted true counts exactly;
    (2) fluxes in the containers are in PISA units (GeV^-1 m^-2 s^-1 sr^-1):
    daemonflux values are bracketed by the package's own values at the
    neighbouring tabulated zenith angles, and the astrophysical flux equals its
    formula; (3) oscillations: upgoing numu survival is ~1 at high energy and
    clearly below 1 near 100 GeV.
    """
    # pylint: disable=import-outside-toplevel
    from daemonflux import Flux
    set_verbosity(verbosity)
    channels = ['nue_cc', 'numu_cc', 'nutau_cc', 'numubar_nc']

    with tempfile.TemporaryDirectory() as tmpdir:
        response_file = os.path.join(tmpdir, 'toy_response.json')
        aeff, response = make_toy_response(TRUE_BINNING, RECO_BINNING, channels)
        write_response_file(response_file, TRUE_BINNING, RECO_BINNING, aeff,
                            response, description='atmospheric closure test')
        loaded = load_response_file(response_file)

        pipe_true = Pipeline(make_atmospheric_config(response_file, channels, False))
        true_maps = pipe_true.get_outputs()
        pipe_reco = Pipeline(make_atmospheric_config(response_file, channels, True))
        reco_maps = pipe_reco.get_outputs()

    # (1) Folding the flux-weighted true counts by hand
    livetime_s = LIVETIME.m_as('s')
    for name in channels:
        counts_true = true_maps[name].nominal_values * loaded['aeff'][name] * livetime_s
        reference = np.tensordot(counts_true, loaded['response'][name], axes=2)
        check_closure(reco_maps[name].nominal_values, reference, f'atm {name}')

    # (2a) daemonflux numu (neutrinos only, as the stage requests) in the container, in m^-2
    container = pipe_true.data['numu_cc']
    energy = container['true_energy']
    coszen = container['true_coszen']
    flux_obj = Flux(location='IceCube', use_calibration=True)
    tab_zenith = np.array(sorted(float(z) for z in flux_obj.zenith_angles))
    tab_strings = {float(z): z for z in flux_obj.zenith_angles}
    zenith = np.degrees(np.arccos(coszen))
    n_checked = 0
    for ientry in range(0, energy.size, 7):
        if energy[ientry] > 3e7:
            continue
        i_hi = np.searchsorted(tab_zenith, zenith[ientry])
        if i_hi == 0 or i_hi == tab_zenith.size:
            continue
        bracket = []
        for one_zen in [tab_zenith[i_hi - 1], tab_zenith[i_hi]]:
            e3_flux = flux_obj.flux(np.array([energy[ientry]]), tab_strings[one_zen], 'numu')
            bracket.append(float(np.atleast_1d(e3_flux)[0]) * 1e4 / energy[ientry]**3)   # cm^-2 -> m^-2
        value = container['nu_flux'][ientry, 1]
        assert min(bracket) * 0.98 <= value <= max(bracket) * 1.02, (energy[ientry], value, bracket)
        n_checked += 1
    assert n_checked > 10
    logging.info('daemonflux numu flux in m^-2 checked at %d grid points', n_checked)

    # (2b) astrophysical flux in m^-2
    expected_astro = 0.787e-14 * (energy / 1e5) ** -2.5
    assert np.allclose(container['astro_flux'], expected_astro, rtol=1e-6, atol=0)

    # (3) Oscillations
    prob_mumu = container['prob_mu']
    upgoing = coszen < -0.8
    high = upgoing * (energy > 1e4)
    low = upgoing * (energy > 1e2) * (energy < 2e2)
    assert np.all(np.abs(prob_mumu[high] - 1) < 1e-3)
    assert np.all(prob_mumu[low] < 0.99)
    logging.info('upgoing P(numu->numu): %.4f near 100 GeV, %.5f above 10 TeV',
                 prob_mumu[low].mean(), prob_mumu[high].mean())

    # Rates per year for the toy detector, for orientation only
    for name in channels:
        logging.info('%s: %.4g events/yr in reco binning', name,
                     reco_maps[name].nominal_values.sum())
    logging.info('<< PASS : test_binned_pipeline_atmospheric >>')


if __name__ == '__main__':
    test_binned_pipeline(verbosity=Levels.INFO)
    test_binned_pipeline_astro(verbosity=Levels.INFO)
    test_astro_mixed_weights_refused(verbosity=Levels.INFO)
    test_astro_adds_to_weights(verbosity=Levels.INFO)
    test_binned_pipeline_atmospheric(verbosity=Levels.INFO)
