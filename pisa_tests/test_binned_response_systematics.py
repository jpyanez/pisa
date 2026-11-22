"""
Tests of parametrised responses (reco.param) and of response systematics on
tabulated responses (reco.response_matrix): the two implementations are
checked against each other, and both stages against direct numpy
calculations.
"""

from collections import OrderedDict
from copy import deepcopy
import os
import tempfile

import numpy as np

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.pipeline import Pipeline
from pisa.core.param import Param, ParamSet
from pisa.utils.binned_response import (
    apply_map_along_axis, expand_factorised_response, gaussian_smearing_map,
    make_param_response, shift_map, write_response_file
)
from pisa.utils.log import Levels, logging, set_verbosity

__all__ = [
    'test_maps_basic', 'test_energy_scale_cross_check',
    'test_energy_smear_cross_check', 'test_coszen_smear_cross_check',
    'test_angular_model_width', 'test_stage_closures',
]

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

# Fine reco binning, so that the uniform-within-bin approximation of the
# linear maps is good; true binning kept small for speed
TRUE_BINNING = MultiDimBinning([
    OneDimBinning(name='true_energy', is_log=True, num_bins=12,
                  domain=[1e3, 1e6] * ureg.GeV),
    OneDimBinning(name='true_coszen', is_lin=True, num_bins=8, domain=[-1, 1]),
])
RECO_BINNING = MultiDimBinning([
    OneDimBinning(name='reco_energy', is_log=True, num_bins=150,
                  domain=[1e1, 1e8] * ureg.GeV),
    OneDimBinning(name='reco_coszen', is_lin=True, num_bins=100, domain=[-1, 1]),
    OneDimBinning(name='pid', is_lin=True, num_bins=2, domain=[0, 2],
                  bin_names=['cascade', 'track']),
])

SIGMA_E = 0.2    # log10 E resolution
SIGMA_CZ = 0.1   # cos(zenith) resolution


def gaussian_spec(sigma_e=SIGMA_E, sigma_cz=SIGMA_CZ, cz_model='gaussian_coszen'):
    """
    Parametrisation with constant Gaussian resolutions, the same for both
    classes and all channels.

    Parameters
    ----------
    sigma_e : float
        Width in log10(E_reco / E_true).
    sigma_cz : float
        Width in cos(zenith) ('gaussian_coszen') or in degrees ('angular').
    cz_model : str

    Returns
    -------
    dict
    """
    one_class_e = {'log10_energy': [3, 6], 'bias': [0., 0.], 'sigma': [sigma_e, sigma_e]}
    one_class_cz = {'model': cz_model, 'log10_energy': [3, 6], 'sigma': [sigma_cz, sigma_cz]}
    return {
        'format_version': 1,
        'classes': ['cascade', 'track'],
        'channels': {
            'default': {
                'pid': {'log10_energy': [3, 6], 'fraction': [[0.7, 0.3], [0.7, 0.3]]},
                'energy': {'cascade': deepcopy(one_class_e), 'track': deepcopy(one_class_e)},
                'coszen': {'cascade': deepcopy(one_class_cz), 'track': deepcopy(one_class_cz)},
            }
        },
    }


def full_response(spec, **scales):
    """
    Full response tensor for the 'default' channel of `spec`.

    Parameters
    ----------
    spec : dict
    **scales
        Passed to `make_param_response` (energy_scale, energy_res_scale,
        coszen_res_scale).

    Returns
    -------
    np.ndarray
        Shape TRUE_BINNING.shape + RECO_BINNING.shape.
    """
    parts = make_param_response(TRUE_BINNING, RECO_BINNING, spec, ['default'], **scales)['default']
    return expand_factorised_response(parts['pid'], parts['energy'], parts['coszen'])


def log10_reco_edges():
    """log10 of the reco-energy bin edges in GeV."""
    return np.log10(RECO_BINNING['reco_energy'].bin_edges.m_as('GeV'))


def coszen_reco_edges():
    """Reco cos(zenith) bin edges."""
    return RECO_BINNING['reco_coszen'].bin_edges.m_as('dimensionless')


def max_abs_diff(tensor_a, tensor_b):
    """
    Largest difference in probability per true bin, summed over reco bins.

    Parameters
    ----------
    tensor_a, tensor_b : np.ndarray
        Response tensors of the same shape.

    Returns
    -------
    float
        max over true bins of sum over reco bins of |a - b| (total variation
        distance x 2; 0 means identical, 2 means disjoint).
    """
    n_true = TRUE_BINNING.size
    diff = np.abs(tensor_a - tensor_b).reshape(n_true, -1).sum(axis=1)
    return diff.max()


def test_maps_basic(verbosity=Levels.WARN):
    """Identity at nominal values; reflection conserves probability."""
    set_verbosity(verbosity)
    edges = log10_reco_edges()
    n_bins = len(edges) - 1
    assert np.allclose(shift_map(edges, 0.), np.eye(n_bins))
    assert np.allclose(gaussian_smearing_map(edges, 0.), np.eye(n_bins))

    cz_edges = coszen_reco_edges()
    cz_map = gaussian_smearing_map(cz_edges, 0.3, reflect_range=(-1., 1.))
    assert np.allclose(cz_map.sum(axis=1), 1., atol=1e-6)
    # A shift by exactly one bin width moves each bin to the next one
    width = edges[1] - edges[0]
    one_bin = shift_map(edges, width)
    assert np.allclose(one_bin[:-1, 1:], np.eye(n_bins - 1))
    logging.info('<< PASS : test_maps_basic >>')


def test_energy_scale_cross_check(verbosity=Levels.WARN):
    """Shifting a tabulated response = analytic response with the energy scale."""
    set_verbosity(verbosity)
    spec = gaussian_spec()
    nominal = full_response(spec)
    energy_axis = TRUE_BINNING.num_dims + 0
    for scale in [0.8, 1.15]:
        shifted = apply_map_along_axis(nominal, shift_map(log10_reco_edges(), np.log10(scale)),
                                       energy_axis)
        analytic = full_response(spec, energy_scale=scale)
        diff = max_abs_diff(shifted, analytic)
        logging.info('energy scale %.2f: max probability difference %.2e', scale, diff)
        assert diff < 1e-2
    logging.info('<< PASS : test_energy_scale_cross_check >>')


def test_energy_smear_cross_check(verbosity=Levels.WARN):
    """sigma0 plus extra smearing = analytic width sqrt(sigma0^2 + extra^2)."""
    set_verbosity(verbosity)
    spec = gaussian_spec()
    nominal = full_response(spec)
    energy_axis = TRUE_BINNING.num_dims + 0
    for extra in [0.1, 0.3]:
        smeared = apply_map_along_axis(nominal, gaussian_smearing_map(log10_reco_edges(), extra),
                                       energy_axis)
        res_scale = np.sqrt(SIGMA_E**2 + extra**2) / SIGMA_E
        analytic = full_response(spec, energy_res_scale=res_scale)
        diff = max_abs_diff(smeared, analytic)
        logging.info('extra energy smear %.2f: max probability difference %.2e', extra, diff)
        assert diff < 1e-2
    logging.info('<< PASS : test_energy_smear_cross_check >>')


def test_coszen_smear_cross_check(verbosity=Levels.WARN):
    """Same for cos(zenith), including the reflection at -1 and +1."""
    set_verbosity(verbosity)
    spec = gaussian_spec()
    nominal = full_response(spec)
    coszen_axis = TRUE_BINNING.num_dims + 1
    for extra in [0.05, 0.2]:
        smeared = apply_map_along_axis(
            nominal, gaussian_smearing_map(coszen_reco_edges(), extra, reflect_range=(-1., 1.)),
            coszen_axis)
        res_scale = np.sqrt(SIGMA_CZ**2 + extra**2) / SIGMA_CZ
        analytic = full_response(spec, coszen_res_scale=res_scale)
        diff = max_abs_diff(smeared, analytic)
        logging.info('extra coszen smear %.2f: max probability difference %.2e', extra, diff)
        assert diff < 2e-2
    logging.info('<< PASS : test_coszen_smear_cross_check >>')


def test_angular_model_width(verbosity=Levels.WARN):
    """At the horizon, the spread of cos(zenith)_reco equals sigma in radians."""
    set_verbosity(verbosity)
    sigma_deg = 5.
    spec = gaussian_spec(sigma_cz=sigma_deg, cz_model='angular')
    parts = make_param_response(TRUE_BINNING, RECO_BINNING, spec, ['default'])['default']
    coszen_probs = parts['coszen'][0, 0]          # class 0, lowest true energy: (n_cz_true, n_cz_reco)
    cz_true = TRUE_BINNING['true_coszen'].weighted_centers.m
    cz_reco = RECO_BINNING['reco_coszen'].weighted_centers.m
    assert np.allclose(coszen_probs.sum(axis=1), 1., atol=1e-9)
    # True bins next to the horizon (|cz| = 0.125)
    for itrue in np.where(np.abs(cz_true) < 0.2)[0]:
        probs = coszen_probs[itrue]
        mean = np.sum(probs * cz_reco)
        std = np.sqrt(np.sum(probs * (cz_reco - mean)**2))
        expected = np.radians(sigma_deg) * np.sqrt(1. - cz_true[itrue]**2)
        logging.info('cz_true %.3f: std %.4f, expected %.4f', cz_true[itrue], std, expected)
        assert abs(mean - cz_true[itrue]) < 0.01
        assert abs(std / expected - 1.) < 0.08
    logging.info('<< PASS : test_angular_model_width >>')


def run_stage_pipeline(stage_key, stage_cfg, channels):
    """
    Run binned_grid -> one reco stage and return the reco maps.

    Parameters
    ----------
    stage_key : tuple
        ('reco', 'param') or ('reco', 'response_matrix').
    stage_cfg : dict
        Configuration of that stage.
    channels : list of str

    Returns
    -------
    MapSet
    """
    cfg = OrderedDict()
    cfg['pipeline'] = {'name': 'reco_test', 'output_binning': RECO_BINNING,
                       'output_key': ('weights'), 'detector_name': None}
    cfg[('data', 'binned_grid')] = {'calc_mode': TRUE_BINNING, 'apply_mode': TRUE_BINNING,
                                    'output_names': channels}
    cfg[stage_key] = stage_cfg
    return Pipeline(cfg).get_outputs()


def bin_volumes():
    """dE * 2 pi dcosz per true bin, GeV sr, shape of TRUE_BINNING."""
    widths_e = TRUE_BINNING['true_energy'].bin_widths.m_as('GeV')
    widths_cz = TRUE_BINNING['true_coszen'].bin_widths.m
    return widths_e[:, None] * 2 * np.pi * widths_cz[None, :]


def test_stage_closures(verbosity=Levels.WARN):
    """Both stages inside a pipeline equal the direct numpy calculation."""
    set_verbosity(verbosity)
    channels = ['nue_cc', 'numu_cc']
    spec = gaussian_spec(sigma_cz=4., cz_model='angular')
    scales = {'energy_scale': 1.1, 'energy_res_scale': 1.2, 'coszen_res_scale': 0.8}

    # reco.param with non-nominal parameters
    param_cfg = {
        'calc_mode': TRUE_BINNING, 'apply_mode': RECO_BINNING, 'param_spec': spec,
        'params': ParamSet([
            Param(name='reco_energy_scale', value=scales['energy_scale'], **PARAM_DEFAULTS),
            Param(name='reco_energy_res_scale', value=scales['energy_res_scale'], **PARAM_DEFAULTS),
            Param(name='reco_coszen_res_scale', value=scales['coszen_res_scale'], **PARAM_DEFAULTS),
        ]),
    }
    maps = run_stage_pipeline(('reco', 'param'), param_cfg, channels)
    reference = np.tensordot(bin_volumes(), full_response(spec, **scales), axes=2)
    for name in channels:
        assert np.allclose(maps[name].nominal_values, reference, rtol=1e-6, atol=0)
    logging.info('reco.param pipeline closure ok (total %.4g)', reference.sum())

    # reco.response_matrix with systematics on a tabulated response
    nominal = full_response(spec)
    with tempfile.TemporaryDirectory() as tmpdir:
        response_file = os.path.join(tmpdir, 'tabulated.json')
        aeff = {name: np.ones(TRUE_BINNING.shape) for name in channels}
        write_response_file(response_file, TRUE_BINNING, RECO_BINNING, aeff,
                            {name: nominal for name in channels})
        matrix_cfg = {
            'calc_mode': TRUE_BINNING, 'apply_mode': RECO_BINNING,
            'response_file': response_file,
            'params': ParamSet([
                Param(name='reco_energy_scale', value=0.9, **PARAM_DEFAULTS),
                Param(name='reco_energy_extra_smear', value=0.15, **PARAM_DEFAULTS),
                Param(name='reco_coszen_extra_smear', value=0.05, **PARAM_DEFAULTS),
            ]),
        }
        maps = run_stage_pipeline(('reco', 'response_matrix'), matrix_cfg, channels)
    energy_map = (shift_map(log10_reco_edges(), np.log10(0.9))
                  @ gaussian_smearing_map(log10_reco_edges(), 0.15))
    cz_map = gaussian_smearing_map(coszen_reco_edges(), 0.05, reflect_range=(-1., 1.))
    modified = apply_map_along_axis(nominal, energy_map, 2)
    modified = apply_map_along_axis(modified, cz_map, 3)
    reference = np.tensordot(bin_volumes(), modified, axes=2)
    for name in channels:
        assert np.allclose(maps[name].nominal_values, reference, rtol=1e-6, atol=0)
    logging.info('reco.response_matrix pipeline closure ok (total %.4g)', reference.sum())
    logging.info('<< PASS : test_stage_closures >>')


if __name__ == '__main__':
    test_maps_basic(verbosity=Levels.INFO)
    test_energy_scale_cross_check(verbosity=Levels.INFO)
    test_energy_smear_cross_check(verbosity=Levels.INFO)
    test_coszen_smear_cross_check(verbosity=Levels.INFO)
    test_angular_model_width(verbosity=Levels.INFO)
    test_stage_closures(verbosity=Levels.INFO)
