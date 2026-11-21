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

__all__ = ['TRUE_BINNING', 'RECO_BINNING', 'CHANNELS', 'test_binned_pipeline']

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


def make_test_config(response_file):
    """
    Pipeline configuration for the closure test.

    Parameters
    ----------
    response_file : str
        Binned response file with TRUE_BINNING and RECO_BINNING.

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


def expected_counts(response_file):
    """
    Reco counts summed over channels, computed directly with numpy.

    Parameters
    ----------
    response_file : str

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

    total = np.zeros(RECO_BINNING.shape)
    for name in CHANNELS:
        counts_true = volume * loaded['aeff'][name] * 1e4 * LIVETIME.m_as('s')
        # Sum over the true axes of counts_true[e, z] * R[e, z, ...]
        total += np.tensordot(counts_true, loaded['response'][name], axes=2)
    return total


def test_binned_pipeline(verbosity=Levels.WARN):
    """Pipeline output equals the direct numpy calculation."""
    set_verbosity(verbosity)

    with tempfile.TemporaryDirectory() as tmpdir:
        response_file = os.path.join(tmpdir, 'toy_response.json')
        aeff, response = make_toy_response(TRUE_BINNING, RECO_BINNING, CHANNELS)
        # Store one channel as a full tensor to exercise both kinds
        parts = response['numu_cc']
        response['numu_cc'] = expand_factorised_response(
            parts['pid'], parts['energy'], parts['coszen'])
        write_response_file(response_file, TRUE_BINNING, RECO_BINNING, aeff,
                            response, description='closure test')

        dmaker = DistributionMaker([make_test_config(response_file)])
        output = dmaker.get_outputs(return_sum=True)[0]
        reference = expected_counts(response_file)

    output_hist = output.nominal_values
    rel_diff = np.abs(output_hist - reference) / np.maximum(reference, 1e-30)
    logging.info('Max relative difference: %.3g', rel_diff.max())
    assert output_hist.shape == reference.shape
    # Guard against a trivial pass with empty maps
    assert reference.sum() > 0 and np.count_nonzero(reference) > reference.size // 2
    logging.info('Total expected counts: %.4g', reference.sum())
    assert np.allclose(output_hist, reference, rtol=1e-5, atol=0), rel_diff.max()
    logging.info('<< PASS : test_binned_pipeline >>')


if __name__ == '__main__':
    test_binned_pipeline(verbosity=Levels.INFO)
