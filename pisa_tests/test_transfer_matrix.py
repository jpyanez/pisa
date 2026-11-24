"""
Tests of the Earth-transmission transfer matrices: the `absorption.transfer_matrix`
stage against numpy (with an analytic toy matrix), file round trip and merging,
and, if nuSQuIDS is installed, matrices built with nuSQuIDS against a direct
nuSQuIDS propagation of the same flux.
"""

from collections import OrderedDict
import os
import tempfile

import numpy as np

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.core.container import ContainerSet
from pisa.core.distribution_maker import DistributionMaker
from pisa.core.param import Param, ParamSet
from pisa.stages.absorption.transfer_matrix import transfer_matrix
from pisa.stages.data.binned_grid import binned_grid, parse_flavint_name
from pisa.utils.binned_response import (
    load_response_file, make_toy_response, write_response_file
)
from pisa.utils.log import Levels, logging, set_verbosity
from pisa.utils.transfer_matrix import (
    bin_centres, channel_index, load_transfer_matrix_file, make_toy_transfer_matrix,
    merge_transfer_matrix_files, write_transfer_matrix_file
)
from pisa_tests.test_binned_pipeline import (
    LIVETIME, PARAM_DEFAULTS, RECO_BINNING, TRUE_BINNING, nominal_response_systematics
)

__all__ = ['test_stage_vs_numpy', 'test_stage_checks', 'test_file_round_trip_and_merge',
           'test_pipeline_with_transfer_matrix', 'test_nusquids_matrix_vs_direct']

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


# All six channels as CC, plus one NC container that shares the numu flux
ALL_CHANNELS = ['nue_cc', 'numu_cc', 'nutau_cc', 'nuebar_cc', 'numubar_cc',
                'nutaubar_cc', 'numu_nc']

ENERGY_DIM = OneDimBinning(name='true_energy', is_log=True, num_bins=12,
                           domain=[1e3, 1e8] * ureg.GeV)
COSZEN_DIM = OneDimBinning(name='true_coszen', is_lin=True, num_bins=5, domain=[-1, 1])


def channel_flux(name, e_centres):
    """
    A different power law for each channel, so that channel mixing is visible.

    Parameters
    ----------
    name : str
        Container name.
    e_centres : np.ndarray
        Energies in GeV.

    Returns
    -------
    np.ndarray
        Flux at `e_centres`, arbitrary units.
    """
    flav, nubar = parse_flavint_name(name)
    ichan = channel_index(flav, nubar)
    return (1. + ichan) * (e_centres / 1e3) ** (-2. - 0.1 * ichan)


def make_grid(binning, channel_names):
    """
    Containers on `binning` from data.binned_grid, weights = initial weights.

    Returns
    -------
    ContainerSet
    """
    grid = binned_grid(output_names=channel_names, calc_mode=binning, apply_mode=binning)
    grid.data = ContainerSet('data')
    grid.setup()
    grid.apply()
    return grid.data


def run_stage(binning, matrix_file, data):
    """Set up and apply absorption.transfer_matrix on `data`; returns the stage."""
    stage = transfer_matrix(transfer_matrix_file=matrix_file, params=ParamSet([]),
                            apply_mode=binning)
    stage.data = data
    stage.setup()
    stage.apply()
    return stage


def write_toy_file(tmpdir):
    """Toy matrix on ENERGY_DIM x COSZEN_DIM; returns (file name, matrix)."""
    energy_edges = ENERGY_DIM.bin_edges.m_as('GeV')
    coszen_edges = COSZEN_DIM.bin_edges.m
    matrix = make_toy_transfer_matrix(energy_edges, coszen_edges)
    fname = os.path.join(tmpdir, 'toy_transfer_matrix.hdf5')
    write_transfer_matrix_file(fname, matrix, energy_edges, coszen_edges,
                               description='analytic toy, not physics')
    return fname, matrix


def test_stage_vs_numpy(verbosity=Levels.WARN):
    """The stage gives initial_weights * (T @ flux) per zenith bin, for both
    orders of the binning dimensions."""
    set_verbosity(verbosity)
    with tempfile.TemporaryDirectory() as tmpdir:
        matrix_file, matrix = write_toy_file(tmpdir)
        e_centres, _ = bin_centres(ENERGY_DIM.bin_edges.m_as('GeV'), COSZEN_DIM.bin_edges.m)
        n_cz = COSZEN_DIM.num_bins

        # Expected detector flux with numpy, [cz, channel, E]
        flux_in = np.zeros((n_cz, 6, len(e_centres)))
        for name in ALL_CHANNELS:
            flav, nubar = parse_flavint_name(name)
            flux_in[:, channel_index(flav, nubar), :] = channel_flux(name, e_centres)[None, :]
        flux_out = np.einsum('zaibj,zbj->zai', matrix, flux_in)

        for dims in [[ENERGY_DIM, COSZEN_DIM], [COSZEN_DIM, ENERGY_DIM]]:
            binning = MultiDimBinning(dims)
            data = make_grid(binning, ALL_CHANNELS)
            for container in data:
                flux_1d = channel_flux(container.name, e_centres)
                if binning.names[0] == 'true_energy':
                    flux_grid = np.repeat(flux_1d[:, None], n_cz, axis=1)
                else:
                    flux_grid = np.repeat(flux_1d[None, :], n_cz, axis=0)
                container['weights'] = container['initial_weights'] * flux_grid.ravel()

            stage = run_stage(binning, matrix_file, data)

            for container in stage.data:
                flav, nubar = parse_flavint_name(container.name)
                expected_grid = flux_out[:, channel_index(flav, nubar), :]
                if binning.names[0] == 'true_energy':
                    expected_grid = expected_grid.T
                expected = container['initial_weights'] * expected_grid.ravel()
                # The toy absorbs everything in some bins, so compare with
                # allclose (exact zeros agree) rather than relative differences
                assert np.allclose(container['weights'], expected, rtol=1e-12, atol=0), \
                    container.name
            logging.info('dims %s: stage agrees with numpy', binning.names)

        # The agreement includes cross-channel terms: nuebar gains from nu_tau
        # regeneration, so the nuebar block alone would not give flux_out
        nuebar_alone = np.einsum('zij,zj->zi', matrix[:, 3, :, 3, :], flux_in[:, 3, :])
        upgoing = np.argmin(COSZEN_DIM.weighted_centers.m)
        nonzero = nuebar_alone[upgoing] > 0
        gain = flux_out[upgoing, 3, nonzero] / nuebar_alone[upgoing, nonzero] - 1.
        logging.info('nuebar gain from regeneration, most upgoing bin: max %.3f', gain.max())
        assert gain.max() > 0.01
    logging.info('<< PASS : test_stage_vs_numpy >>')


def test_stage_checks(verbosity=Levels.WARN):
    """The stage refuses missing channels, inconsistent CC/NC fluxes, and a
    binning different from the file's."""
    set_verbosity(verbosity)
    binning = MultiDimBinning([ENERGY_DIM, COSZEN_DIM])
    with tempfile.TemporaryDirectory() as tmpdir:
        matrix_file, _ = write_toy_file(tmpdir)

        # Missing nutaubar
        try:
            run_stage(binning, matrix_file, make_grid(binning, ALL_CHANNELS[:5]))
        except ValueError as err:
            assert 'missing' in str(err)
        else:
            raise AssertionError('missing channel not detected')

        # numu_nc with a different flux than numu_cc
        data = make_grid(binning, ALL_CHANNELS)
        data['numu_nc']['weights'] = 2. * data['numu_nc']['initial_weights']
        try:
            run_stage(binning, matrix_file, data)
        except ValueError as err:
            assert 'different fluxes' in str(err)
        else:
            raise AssertionError('inconsistent CC/NC fluxes not detected')

        # Different energy binning
        other_binning = MultiDimBinning([
            OneDimBinning(name='true_energy', is_log=True, num_bins=12,
                          domain=[1e3, 1e9] * ureg.GeV),
            COSZEN_DIM,
        ])
        try:
            run_stage(other_binning, matrix_file, make_grid(other_binning, ALL_CHANNELS))
        except ValueError as err:
            assert 'does not match' in str(err)
        else:
            raise AssertionError('binning mismatch not detected')
    logging.info('<< PASS : test_stage_checks >>')


def test_file_round_trip_and_merge(verbosity=Levels.WARN):
    """Writing and loading keeps the matrix; two partial files merge to the full one."""
    set_verbosity(verbosity)
    energy_edges = ENERGY_DIM.bin_edges.m_as('GeV')
    coszen_edges = COSZEN_DIM.bin_edges.m
    matrix = make_toy_transfer_matrix(energy_edges, coszen_edges)
    settings = {'cross_sections': 'csms', 'tau_regeneration': True}
    with tempfile.TemporaryDirectory() as tmpdir:
        # Split the input columns in two halves, as two jobs would
        part_files = []
        first_half = np.zeros((6, ENERGY_DIM.num_bins), dtype=bool)
        first_half[:3] = True
        for ipart, computed in enumerate([first_half, ~first_half]):
            part_matrix = np.where(computed[None, None, None, :, :], matrix, 0.)
            fname = os.path.join(tmpdir, f'part{ipart}.hdf5')
            write_transfer_matrix_file(fname, part_matrix, energy_edges, coszen_edges,
                                       computed=computed, settings=settings)
            part_files.append(fname)

        merged_file = os.path.join(tmpdir, 'merged.hdf5')
        computed = merge_transfer_matrix_files(part_files, merged_file)
        assert computed.all()
        loaded = load_transfer_matrix_file(merged_file)
        assert np.array_equal(loaded['matrix'], matrix)
        assert loaded['settings'] == settings
        assert np.allclose(loaded['energy_binning'].bin_edges.m_as('GeV'), energy_edges)

        # Merging a file with itself repeats columns
        try:
            merge_transfer_matrix_files([part_files[0], part_files[0]],
                                        os.path.join(tmpdir, 'bad.hdf5'))
        except ValueError as err:
            assert 'repeats columns' in str(err)
        else:
            raise AssertionError('repeated columns not detected')
    logging.info('<< PASS : test_file_round_trip_and_merge >>')


def test_pipeline_with_transfer_matrix(verbosity=Levels.WARN):
    """binned_grid -> astrophysical -> transfer_matrix -> aeff.table ->
    response_matrix agrees with the same calculation in numpy."""
    set_verbosity(verbosity)
    energy_edges = TRUE_BINNING['true_energy'].bin_edges.m_as('GeV')
    coszen_edges = TRUE_BINNING['true_coszen'].bin_edges.m
    astro_norm = 1.4
    with tempfile.TemporaryDirectory() as tmpdir:
        matrix = make_toy_transfer_matrix(energy_edges, coszen_edges)
        matrix_file = os.path.join(tmpdir, 'toy_transfer_matrix.hdf5')
        write_transfer_matrix_file(matrix_file, matrix, energy_edges, coszen_edges)
        response_file = os.path.join(tmpdir, 'toy_response.json')
        aeff, response = make_toy_response(TRUE_BINNING, RECO_BINNING, ALL_CHANNELS)
        write_response_file(response_file, TRUE_BINNING, RECO_BINNING, aeff, response)

        cfg = OrderedDict()
        cfg['pipeline'] = {'name': 'transfer_matrix_test', 'output_binning': RECO_BINNING,
                           'output_key': ('weights'), 'detector_name': None}
        cfg[('data', 'binned_grid')] = {'calc_mode': TRUE_BINNING, 'apply_mode': TRUE_BINNING,
                                        'output_names': ALL_CHANNELS}
        cfg[('flux', 'astrophysical')] = {
            'calc_mode': TRUE_BINNING, 'apply_mode': TRUE_BINNING, 'astro_only': True,
            'params': ParamSet([
                Param(name='astro_norm', value=astro_norm, **PARAM_DEFAULTS),
                Param(name='astro_delta', value=0., **PARAM_DEFAULTS),
            ]),
        }
        cfg[('absorption', 'transfer_matrix')] = {
            'calc_mode': None, 'apply_mode': TRUE_BINNING,
            'transfer_matrix_file': matrix_file, 'params': ParamSet([]),
        }
        cfg[('aeff', 'table')] = {
            'calc_mode': None, 'apply_mode': TRUE_BINNING, 'response_file': response_file,
            'params': ParamSet([
                Param(name='livetime', value=LIVETIME, **PARAM_DEFAULTS),
                Param(name='aeff_scale', value=1.0, **PARAM_DEFAULTS),
            ]),
        }
        cfg[('reco', 'response_matrix')] = {
            'calc_mode': TRUE_BINNING, 'apply_mode': RECO_BINNING,
            'response_file': response_file, 'params': nominal_response_systematics(),
        }
        output = DistributionMaker([cfg]).get_outputs(return_sum=True)[0].nominal_values

        # Same thing with numpy: astrophysical flux, the same in all channels
        loaded = load_response_file(response_file)
        e_centres, _ = bin_centres(energy_edges, coszen_edges)
        n_cz = len(coszen_edges) - 1
        astro_1d = astro_norm * 0.787e-14 * (e_centres / 1e5) ** -2.5
        flux_in = np.repeat(np.repeat(astro_1d[None, None, :], 6, axis=1), n_cz, axis=0)
        flux_out = np.einsum('zaibj,zbj->zai', matrix, flux_in)
        widths_e = TRUE_BINNING['true_energy'].bin_widths.m_as('GeV')
        widths_cz = TRUE_BINNING['true_coszen'].bin_widths.m
        volume = widths_e[:, None] * 2 * np.pi * widths_cz[None, :]
        reference = np.zeros(RECO_BINNING.shape)
        for name in ALL_CHANNELS:
            flav, nubar = parse_flavint_name(name)
            flux_e_cz = flux_out[:, channel_index(flav, nubar), :].T
            counts_true = volume * flux_e_cz * loaded['aeff'][name] * LIVETIME.m_as('s')
            reference += np.tensordot(counts_true, loaded['response'][name], axes=2)

    rel = np.abs(output - reference) / np.maximum(reference, 1e-300)
    logging.info('pipeline with transfer matrix: total %.4g, max relative difference %.2e',
                 reference.sum(), rel.max())
    assert reference.sum() > 0
    assert np.allclose(output, reference, rtol=1e-10, atol=0), rel.max()
    logging.info('<< PASS : test_pipeline_with_transfer_matrix >>')


def test_nusquids_matrix_vs_direct(verbosity=Levels.WARN):
    """Matrices built with nuSQuIDS reproduce a direct nuSQuIDS propagation of
    a flux in all six channels, contain no flux moving up in energy, are the
    identity for downgoing bins, and a split build merges to the full one.
    Skipped if nuSQuIDS is not installed."""
    set_verbosity(verbosity)
    try:
        import nuSQuIDS as nsq  # pylint: disable=import-outside-toplevel
    except ImportError:
        logging.warning('nuSQuIDS not installed; skipping test_nusquids_matrix_vs_direct')
        return
    # pylint: disable=import-outside-toplevel, protected-access
    from pisa.utils.transfer_matrix import (
        DEFAULT_SETTINGS, _make_nusquids_atm, _read_detector_flux, build_transfer_matrices
    )

    energy_edges = np.logspace(3, 8, 11)          # GeV
    coszen_edges = np.array([-1., -0.6, -0.2, 0.2, 1.])
    e_nodes, cz_nodes = bin_centres(energy_edges, coszen_edges)
    n_e = len(e_nodes)
    built = build_transfer_matrices(energy_edges, coszen_edges, verbose=False)
    matrix = built['matrix']

    # Direct propagation of order-one fluxes (the solver tolerance is absolute)
    flux_in = np.zeros((len(cz_nodes), 6, n_e))
    for ichan in range(6):
        flux_in[:, ichan, :] = (1. + ichan) * (e_nodes / 1e3) ** (-1.5 - 0.1 * ichan)
    atm, units = _make_nusquids_atm(e_nodes, cz_nodes, DEFAULT_SETTINGS)
    state = np.zeros((len(cz_nodes), n_e, 2, 3))
    for ichan in range(6):
        state[:, :, ichan // 3, ichan % 3] = flux_in[:, ichan, :]
    atm.Set_initial_state(state, nsq.Basis.flavor)
    atm.EvolveState()
    direct = _read_detector_flux(atm, units, e_nodes, cz_nodes)
    via_matrix = np.einsum('zaibj,zbj->zai', matrix, flux_in)

    # Compare where the transmitted flux is not negligible
    significant = direct > 1e-6 * flux_in
    rel = np.abs(via_matrix - direct)[significant] / direct[significant]
    logging.info('nuSQuIDS matrix vs direct: max relative difference %.2e (median %.2e) '
                 'over %d points', rel.max(), np.median(rel), rel.size)
    assert rel.max() < 1e-3, rel.max()

    # No flux moves to higher energy: T[..., E_out, b, E_in] = 0 for E_out > E_in
    for ie_out in range(n_e):
        assert np.all(np.abs(matrix[:, :, ie_out, :, :ie_out]) < 1e-10)

    # Downgoing bins (cz > 0) see no Earth: identity
    downgoing = cz_nodes > 0
    identity = np.eye(6 * n_e).reshape(6, n_e, 6, n_e)
    assert np.allclose(matrix[downgoing], identity[None], atol=1e-4)

    # Two half builds give the same columns as the full build
    first_half = []
    second_half = []
    for ichan in range(6):
        for ie in range(n_e):
            if (ichan * n_e + ie) % 2 == 0:
                first_half.append((ichan, ie))
            else:
                second_half.append((ichan, ie))
    with tempfile.TemporaryDirectory() as tmpdir:
        part_files = []
        for ipart, columns in enumerate([first_half, second_half]):
            part = build_transfer_matrices(energy_edges, coszen_edges, input_columns=columns,
                                           verbose=False)
            fname = os.path.join(tmpdir, f'part{ipart}.hdf5')
            write_transfer_matrix_file(fname, part['matrix'], energy_edges, coszen_edges,
                                       computed=part['computed'], settings=part['settings'])
            part_files.append(fname)
        merged_file = os.path.join(tmpdir, 'merged.hdf5')
        merge_transfer_matrix_files(part_files, merged_file)
        merged = load_transfer_matrix_file(merged_file)
    assert np.allclose(merged['matrix'], matrix, rtol=1e-6, atol=1e-12)
    logging.info('<< PASS : test_nusquids_matrix_vs_direct >>')


if __name__ == '__main__':
    test_stage_vs_numpy(verbosity=Levels.INFO)
    test_stage_checks(verbosity=Levels.INFO)
    test_file_round_trip_and_merge(verbosity=Levels.INFO)
    test_pipeline_with_transfer_matrix(verbosity=Levels.INFO)
    test_nusquids_matrix_vs_direct(verbosity=Levels.INFO)
