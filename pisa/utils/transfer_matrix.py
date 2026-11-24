"""
Build, read and write Earth-transmission transfer matrices for
distribution-mode pipelines.

The transport of neutrinos through the Earth (CC absorption, NC scattering to
lower energies, tau regeneration with its secondary neutrinos, Glashow
resonance) is linear in the flux. For a fixed Earth model and cross sections
the flux at the detector is therefore a linear map of the flux at the surface,
one map per zenith bin:

    Phi_det[cz, a, E_out] = sum_{b, E_in} T[cz, a, E_out, b, E_in] * Phi_surf[cz, b, E_in]

where a, b run over the six channels nue, numu, nutau, nuebar, numubar,
nutaubar (index = flavour + 3 * is_antineutrino). Fluxes are differential
(per GeV) and evaluated at the true-energy bin centres, so T is
dimensionless.

The matrices are built with nuSQuIDS (`nuSQUIDSAtm`) with oscillations
switched off: oscillations are left to an oscillation stage earlier in the
pipeline, and the matrices then do not depend on the oscillation parameters.
Each column of T is one nuSQuIDS propagation of a unit flux in one channel at
one energy node (all zenith nodes at once). The energy and cos(zenith) nodes
are the bin centres of the pipeline's true binning.

File layout (HDF5)::

    /transfer_matrix        float64 [n_cz, 6, n_E, 6, n_E]
    /computed               bool [6, n_E], which input columns are filled
    /true_energy_edges      float64 [n_E + 1], GeV
    /true_coszen_edges      float64 [n_cz + 1]
    attrs: format_version, energy_name, coszen_name, description, and the
           builder settings (cross_sections, tau_regeneration, ...)

Limitations:

* Flux above the highest true-energy bin is not propagated, so NC feed-down
  from higher energies is missing. The true binning should extend about a
  decade above the energies of interest.
* nuSQuIDS places the detector at the Earth's surface; the overburden of a
  detector at depth is neglected.
"""

import os
import time

import h5py
import numpy as np

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.utils.resources import find_resource

__all__ = [
    'FORMAT_VERSION',
    'CHANNELS',
    'channel_index',
    'bin_centres',
    'build_transfer_matrices',
    'write_transfer_matrix_file',
    'load_transfer_matrix_file',
    'merge_transfer_matrix_files',
    'make_toy_transfer_matrix',
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


FORMAT_VERSION = 1

# Channel order of the matrix axes: flavour + 3 * is_antineutrino, which is
# also nuSQuIDS' (rho, flavour) order with rho = 0 for neutrinos
CHANNELS = ['nue', 'numu', 'nutau', 'nuebar', 'numubar', 'nutaubar']

# Builder settings stored as file attributes, with their defaults
DEFAULT_SETTINGS = {
    'cross_sections': 'csms',
    'tau_regeneration': True,
    'glashow_resonance': True,
    'rel_error': 1e-7,
    'abs_error': 1e-14,
}


def channel_index(flav, nubar):
    """
    Index of a channel along the matrix axes.

    Parameters
    ----------
    flav : int
        0, 1 or 2 for e, mu, tau.
    nubar : int
        +1 for neutrinos, -1 for antineutrinos.

    Returns
    -------
    int
        flav + 3 for antineutrinos, flav for neutrinos.
    """
    if nubar < 0:
        return flav + 3
    return flav


def bin_centres(energy_edges_gev, coszen_edges):
    """
    Nodes used for the matrices: geometric centres of the (log-spaced) energy
    bins and linear centres of the cos(zenith) bins, as PISA's
    `OneDimBinning.weighted_centers` gives them.

    Parameters
    ----------
    energy_edges_gev : np.ndarray
        True-energy bin edges in GeV, shape (n_E + 1,).
    coszen_edges : np.ndarray
        True cos(zenith) bin edges, shape (n_cz + 1,).

    Returns
    -------
    energy_nodes_gev : np.ndarray
        Shape (n_E,), GeV.
    coszen_nodes : np.ndarray
        Shape (n_cz,).
    """
    energy_edges_gev = np.asarray(energy_edges_gev, dtype=float)
    coszen_edges = np.asarray(coszen_edges, dtype=float)
    energy_nodes_gev = np.sqrt(energy_edges_gev[:-1] * energy_edges_gev[1:])
    coszen_nodes = 0.5 * (coszen_edges[:-1] + coszen_edges[1:])
    return energy_nodes_gev, coszen_nodes


def _make_nusquids_atm(energy_nodes_gev, coszen_nodes, settings):
    """
    Set up a nuSQUIDSAtm object for basis propagations.

    Parameters
    ----------
    energy_nodes_gev : np.ndarray
        Energy nodes in GeV, increasing.
    coszen_nodes : np.ndarray
        cos(zenith) nodes, increasing.
    settings : dict
        Keys of `DEFAULT_SETTINGS`.

    Returns
    -------
    nuSQuIDS.nuSQUIDSAtm
    units : nuSQuIDS.Const
    """
    # nuSQuIDS is only needed to build matrices, not to use them
    import nuSQuIDS as nsq  # pylint: disable=import-outside-toplevel

    if 'NUSQUIDS_DATA_PATH' not in os.environ:
        print('Warning: NUSQUIDS_DATA_PATH is not set; nuSQuIDS may not find '
              'its cross-section and Earth-model tables')

    units = nsq.Const()
    if settings['cross_sections'] == 'csms':
        cross_sections = nsq.loadDefaultCrossSections()
    elif settings['cross_sections'] == 'wcg24':
        cross_sections = nsq.loadWCG24CrossSections()
    else:
        raise ValueError(f'Unknown cross_sections "{settings["cross_sections"]}"; '
                         'use "csms" or "wcg24"')

    atm = nsq.nuSQUIDSAtm(coszen_nodes, energy_nodes_gev * units.GeV, 3,
                          nsq.NeutrinoType.both, True, cross_sections)
    # Oscillations are applied by an oscillation stage, not here
    atm.Set_IncludeOscillations(False)
    atm.Set_TauRegeneration(settings['tau_regeneration'])
    atm.Set_GlashowResonance(settings['glashow_resonance'])
    # The absolute tolerance must be small compared with the unit inputs'
    # transmitted fractions (see the module notes)
    atm.Set_rel_error(settings['rel_error'])
    atm.Set_abs_error(settings['abs_error'])
    atm.Set_ProgressBar(False)
    return atm, units


def _read_detector_flux(atm, units, energy_nodes_gev, coszen_nodes):
    """
    Flux at the detector at all nodes after an evolution.

    Returns
    -------
    np.ndarray
        Shape (n_cz, 6, n_E), channel order of `CHANNELS`.
    """
    n_cz = len(coszen_nodes)
    n_e = len(energy_nodes_gev)
    out = np.zeros((n_cz, 6, n_e))
    for icz, one_cz in enumerate(coszen_nodes):
        for ichan in range(6):
            flav = ichan % 3
            rho = ichan // 3
            for ie, one_e in enumerate(energy_nodes_gev):
                out[icz, ichan, ie] = atm.EvalFlavor(flav, one_cz, one_e * units.GeV, rho)
    return out


def build_transfer_matrices(energy_edges_gev, coszen_edges,
                            input_columns=None,
                            verbose=True,
                            **settings):
    """
    Build transfer matrices with nuSQuIDS, one propagation per input column.

    Parameters
    ----------
    energy_edges_gev : np.ndarray
        True-energy bin edges in GeV, log-spaced, shape (n_E + 1,).
    coszen_edges : np.ndarray
        True cos(zenith) bin edges in [-1, 1], shape (n_cz + 1,).
    input_columns : list of (int, int) or None
        (channel index, energy index) pairs to compute; None for all 6 * n_E.
        Used to split a build across jobs.
    verbose : bool
        Print progress.
    **settings
        Overrides of `DEFAULT_SETTINGS`:
        cross_sections : 'csms' (nuSQuIDS default) or 'wcg24'
        tau_regeneration : bool
        glashow_resonance : bool
        rel_error, abs_error : float, nuSQuIDS solver tolerances

    Returns
    -------
    dict
        'matrix' : np.ndarray [n_cz, 6, n_E, 6, n_E], zero in columns not computed
        'computed' : np.ndarray of bool [6, n_E]
        'settings' : dict, the settings used
    """
    # nuSQuIDS is only needed to build matrices, not to use them
    import nuSQuIDS as nsq  # pylint: disable=import-outside-toplevel

    for key in settings:
        if key not in DEFAULT_SETTINGS:
            raise ValueError(f'Unknown setting "{key}"')
    full_settings = dict(DEFAULT_SETTINGS)
    full_settings.update(settings)

    energy_edges_gev = np.asarray(energy_edges_gev, dtype=float)
    log_widths = np.diff(np.log(energy_edges_gev))
    if not np.allclose(log_widths, log_widths[0], rtol=1e-6):
        raise ValueError('The true-energy bins must be log-spaced for nuSQuIDS')
    energy_nodes_gev, coszen_nodes = bin_centres(energy_edges_gev, coszen_edges)
    n_e = len(energy_nodes_gev)
    n_cz = len(coszen_nodes)

    if input_columns is None:
        input_columns = []
        for ichan in range(6):
            for ie in range(n_e):
                input_columns.append((ichan, ie))

    atm, units = _make_nusquids_atm(energy_nodes_gev, coszen_nodes, full_settings)
    matrix = np.zeros((n_cz, 6, n_e, 6, n_e))
    computed = np.zeros((6, n_e), dtype=bool)

    if verbose:
        print('Building', len(input_columns), 'columns on', n_cz, 'cos(zenith) x',
              n_e, 'energy nodes, settings', full_settings)
    t_start = time.time()
    for icol, (ichan, ie) in enumerate(input_columns):
        # Unit flux in one channel at one energy node, all zenith nodes
        state = np.zeros((n_cz, n_e, 2, 3))
        state[:, ie, ichan // 3, ichan % 3] = 1.0
        atm.Set_initial_state(state, nsq.Basis.flavor)
        atm.EvolveState()
        matrix[:, :, :, ichan, ie] = _read_detector_flux(atm, units, energy_nodes_gev,
                                                         coszen_nodes)
        computed[ichan, ie] = True
        if verbose and (icol + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print('  %d / %d columns, %.0f s' % (icol + 1, len(input_columns), elapsed),
                  flush=True)

    if verbose:
        print('Done in %.0f s' % (time.time() - t_start))
    return {'matrix': matrix, 'computed': computed, 'settings': full_settings}


def write_transfer_matrix_file(fname, matrix, energy_edges_gev, coszen_edges,
                               computed=None,
                               settings=None,
                               energy_name='true_energy',
                               coszen_name='true_coszen',
                               description='',
                               overwrite=False):
    """
    Write transfer matrices to an HDF5 file (layout in the module docstring).

    Parameters
    ----------
    fname : str
        Output path (.hdf5 or .h5).
    matrix : np.ndarray
        Shape (n_cz, 6, n_E, 6, n_E).
    energy_edges_gev : np.ndarray
        Shape (n_E + 1,), GeV.
    coszen_edges : np.ndarray
        Shape (n_cz + 1,).
    computed : np.ndarray of bool or None
        Shape (6, n_E); None means all columns are filled.
    settings : dict or None
        Builder settings, stored as attributes.
    energy_name, coszen_name : str
        Names of the binning dimensions the matrices belong to.
    description : str
    overwrite : bool
        Whether an existing file may be replaced.
    """
    if os.path.exists(fname) and not overwrite:
        raise FileExistsError(f'{fname} exists; pass overwrite=True to replace it')
    n_e = len(energy_edges_gev) - 1
    n_cz = len(coszen_edges) - 1
    if matrix.shape != (n_cz, 6, n_e, 6, n_e):
        raise ValueError(f'Matrix shape {matrix.shape}, expected {(n_cz, 6, n_e, 6, n_e)}')
    if computed is None:
        computed = np.ones((6, n_e), dtype=bool)

    with h5py.File(fname, 'w') as fobj:
        fobj.create_dataset('transfer_matrix', data=matrix, compression='gzip')
        fobj.create_dataset('computed', data=computed)
        fobj.create_dataset('true_energy_edges', data=np.asarray(energy_edges_gev, dtype=float))
        fobj.create_dataset('true_coszen_edges', data=np.asarray(coszen_edges, dtype=float))
        fobj.attrs['format_version'] = FORMAT_VERSION
        fobj.attrs['energy_name'] = energy_name
        fobj.attrs['coszen_name'] = coszen_name
        fobj.attrs['description'] = description
        if settings is not None:
            for key, value in settings.items():
                fobj.attrs[key] = value


def load_transfer_matrix_file(source):
    """
    Load a transfer-matrix file.

    Parameters
    ----------
    source : str
        Path or PISA resource of the HDF5 file.

    Returns
    -------
    dict
        'matrix' : np.ndarray [n_cz, 6, n_E, 6, n_E]
        'computed' : np.ndarray of bool [6, n_E]
        'energy_binning' : OneDimBinning (GeV, log)
        'coszen_binning' : OneDimBinning
        'settings' : dict of the builder settings found in the file
        'description' : str
    """
    fname = find_resource(source)
    with h5py.File(fname, 'r') as fobj:
        version = fobj.attrs.get('format_version')
        if version != FORMAT_VERSION:
            raise ValueError(f'Unsupported transfer-matrix format_version {version}, '
                             f'expected {FORMAT_VERSION}')
        matrix = fobj['transfer_matrix'][()]
        computed = fobj['computed'][()].astype(bool)
        energy_edges = fobj['true_energy_edges'][()]
        coszen_edges = fobj['true_coszen_edges'][()]
        energy_name = str(fobj.attrs['energy_name'])
        coszen_name = str(fobj.attrs['coszen_name'])
        description = str(fobj.attrs.get('description', ''))
        settings = {}
        for key in DEFAULT_SETTINGS:
            if key in fobj.attrs:
                value = fobj.attrs[key]
                # h5py gives numbers and bools as numpy scalars, strings as str
                if isinstance(value, np.generic):
                    value = value.item()
                settings[key] = value

    energy_binning = OneDimBinning(name=energy_name, bin_edges=energy_edges * ureg.GeV,
                                   is_log=True)
    coszen_binning = OneDimBinning(name=coszen_name, bin_edges=coszen_edges * ureg.dimensionless,
                                   is_lin=True)
    return {
        'matrix': matrix,
        'computed': computed,
        'energy_binning': energy_binning,
        'coszen_binning': coszen_binning,
        'settings': settings,
        'description': description,
    }


def merge_transfer_matrix_files(file_list, outfile, description='', overwrite=False):
    """
    Combine partial builds (different input columns, same binning and
    settings) into one file.

    Parameters
    ----------
    file_list : list of str
        Partial transfer-matrix files.
    outfile : str
        Output path.
    description : str
    overwrite : bool

    Returns
    -------
    np.ndarray of bool
        Shape (6, n_E): which columns are filled in the merged file. A warning
        is printed if any are missing.
    """
    merged = None
    for ifile, one_file in enumerate(file_list):
        part = load_transfer_matrix_file(one_file)
        if merged is None:
            merged = part
            continue
        # All parts must describe the same calculation
        same_edges = (np.array_equal(part['energy_binning'].bin_edges.m,
                                     merged['energy_binning'].bin_edges.m)
                      and np.array_equal(part['coszen_binning'].bin_edges.m,
                                         merged['coszen_binning'].bin_edges.m))
        if not same_edges or part['settings'] != merged['settings']:
            raise ValueError(f'{one_file} has a different binning or settings than {file_list[0]}')
        if np.any(part['computed'] & merged['computed']):
            raise ValueError(f'{one_file} repeats columns already filled (file {ifile})')
        merged['matrix'][:, :, :, part['computed']] = part['matrix'][:, :, :, part['computed']]
        merged['computed'] = merged['computed'] | part['computed']

    n_missing = np.sum(~merged['computed'])
    if n_missing > 0:
        print('Warning: merged file is missing', n_missing, 'of', merged['computed'].size,
              'columns')
    write_transfer_matrix_file(
        outfile, merged['matrix'],
        merged['energy_binning'].bin_edges.m_as('GeV'),
        merged['coszen_binning'].bin_edges.m,
        computed=merged['computed'], settings=merged['settings'],
        energy_name=merged['energy_binning'].name,
        coszen_name=merged['coszen_binning'].name,
        description=description, overwrite=overwrite,
    )
    return merged['computed']


def make_toy_transfer_matrix(energy_edges_gev, coszen_edges,
                             e_absorb=1e5,       # GeV, absorption scale for a full Earth chord
                             nc_fraction=0.3,    # fraction of interactions that are NC
                             regen_fraction=0.5):  # nu_tau fraction regenerated
    """
    Simple analytic stand-in for nuSQuIDS matrices, for tests. Not physics.

    A neutrino crossing the Earth along chord length L(cz) = -cz (in units of
    the diameter, upgoing only) interacts with probability
    1 - exp(-L * E / e_absorb). A fraction `nc_fraction` of interactions moves
    it, uniformly in log E, to lower energy bins of the same channel. For
    nu_tau, `regen_fraction` of the CC interactions return as nu_tau at lower
    energy and as equal amounts of the secondaries nuebar and numubar
    (antiparticles exchanged for nutaubar).

    Parameters
    ----------
    energy_edges_gev : np.ndarray
        Shape (n_E + 1,), GeV.
    coszen_edges : np.ndarray
        Shape (n_cz + 1,).
    e_absorb : float
        GeV.
    nc_fraction : float
    regen_fraction : float

    Returns
    -------
    np.ndarray
        Shape (n_cz, 6, n_E, 6, n_E).
    """
    energy_nodes_gev, coszen_nodes = bin_centres(energy_edges_gev, coszen_edges)
    n_e = len(energy_nodes_gev)
    n_cz = len(coszen_nodes)
    matrix = np.zeros((n_cz, 6, n_e, 6, n_e))

    for icz, one_cz in enumerate(coszen_nodes):
        chord = max(-one_cz, 0.)
        p_interact = 1. - np.exp(-chord * energy_nodes_gev / e_absorb)
        for ichan in range(6):
            for ie in range(n_e):
                # Survivors stay in place
                matrix[icz, ichan, ie, ichan, ie] = 1. - p_interact[ie]
                if ie == 0:
                    continue
                # Spread moved flux evenly over the lower energy bins
                per_bin = p_interact[ie] / ie
                matrix[icz, ichan, :ie, ichan, ie] += nc_fraction * per_bin
                if ichan % 3 == 2:
                    regen = (1. - nc_fraction) * regen_fraction * per_bin
                    matrix[icz, ichan, :ie, ichan, ie] += regen
                    # Secondaries: nu_tau -> nuebar, numubar; nutaubar -> nue, numu
                    if ichan == 2:
                        secondaries = [3, 4]
                    else:
                        secondaries = [0, 1]
                    for one_sec in secondaries:
                        matrix[icz, one_sec, :ie, ichan, ie] += regen
    return matrix
