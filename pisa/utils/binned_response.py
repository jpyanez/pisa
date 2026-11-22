"""
Read, check and write binned detector-response files for distribution-mode
pipelines.

A response file holds, for each neutrino channel, an effective area on a true
binning and a response P(reco bin | true bin) that maps the true binning onto
a reco binning. Files are JSON (optionally bz2-compressed), read and written
with `pisa.utils.fileio`. Binnings are stored with the keys of
`OneDimBinning.serializable_state`.

File layout::

    {
      "format_version": 1,
      "description": str,
      "provenance": {...},
      "true_binning": [<OneDimBinning state>, ...],    # e.g. true_energy, true_coszen
      "reco_binning": [<OneDimBinning state>, ...],    # e.g. reco_energy, reco_coszen, pid
      "channels": {
        "<container name>": {
          "aeff": {"units": "m**2", "values": [n_true_E][n_true_cz]},
          "response": {"kind": "full", "values": [true dims...][reco dims...]}
                   or {"kind": "factorised",
                       "pid":    [nE][ncz][npid],
                       "energy": [npid][nE][nE_reco],
                       "coszen": [npid][nE][ncz][ncz_reco]}
        }, ...
      }
    }

Conventions:

* The effective area is after event selection, for all selected events
  whatever their reco values.
* The response gives P(reco bin | true bin, selected). For each true bin the
  sum over reco bins is at most 1; the rest fell outside the reco binning.
* Arrays are in C order, so that flattening them matches the flattening of a
  `MultiDimBinning`: `counts_true.ravel() @ R.reshape(n_true, n_reco)` gives
  the reco counts.
* The factorised form assumes that the class (pid) depends on true energy and
  cos(zenith), the reco energy on true energy and class, and the reco
  cos(zenith) on true energy, true cos(zenith) and class. It is only defined
  for true binnings (energy, coszen) and reco binnings (energy, coszen, pid),
  in that order.
"""

import numpy as np
from scipy.stats import norm

from pisa import ureg
from pisa.core.binning import MultiDimBinning, OneDimBinning
from pisa.utils.fileio import from_file, to_file

__all__ = [
    'FORMAT_VERSION',
    'load_response_file',
    'write_response_file',
    'expand_factorised_response',
    'check_response',
    'make_toy_response',
    'apply_response_matrix',
    'shift_map',
    'gaussian_smearing_map',
    'apply_map_along_axis',
    'make_param_response',
    'load_param_spec',
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

# Allowed excess of a response column sum above 1, from rounding
SUM_TOLERANCE = 1e-6


def _binning_from_states(states):
    """
    Build a MultiDimBinning from a list of OneDimBinning state dicts.

    Parameters
    ----------
    states : list of dict
        Each with at least 'name', 'bin_edges' and 'units'; optional keys of
        `OneDimBinning.serializable_state` ('is_log', 'is_lin', 'bin_names',
        'tex') are passed on.

    Returns
    -------
    MultiDimBinning
    """
    dims = []
    for one_state in states:
        units = one_state.get('units', 'dimensionless')
        kwargs = {
            'name': one_state['name'],
            'bin_edges': np.asarray(one_state['bin_edges'], dtype=float) * ureg(units),
        }
        for key in ['is_log', 'is_lin', 'bin_names', 'tex']:
            if one_state.get(key) is not None:
                kwargs[key] = one_state[key]
        dims.append(OneDimBinning(**kwargs))
    return MultiDimBinning(dims)


def expand_factorised_response(pid, energy, coszen):
    """
    Build the full response tensor from its factorised parts.

    Parameters
    ----------
    pid : np.ndarray
        P(class | E_true, cz_true), shape (nE, ncz, npid).
    energy : np.ndarray
        P(E_reco | E_true, class), shape (npid, nE, nE_reco).
    coszen : np.ndarray
        P(cz_reco | E_true, cz_true, class), shape (npid, nE, ncz, ncz_reco).

    Returns
    -------
    np.ndarray
        P(E_reco, cz_reco, class | E_true, cz_true),
        shape (nE, ncz, nE_reco, ncz_reco, npid).
    """
    pid = np.asarray(pid, dtype=float)
    energy = np.asarray(energy, dtype=float)
    coszen = np.asarray(coszen, dtype=float)

    n_e, n_cz, n_pid = pid.shape
    if energy.shape[:2] != (n_pid, n_e):
        raise ValueError(f'energy part has shape {energy.shape}, expected ({n_pid}, {n_e}, nE_reco)')
    if coszen.shape[:3] != (n_pid, n_e, n_cz):
        raise ValueError(f'coszen part has shape {coszen.shape}, expected ({n_pid}, {n_e}, {n_cz}, ncz_reco)')

    # Indices: e = true E, z = true cz, p = class, r = reco E, s = reco cz
    full = np.einsum('ezp,per,pezs->ezrsp', pid, energy, coszen)
    return full


def check_response(response, n_true, label=''):
    """
    Check that a response is a set of probabilities per true bin.

    Parameters
    ----------
    response : np.ndarray
        Response tensor whose first dimensions are the true binning.
    n_true : int
        Number of true bins (product of the true dimensions).
    label : str
        Name used in error messages.

    Raises
    ------
    ValueError
        If any entry is negative or not finite, or if any true bin sums to
        more than 1.
    """
    flat = response.reshape(n_true, -1)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f'{label}: response has non-finite entries')
    if np.any(flat < 0):
        raise ValueError(f'{label}: response has negative entries')
    column_sums = flat.sum(axis=1)
    if np.any(column_sums > 1 + SUM_TOLERANCE):
        raise ValueError(
            f'{label}: response sums to more than 1 for some true bins '
            f'(max {column_sums.max():.6g})'
        )


def load_response_file(source):
    """
    Load and check a binned response file.

    Parameters
    ----------
    source : str or dict
        Path/resource of a JSON (or .json.bz2) response file, or the already
        loaded dict.

    Returns
    -------
    dict
        Keys:
        'true_binning' : MultiDimBinning
        'reco_binning' : MultiDimBinning
        'aeff' : dict, container name -> np.ndarray in m^2, shape of the true binning
        'response' : dict, container name -> np.ndarray, shape
            (true binning shape + reco binning shape)
        'description', 'provenance' : as in the file
    """
    content = from_file(source) if isinstance(source, str) else source

    version = content.get('format_version')
    if version != FORMAT_VERSION:
        raise ValueError(f'Unsupported response format_version {version}, expected {FORMAT_VERSION}')

    true_binning = _binning_from_states(content['true_binning'])
    reco_binning = _binning_from_states(content['reco_binning'])

    out = {
        'true_binning': true_binning,
        'reco_binning': reco_binning,
        'aeff': {},
        'response': {},
        'description': content.get('description', ''),
        'provenance': content.get('provenance', {}),
    }

    for name, channel in content['channels'].items():
        # Effective area, converted to m^2
        aeff_units = ureg(channel['aeff'].get('units', 'm**2'))
        aeff = np.asarray(channel['aeff']['values'], dtype=float)
        aeff = aeff * aeff_units.to('m**2').magnitude
        if aeff.shape != true_binning.shape:
            raise ValueError(f'{name}: aeff shape {aeff.shape} does not match true binning {true_binning.shape}')
        if np.any(aeff < 0) or not np.all(np.isfinite(aeff)):
            raise ValueError(f'{name}: aeff has negative or non-finite entries')
        out['aeff'][name] = aeff

        # Response, expanded to the full tensor if needed
        resp = channel['response']
        if resp['kind'] == 'full':
            full = np.asarray(resp['values'], dtype=float)
        elif resp['kind'] == 'factorised':
            if true_binning.num_dims != 2 or reco_binning.num_dims != 3:
                raise ValueError('factorised responses need 2 true and 3 reco dimensions')
            full = expand_factorised_response(resp['pid'], resp['energy'], resp['coszen'])
        else:
            raise ValueError(f'{name}: unknown response kind "{resp["kind"]}"')

        expected_shape = tuple(true_binning.shape) + tuple(reco_binning.shape)
        if full.shape != expected_shape:
            raise ValueError(f'{name}: response shape {full.shape}, expected {expected_shape}')
        check_response(full, true_binning.size, label=name)
        out['response'][name] = full

    return out


def write_response_file(fname, true_binning, reco_binning, aeff, response,
                        description='', provenance=None, overwrite=False):
    """
    Write a binned response file.

    Parameters
    ----------
    fname : str
        Output path ending in .json or .json.bz2.
    true_binning, reco_binning : MultiDimBinning
    aeff : dict
        Container name -> effective area in m^2, shape of `true_binning`.
    response : dict
        Container name -> either a full tensor (np.ndarray, shape true + reco)
        or a dict {'pid': ..., 'energy': ..., 'coszen': ...} with the
        factorised parts (see `expand_factorised_response`).
    description : str
    provenance : dict or None
        Where the response comes from (simulation, builder, date, ...).
    overwrite : bool
        Whether an existing file may be replaced.
    """
    channels = {}
    for name in aeff:
        one_resp = response[name]
        if isinstance(one_resp, dict):
            resp_entry = {'kind': 'factorised'}
            for key in ['pid', 'energy', 'coszen']:
                resp_entry[key] = np.asarray(one_resp[key], dtype=float)
        else:
            resp_entry = {'kind': 'full', 'values': np.asarray(one_resp, dtype=float)}
        channels[name] = {
            'aeff': {'units': 'm**2', 'values': np.asarray(aeff[name], dtype=float)},
            'response': resp_entry,
        }

    content = {
        'format_version': FORMAT_VERSION,
        'description': description,
        'provenance': provenance if provenance is not None else {},
        'true_binning': [dict(dim.serializable_state) for dim in true_binning],
        'reco_binning': [dict(dim.serializable_state) for dim in reco_binning],
        'channels': channels,
    }
    to_file(content, fname, overwrite=overwrite)


def _gaussian_bin_probabilities(centres, sigma, reco_edges):
    """
    Probability that a Gaussian around each centre falls in each reco bin.

    Parameters
    ----------
    centres : np.ndarray
        Means, shape (n,).
    sigma : float or np.ndarray
        Width(s), scalar or shape (n,).
    reco_edges : np.ndarray
        Reco bin edges in the same variable, shape (m + 1,).

    Returns
    -------
    np.ndarray
        Shape (n, m). Rows sum to at most 1 (the tails outside the edges are lost).
    """
    sigma = np.broadcast_to(sigma, centres.shape)
    cdf = norm.cdf((reco_edges[None, :] - centres[:, None]) / sigma[:, None])
    return np.diff(cdf, axis=1)


def make_toy_response(true_binning, reco_binning, channel_names,
                      log10e_sigma=(0.25, 0.4),   # cascade, track; in log10(E)
                      coszen_sigma=(0.15, 0.02),  # cascade, track
                      track_fraction_numu_cc=0.8,
                      track_fraction_other=0.1,
                      aeff_norm=1.0,              # m^2 at 1 TeV
                      aeff_index=0.5):
    """
    Make a simple analytic detector response for tests and examples.

    Gaussian smearing in log10(E) and in cos(zenith), a fixed track
    probability per channel, and a power-law effective area. Two reco classes
    are assumed, in the order (cascade, track).

    Parameters
    ----------
    true_binning : MultiDimBinning
        Dimensions (true_energy [GeV, log], true_coszen).
    reco_binning : MultiDimBinning
        Dimensions (reco_energy [GeV, log], reco_coszen, pid with 2 bins).
    channel_names : sequence of str
        Container names, e.g. 'nue_cc'.
    log10e_sigma : tuple of float
        Energy resolution in log10(E) for cascades and tracks.
    coszen_sigma : tuple of float
        cos(zenith) resolution for cascades and tracks.
    track_fraction_numu_cc : float
        Probability that a numu/numubar CC event is classified as a track.
    track_fraction_other : float
        Same for all other channels.
    aeff_norm : float
        Effective area at 1 TeV, m^2.
    aeff_index : float
        A_eff proportional to (E / 1 TeV)**aeff_index.

    Returns
    -------
    aeff : dict
        Container name -> effective area, m^2, shape of `true_binning`.
    response : dict
        Container name -> factorised response dict ('pid', 'energy', 'coszen').
    """
    e_true = true_binning.dims[0].weighted_centers.m_as('GeV')
    cz_true = true_binning.dims[1].weighted_centers.m_as('dimensionless')
    log10e_reco_edges = np.log10(reco_binning.dims[0].bin_edges.m_as('GeV'))
    cz_reco_edges = reco_binning.dims[1].bin_edges.m_as('dimensionless')
    n_e, n_cz = len(e_true), len(cz_true)
    if reco_binning.dims[2].num_bins != 2:
        raise ValueError('make_toy_response assumes 2 reco classes (cascade, track)')

    aeff_one = aeff_norm * (e_true / 1e3) ** aeff_index
    aeff_2d = np.repeat(aeff_one[:, None], n_cz, axis=1)

    # Energy and direction smearing, per class
    energy = []
    coszen = []
    for iclass in range(2):
        energy.append(_gaussian_bin_probabilities(
            np.log10(e_true), log10e_sigma[iclass], log10e_reco_edges))
        one_cz = _gaussian_bin_probabilities(cz_true, coszen_sigma[iclass], cz_reco_edges)
        coszen.append(np.repeat(one_cz[None, :, :], n_e, axis=0))
    energy = np.array(energy)
    coszen = np.array(coszen)

    aeff = {}
    response = {}
    for name in channel_names:
        is_numu_cc = name.split('_')[0] in ('numu', 'numubar') and name.endswith('_cc')
        f_track = track_fraction_numu_cc if is_numu_cc else track_fraction_other
        pid = np.zeros((n_e, n_cz, 2))
        pid[:, :, 0] = 1. - f_track
        pid[:, :, 1] = f_track
        aeff[name] = aeff_2d.copy()
        response[name] = {'pid': pid, 'energy': energy, 'coszen': coszen}
    return aeff, response


# ---------------------------------------------------------------------------
# Applying a response, and linear maps that modify it (response systematics)
# ---------------------------------------------------------------------------

def apply_response_matrix(container, matrix, calc_mode, apply_mode):
    """
    Map the weights of a container from the true onto the reco binning.

    Parameters
    ----------
    container : pisa.core.container.Container
        Holds 'weights' (expected counts per true bin) on `calc_mode`.
    matrix : np.ndarray
        P(reco bin | true bin), shape (calc_mode.size, apply_mode.size).
    calc_mode, apply_mode : MultiDimBinning
        True and reco binnings.
    """
    container.representation = calc_mode
    counts_reco = container['weights'] @ matrix
    # Switch the container to the reco binning and store the result
    container.representation = apply_mode
    container['weights'] = counts_reco


def shift_map(edges, shift):
    """
    Bin-to-bin map for a constant shift of the variable.

    The content of each bin is taken as uniform within the bin, so the map is
    exact for histograms with flat densities inside bins.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges, shape (n + 1,), in the variable that is shifted (e.g.
        log10 of the reco energy).
    shift : float
        Amount added to the variable.

    Returns
    -------
    np.ndarray
        M[i, j] = fraction of bin i that ends up in bin j, shape (n, n).
        Rows sum to less than 1 when content moves outside the edges.
    """
    lo = edges[:-1][:, None] + shift
    hi = edges[1:][:, None] + shift
    overlap = np.minimum(hi, edges[1:][None, :]) - np.maximum(lo, edges[:-1][None, :])
    overlap = np.clip(overlap, 0., None)
    return overlap / (edges[1:] - edges[:-1])[:, None]


def _gauss_cdf_antiderivative(t, sigma):
    """
    Antiderivative in t of Phi(t / sigma), the Gaussian CDF.

    Parameters
    ----------
    t : np.ndarray
    sigma : float, > 0

    Returns
    -------
    np.ndarray
        t * Phi(t / sigma) + sigma * phi(t / sigma).
    """
    u = t / sigma
    return t * norm.cdf(u) + sigma * norm.pdf(u)


def gaussian_smearing_map(edges, sigma, reflect_range=None):
    """
    Bin-to-bin map for an extra Gaussian smearing of the variable.

    A value uniform within bin i is smeared by N(0, sigma); M[i, j] is the
    probability that it ends up in bin j, integrated exactly.

    Parameters
    ----------
    edges : np.ndarray
        Bin edges, shape (n + 1,).
    sigma : float
        Width of the extra smearing, same units as `edges`. 0 gives the
        identity.
    reflect_range : tuple of float or None
        Physical limits (low, high) of the variable, e.g. (-1, 1) for
        cos(zenith). Probability smeared beyond a limit is reflected back
        (once), so it is conserved. If None, it is lost.

    Returns
    -------
    np.ndarray
        Shape (n, n).
    """
    n_bins = len(edges) - 1
    if sigma <= 0:
        return np.eye(n_bins)
    a_i = edges[:-1][:, None]
    b_i = edges[1:][:, None]
    width_i = b_i - a_i

    def prob_into(a_j, b_j):
        # P(x + N(0, sigma) in [a_j, b_j] | x uniform in [a_i, b_i])
        g = _gauss_cdf_antiderivative
        num = (g(b_j - a_i, sigma) - g(b_j - b_i, sigma)
               - g(a_j - a_i, sigma) + g(a_j - b_i, sigma))
        return num / width_i

    a_j = edges[:-1][None, :]
    b_j = edges[1:][None, :]
    out = prob_into(a_j, b_j)
    if reflect_range is not None:
        low, high = reflect_range
        # Values smeared below `low` land at 2*low - x, above `high` at 2*high - x
        out = out + prob_into(2 * low - b_j, 2 * low - a_j)
        out = out + prob_into(2 * high - b_j, 2 * high - a_j)
    return np.clip(out, 0., None)


def apply_map_along_axis(tensor, bin_map, axis):
    """
    Apply a bin-to-bin map along one axis of a response tensor.

    Parameters
    ----------
    tensor : np.ndarray
        Response tensor.
    bin_map : np.ndarray
        Shape (n, n) with n = tensor.shape[axis]; entry [i, j] moves content
        from bin i to bin j.
    axis : int

    Returns
    -------
    np.ndarray
        Same shape as `tensor`.
    """
    moved = np.tensordot(tensor, bin_map, axes=([axis], [0]))
    # tensordot puts the new axis last; move it back into place
    return np.moveaxis(moved, -1, axis)


# ---------------------------------------------------------------------------
# Parametrised (analytic) responses
# ---------------------------------------------------------------------------

def load_param_spec(source):
    """
    Load a parametrised-response specification.

    Parameters
    ----------
    source : str or dict
        Path/resource of a JSON file, or the loaded dict.

    Returns
    -------
    dict
        The specification, with 'classes' and 'channels' checked.

    Notes
    -----
    Format::

        {
          "format_version": 1,
          "description": str,
          "classes": ["cascade", "track"],          # order of the pid reco bins
          "channels": {
            "<container name or 'default'>": {
              "pid": {"log10_energy": [...],          # log10(E_true / GeV)
                      "fraction": [[f_class0, f_class1, ...], ...]},
              "energy": {"<class>": {"log10_energy": [...],
                                      "bias": [...],   # mean of log10(E_reco/E_true)
                                      "sigma": [...]}},# width of log10(E_reco/E_true)
              "coszen": {"<class>": {"model": "gaussian_coszen" or "angular",
                                      "log10_energy": [...],
                                      "sigma": [...]}} # in cos(zenith), or degrees
            }, ...
          }
        }

    Values between the tabulated energies are interpolated linearly in
    log10(E_true) and held constant outside them. A channel missing from
    'channels' uses 'default' if present.
    """
    spec = from_file(source) if isinstance(source, str) else source
    if spec.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'Unsupported format_version {spec.get("format_version")}')
    classes = list(spec['classes'])
    for name, channel in spec['channels'].items():
        fractions = np.asarray(channel['pid']['fraction'], dtype=float)
        if fractions.shape != (len(channel['pid']['log10_energy']), len(classes)):
            raise ValueError(f'{name}: pid fractions must have shape (n_energies, n_classes)')
        if np.any(fractions < 0) or np.any(np.abs(fractions.sum(axis=1) - 1) > 1e-6):
            raise ValueError(f'{name}: pid fractions must be >= 0 and sum to 1')
        for one_class in classes:
            if one_class not in channel['energy'] or one_class not in channel['coszen']:
                raise ValueError(f'{name}: energy and coszen needed for class "{one_class}"')
            model = channel['coszen'][one_class].get('model', 'gaussian_coszen')
            if model not in ('gaussian_coszen', 'angular'):
                raise ValueError(f'{name}: unknown coszen model "{model}"')
    return spec


def _interp_log10e(log10e, table_log10e, table_values):
    """Linear interpolation in log10(E), constant beyond the table."""
    return np.interp(log10e, np.asarray(table_log10e, dtype=float),
                     np.asarray(table_values, dtype=float))


def _angular_coszen_probabilities(cz_true, sigma_deg, cz_edges,
                                  n_psi=48, n_phi=48):
    """
    P(cos(zenith)_reco in each bin | cos(zenith)_true) for a 2D Gaussian
    angular error of width sigma (Rayleigh-distributed opening angle).

    Parameters
    ----------
    cz_true : np.ndarray
        True cos(zenith) values, shape (n,).
    sigma_deg : np.ndarray
        Angular resolution (2D Gaussian sigma) per entry, degrees, shape (n,).
    cz_edges : np.ndarray
        Reco cos(zenith) bin edges, shape (m + 1,).
    n_psi, n_phi : int
        Quadrature points in opening angle (Rayleigh quantiles) and azimuth.

    Returns
    -------
    np.ndarray
        Shape (n, m). Rows sum to 1 within the [-1, 1] range of cz_edges.
    """
    # Opening angles at the midpoints of n_psi equal-probability slices of a
    # Rayleigh distribution, and n_phi uniformly spaced azimuths
    quantiles = (np.arange(n_psi) + 0.5) / n_psi
    psi_unit = np.sqrt(-2. * np.log(1. - quantiles))   # Rayleigh with sigma = 1
    phi = (np.arange(n_phi) + 0.5) / n_phi * 2. * np.pi

    out = np.zeros((len(cz_true), len(cz_edges) - 1))
    for ientry, (one_cz, one_sigma) in enumerate(zip(cz_true, sigma_deg)):
        psi = np.minimum(psi_unit * np.radians(one_sigma), np.pi)
        sin_theta = np.sqrt(max(0., 1. - one_cz**2))
        cz_reco = (one_cz * np.cos(psi)[:, None]
                   + sin_theta * np.sin(psi)[:, None] * np.cos(phi)[None, :])
        cz_reco = np.clip(cz_reco, -1., 1.)
        counts, _ = np.histogram(cz_reco.ravel(), bins=cz_edges)
        out[ientry] = counts / cz_reco.size
    return out


def make_param_response(true_binning, reco_binning, spec, channel_names,
                        energy_scale=1.0,      # multiplies E_reco
                        energy_res_scale=1.0,  # multiplies the log10 E resolution
                        coszen_res_scale=1.0): # multiplies the coszen/angular resolution
    """
    Build factorised responses from a parametrised specification.

    Parameters
    ----------
    true_binning : MultiDimBinning
        (true_energy [GeV], true_coszen).
    reco_binning : MultiDimBinning
        (reco_energy [GeV], reco_coszen, pid), with pid bins in the order of
        spec['classes'].
    spec : dict
        From `load_param_spec`.
    channel_names : sequence of str
    energy_scale : float
        Reco energies are multiplied by this factor (a shift of
        log10(energy_scale) in log10 E_reco).
    energy_res_scale : float
        Scales the width of log10(E_reco / E_true).
    coszen_res_scale : float
        Scales the cos(zenith) or angular width.

    Returns
    -------
    dict
        Container name -> {'pid', 'energy', 'coszen'} (see
        `expand_factorised_response` for shapes).
    """
    classes = list(spec['classes'])
    if reco_binning.dims[2].num_bins != len(classes):
        raise ValueError('pid binning must have one bin per class in the spec')

    e_true = true_binning.dims[0].weighted_centers.m_as('GeV')
    log10e_true = np.log10(e_true)
    cz_true = true_binning.dims[1].weighted_centers.m_as('dimensionless')
    log10e_reco_edges = np.log10(reco_binning.dims[0].bin_edges.m_as('GeV'))
    cz_reco_edges = reco_binning.dims[1].bin_edges.m_as('dimensionless')
    n_e, n_cz = len(e_true), len(cz_true)

    responses = {}
    for name in channel_names:
        if name in spec['channels']:
            channel = spec['channels'][name]
        elif 'default' in spec['channels']:
            channel = spec['channels']['default']
        else:
            raise KeyError(f'No parametrisation for "{name}" and no "default"')

        # Class fractions versus true energy, same for all directions
        pid = np.zeros((n_e, n_cz, len(classes)))
        fractions = np.asarray(channel['pid']['fraction'], dtype=float)
        for iclass in range(len(classes)):
            one_frac = _interp_log10e(log10e_true, channel['pid']['log10_energy'],
                                      fractions[:, iclass])
            pid[:, :, iclass] = one_frac[:, None]

        energy = np.zeros((len(classes), n_e, len(log10e_reco_edges) - 1))
        coszen = np.zeros((len(classes), n_e, n_cz, len(cz_reco_edges) - 1))
        for iclass, one_class in enumerate(classes):
            # Energy: Gaussian in log10(E_reco / E_true)
            e_par = channel['energy'][one_class]
            bias = _interp_log10e(log10e_true, e_par['log10_energy'], e_par['bias'])
            sigma = _interp_log10e(log10e_true, e_par['log10_energy'], e_par['sigma'])
            mean = log10e_true + bias + np.log10(energy_scale)
            energy[iclass] = _gaussian_bin_probabilities(
                mean, sigma * energy_res_scale, log10e_reco_edges)

            # Direction
            cz_par = channel['coszen'][one_class]
            cz_sigma = _interp_log10e(log10e_true, cz_par['log10_energy'],
                                      cz_par['sigma']) * coszen_res_scale
            model = cz_par.get('model', 'gaussian_coszen')
            for ie in range(n_e):
                if model == 'gaussian_coszen':
                    # Gaussian in cos(zenith), reflected at -1 and +1
                    probs = _gaussian_bin_probabilities(cz_true, cz_sigma[ie], cz_reco_edges)
                    probs += _gaussian_bin_probabilities(-2. - cz_true, cz_sigma[ie], cz_reco_edges)
                    probs += _gaussian_bin_probabilities(2. - cz_true, cz_sigma[ie], cz_reco_edges)
                else:
                    probs = _angular_coszen_probabilities(
                        cz_true, np.full(n_cz, cz_sigma[ie]), cz_reco_edges)
                coszen[iclass, ie] = probs

        responses[name] = {'pid': pid, 'energy': energy, 'coszen': coszen}
    return responses
