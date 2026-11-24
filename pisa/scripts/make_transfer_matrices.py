#!/usr/bin/env python

"""
Build Earth-transmission transfer matrices with nuSQuIDS for the
`absorption.transfer_matrix` stage, or merge partial builds.

The true binning is taken from a binned response file (so that it matches the
pipeline) or given explicitly. A build can be split across jobs: job i of n
computes every n-th input column, and the partial files are merged afterwards.

Examples ::

    # Whole build in one go, binning from a response file
    NUSQUIDS_DATA_PATH=/path/to/nuSQuIDS/data \\
        pisa-make_transfer_matrices --response-file my_response.json -o tm.hdf5

    # Split in 8 jobs, then merge
    pisa-make_transfer_matrices --emin 1e2 --emax 1e8 --n-energy 120 \\
        --n-coszen 20 --n-jobs 8 --job-index 3 -o tm_part3.hdf5
    pisa-make_transfer_matrices --merge tm_part*.hdf5 -o tm.hdf5

Cost (illume, one core, 20 cos(zenith) x 100 energy bins, default
tolerances): about 3.3 s per column, 6 * n_E columns, so ~33 min.
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np

from pisa.utils.binned_response import load_response_file
from pisa.utils.transfer_matrix import (
    DEFAULT_SETTINGS, build_transfer_matrices, merge_transfer_matrix_files,
    write_transfer_matrix_file
)

__all__ = ['parse_args', 'main']

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


def parse_args(description=__doc__):
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = ArgumentParser(description=description,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--outfile', required=True,
                        help='Output HDF5 file')
    parser.add_argument('--merge', nargs='+', default=None, metavar='FILE',
                        help='Merge these partial files into OUTFILE instead of building')

    # True binning: from a response file, or explicit log energy / linear cos(zenith)
    parser.add_argument('--response-file', default=None,
                        help='Binned response file whose true binning to use')
    parser.add_argument('--energy-name', default='true_energy',
                        help='Name of the true-energy dimension')
    parser.add_argument('--coszen-name', default='true_coszen',
                        help='Name of the true cos(zenith) dimension')
    parser.add_argument('--emin', type=float, default=None, help='Lowest energy edge, GeV')
    parser.add_argument('--emax', type=float, default=None, help='Highest energy edge, GeV')
    parser.add_argument('--n-energy', type=int, default=None,
                        help='Number of log-spaced energy bins')
    parser.add_argument('--czmin', type=float, default=-1., help='Lowest cos(zenith) edge')
    parser.add_argument('--czmax', type=float, default=1., help='Highest cos(zenith) edge')
    parser.add_argument('--n-coszen', type=int, default=None,
                        help='Number of linear cos(zenith) bins')

    # Physics and solver settings
    parser.add_argument('--cross-sections', default=DEFAULT_SETTINGS['cross_sections'],
                        choices=['csms', 'wcg24'], help='nuSQuIDS cross-section tables')
    parser.add_argument('--no-tau-regeneration', action='store_true',
                        help='Switch off tau regeneration')
    parser.add_argument('--no-glashow', action='store_true',
                        help='Switch off the Glashow resonance')
    parser.add_argument('--rel-error', type=float, default=DEFAULT_SETTINGS['rel_error'],
                        help='nuSQuIDS relative tolerance')
    parser.add_argument('--abs-error', type=float, default=DEFAULT_SETTINGS['abs_error'],
                        help='nuSQuIDS absolute tolerance (inputs are order one)')

    # Splitting across jobs
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of jobs in the split')
    parser.add_argument('--job-index', type=int, default=0, help='This job, 0 to n_jobs - 1')
    parser.add_argument('--description', default='', help='Stored in the file')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace OUTFILE if it exists')
    return parser.parse_args()


def get_edges(args):
    """
    True-energy (GeV) and cos(zenith) edges from the arguments.

    Returns
    -------
    energy_edges_gev, coszen_edges : np.ndarray
    """
    if args.response_file is not None:
        true_binning = load_response_file(args.response_file)['true_binning']
        energy_edges_gev = true_binning[args.energy_name].bin_edges.m_as('GeV')
        coszen_edges = true_binning[args.coszen_name].bin_edges.m
        return energy_edges_gev, coszen_edges

    for one_name in ['emin', 'emax', 'n_energy', 'n_coszen']:
        if getattr(args, one_name) is None:
            raise ValueError('Give --response-file, or --emin, --emax, --n-energy '
                             'and --n-coszen')
    energy_edges_gev = np.logspace(np.log10(args.emin), np.log10(args.emax),
                                   args.n_energy + 1)
    coszen_edges = np.linspace(args.czmin, args.czmax, args.n_coszen + 1)
    return energy_edges_gev, coszen_edges


def main():
    """Build (part of) a transfer-matrix file, or merge partial files."""
    args = parse_args()

    if args.merge is not None:
        print('Merging', len(args.merge), 'files into', args.outfile)
        merge_transfer_matrix_files(args.merge, args.outfile,
                                    description=args.description,
                                    overwrite=args.overwrite)
        return

    energy_edges_gev, coszen_edges = get_edges(args)
    n_e = len(energy_edges_gev) - 1

    # Columns of this job: every n_jobs-th, so that jobs mix low and high energies
    if not 0 <= args.job_index < args.n_jobs:
        raise ValueError('--job-index must be in [0, n_jobs)')
    input_columns = []
    for ichan in range(6):
        for ie in range(n_e):
            if (ichan * n_e + ie) % args.n_jobs == args.job_index:
                input_columns.append((ichan, ie))

    result = build_transfer_matrices(
        energy_edges_gev, coszen_edges, input_columns=input_columns,
        cross_sections=args.cross_sections,
        tau_regeneration=not args.no_tau_regeneration,
        glashow_resonance=not args.no_glashow,
        rel_error=args.rel_error,
        abs_error=args.abs_error,
    )
    write_transfer_matrix_file(
        args.outfile, result['matrix'], energy_edges_gev, coszen_edges,
        computed=result['computed'], settings=result['settings'],
        energy_name=args.energy_name, coszen_name=args.coszen_name,
        description=args.description, overwrite=args.overwrite,
    )
    print('Wrote', args.outfile)


if __name__ == '__main__':
    main()
