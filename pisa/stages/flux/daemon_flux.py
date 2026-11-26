"""
Implementation of DAEMON flux (https://arxiv.org/abs/2303.00022)
by Juan Pablo Yañez and Anatoli Fedynitch for use in PISA.

Module originally authored by Maria Liubarska, J.P. Yañez 2023
"""

from daemonflux import Flux
from daemonflux import __version__ as daemon_version
import fast_interp

import numpy as np
from packaging.version import Version
from scipy import interpolate
from scipy import sparse

from pisa import FTYPE, ureg
from pisa.core.binning import OneDimBinning
from pisa.core.container import Container, ContainerSet
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.random_numbers import get_random_state

__all__ = ['MIN_VERSION', 'MIN_VERSION_SKY', 'ENERGY_GRID', 'daemon_flux', 'SkyGridFlux',
           'pisa_param_name', 'init_test', 'test_linear_flux', 'test_sky_flux']

MIN_VERSION = "0.8.0"
"""Minimum daemonflux package version for correct chi2 prior penalty"""

MIN_VERSION_SKY = "0.9.0"
"""Minimum daemonflux package version for version-2 response bundles with sky maps"""

ENERGY_GRID = np.logspace(-1, 5, 500, dtype=FTYPE) * ureg.GeV
"""Default array of true neutrino energies at which to request fluxes
from daemonflux"""

"""
TODO/Notes:
* does every parameter affect all fluxes (i.e., further caching potential)?
* any benefit from optimising/adapting arguments of fast_interp.interp2d?
* We only seem to get performance benefits from fast_interp package when
  its JIT target is "parallel", even when numba.get_num_threads() = 1
  (PISA_NUM_THREADS=1). The latter is ensured when TARGET="cpu".
  This parallel compilation happens whenever true_energy.size >
  fast_interp.fast_interp.serial_cutoffs[2] (default value: 400).
  Accordingly, the number of events/grid size needs to exceed this.
  Serial compilation seems to be much slower than not using fast_interp
  in the first place, at least for the public 3-year event sample with
  calc_mode="events". However, it could be enforced by calling:
  fast_interp.set_serial_cutoffs(dimension=2, cutoff=np.inf).
"""
class daemon_flux(Stage):  # pylint: disable=invalid-name
    """
    DAEMON flux stage

    Parameters
    ----------

    calibration_file: str, optional
        Path to the calibration file to be used

    use_fast_interp : bool (default: False)
        Whether to use numba-accelerated flux interpolation
        (experimental)

    energy_grid : array-like (default: None)
        Array of true neutrino energies at which to request fluxes
        from daemonflux. If `None`, the default :py:data:`ENERGY_GRID`
        will be used.

    use_linear_flux : bool (default: False)
        Precompute, once at setup, the flux of every event at the nominal
        parameters and its change per unit (1 sigma) shift of each daemonflux
        parameter. Each compute is then a matrix-vector product instead of
        rebuilding and evaluating the flux splines. This is exact (to
        floating-point rounding) because the daemonflux flux,
        exp(spline(log E)) * (1 + sum_k v_k J_k(log E)), is linear in the
        parameters and so are the spline interpolation steps of this stage.
        Costs 1 + n_params flux evaluations at setup and
        (n_params + 1) * 2 floats per event of memory.

    bundle : str, optional
        Path to a daemonflux bundle manifest (daemonflux version-2 response
        files, e.g. a 3D sky of one site). Requires `sky_site`. The parameters
        are then those of the bundle's basis, and `calibration_file`, if
        given, replaces the bundle's own calibration. Fluxes are evaluated by
        :py:class:`SkyGridFlux`; `use_linear_flux` and `use_fast_interp` do
        not apply.

    sky_site : str, optional
        Sky map of the bundle to use, e.g. "SouthPole". Fluxes are averaged
        over azimuth.

    sky_points_per_decade : int (default: 60)
        Energy grid density of :py:class:`SkyGridFlux` (log-uniform, spanning
        the events' true energies).

    sky_cz_nodes : int (default: 401)
        Number of cos(zenith) nodes per hemisphere of :py:class:`SkyGridFlux`.

    params: ParamSet
        Must have parameters::

            daemon_K_158G : quantity (dimensionless)
            daemon_K_2P : quantity (dimensionless)
            daemon_K_31G : quantity (dimensionless)
            daemon_antiK_158G : quantity (dimensionless)
            daemon_antiK_2P : quantity (dimensionless)
            daemon_antiK_31G : quantity (dimensionless)
            daemon_n_158G : quantity (dimensionless)
            daemon_n_2P : quantity (dimensionless)
            daemon_p_158G : quantity (dimensionless)
            daemon_p_2P : quantity (dimensionless)
            daemon_pi_158G : quantity (dimensionless)
            daemon_pi_20T : quantity (dimensionless)
            daemon_pi_2P : quantity (dimensionless)
            daemon_pi_31G : quantity (dimensionless)
            daemon_antipi_158G : quantity (dimensionless)
            daemon_antipi_20T : quantity (dimensionless)
            daemon_antipi_2P : quantity (dimensionless)
            daemon_antipi_31G : quantity (dimensionless)
            daemon_GSF_1 : quantity (dimensionless)
            daemon_GSF_2 : quantity (dimensionless)
            daemon_GSF_3 : quantity (dimensionless)
            daemon_GSF_4 : quantity (dimensionless)
            daemon_GSF_5 : quantity (dimensionless)
            daemon_GSF_6 : quantity (dimensionless)

    Notes
    -----
    Implements no apply.

    Expected container keys are::

        "true_energy", "true_coszen", "nubar"
    """

    def __init__(self, calibration_file=None, use_fast_interp=False, energy_grid=None,
                 use_linear_flux=False, bundle=None, sky_site=None,
                 sky_points_per_decade=60, sky_cz_nodes=401, **std_kwargs):
        self.cal_file = calibration_file
        """Custom daemonflux calibration file"""
        if self.cal_file is not None:
            logging.debug('Requested custom DAEMON flux calibration file: %s',
                          self.cal_file)

        if Version(daemon_version) < Version(MIN_VERSION):
            logging.fatal(
                "Detected daemonflux version %s < %s, which will lead to"
                " incorrect penalty calculations. You must update your"
                " daemonflux package to use this stage. You can do it by"
                " running 'pip install daemonflux --upgrade'.",
                daemon_version, MIN_VERSION
            )
            raise RuntimeError(f'Detected daemonflux version < {MIN_VERSION}')

        self.fast_interp = use_fast_interp
        """Use numba-accelerated flux interpolation"""

        if isinstance(use_linear_flux, str):
            use_linear_flux = use_linear_flux.strip().lower() == 'true'
        self.linear_flux = use_linear_flux
        """Use precomputed per-event fluxes and their linear parameter dependence"""

        self.bundle = bundle
        """daemonflux bundle manifest (version-2 responses), or None"""
        self.sky_site = sky_site
        """Sky map of the bundle"""
        self.sky_points_per_decade = int(sky_points_per_decade)
        self.sky_cz_nodes = int(sky_cz_nodes)
        self.sky_grid = None
        """SkyGridFlux, built at setup when a bundle is used"""

        if self.bundle is not None:
            if Version(daemon_version) < Version(MIN_VERSION_SKY):
                raise RuntimeError(
                    f'daemonflux {daemon_version} cannot read bundles; need'
                    f' >= {MIN_VERSION_SKY}'
                )
            if self.sky_site is None:
                raise ValueError('`sky_site` is required with `bundle`')
            self.flux_obj = Flux(bundle=self.bundle, cal_file=self.cal_file)
            if self.sky_site not in self.flux_obj.sky_sites:
                raise ValueError(
                    f'Sky site {self.sky_site} not in the bundle'
                    f' ({self.flux_obj.sky_sites})'
                )
        else:
            self.flux_obj = Flux(location='IceCube', use_calibration=True,
                                 cal_file=self.cal_file)
        """daemonflux flux object"""

        # self.flux_obj.zenith_angles is list of strings of (approx.) uniformly-spaced
        # values in deg between 0° & 180° -> make ascending array first
        if self.bundle is None:
            self._icangles_asc = np.array(sorted(map(float, self.flux_obj.zenith_angles),
                                           reverse=False), dtype=FTYPE)
            self._costheta_angles_asc = np.cos(np.deg2rad(self._icangles_asc))[::-1]

        if energy_grid is None:
            energy_grid = ENERGY_GRID.m_as("GeV")
        else:
            # user requested a custom grid
            if isinstance(energy_grid, str):
                energy_grid = eval(energy_grid)
            # required to be specified with energy units
            assert (hasattr(energy_grid, "magnitude") and
                    energy_grid.is_compatible_with("GeV"))
            energy_grid = energy_grid.m_as("GeV")
            if not isinstance(energy_grid, (list, np.ndarray)):
                energy_grid = [energy_grid]
            logging.debug("Requested custom energy grid (GeV) to pass to daemonflux: "
                          "%s", energy_grid)
        self._egrid = energy_grid

        if self.fast_interp:
            logging.debug("Using daemon_flux service with fast interpolation...")
            # Obtain uniform spacings in interpolation dimensions:
            # Cosine zenith we got from daemonflux
            costheta_deltas = self._costheta_angles_asc[1:] - self._costheta_angles_asc[:-1]
            # TODO: These deltas don't quite seem to agree at desired precision, looks
            # like numerical inaccuracy.
            #assert np.allclose(costheta_delta[0], costheta_deltas, **ALLCLOSE_KW)
            self._costheta_delta = costheta_deltas[0]

            if not OneDimBinning.is_bin_spacing_log_uniform(self._egrid):
                # Could in principle relax this, but then couldn't just assume that
                # log(energy) is uniform.
                raise ValueError(
                    "Energy grid required to have log-uniform spacing"
                    " (energy_grid=np.logspace(...)) when fast_interp=True."
                )
            # Energy dimension is made uniform by taking log
            self._egrid_log = np.log10(self._egrid)
            egrid_log_deltas = self._egrid_log[1:] - self._egrid_log[:-1]
            if not np.allclose(egrid_log_deltas[0], egrid_log_deltas, **ALLCLOSE_KW):
                raise ValueError("Need uniformly-spaced log-energy values!")
            self._egrid_log_delta = egrid_log_deltas[0]

        if self.bundle is None:
            self.daemon_names = self.flux_obj.params.known_parameters
        else:
            self.daemon_names = list(
                self.flux_obj._get_flux_instance(self.sky_site)._params.known_parameters
            )
        """Parameter names from daemonflux"""

        self.daemon_params = [pisa_param_name(p) for p in self.daemon_names]
        """PISA config-compatible parameter names with added prefixes"""

        # Internal parameter to carry on chi2 penalty from daemonflux (using
        # covariance matrix)
        daemon_chi2 = Param(
            name='daemon_chi2',
            nominal_value=0., value=0.,
            prior=None, range=None, is_fixed=True
        )

        # Saving number of parameters into a internal param in order to check
        # that we don't have non-daemonflux params with 'daemon_' in their
        # name, which will make prior penalty calculation incorrect
        daemon_params_len = Param(
            name='daemon_params_len',
            nominal_value=len(self.daemon_names)+2,
            value=len(self.daemon_names)+2,
            prior=None, range=None, is_fixed=True
        )
        std_kwargs['params'].update([daemon_chi2, daemon_params_len])

        expected_params = tuple(
            self.daemon_params + ['daemon_chi2', 'daemon_params_len']
        )
        expected_container_keys = (
            'true_energy',
            'true_coszen',
            'nubar',
        )

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Just prepare empty nue(bar) & numu(bar) atmos. flux arrays on setup.
        """
        for container in self.data:
            container['nu_flux'] = np.empty((container.size, 2), dtype=FTYPE)

        if self.bundle is not None:
            self.setup_sky_flux()
        elif self.linear_flux:
            self.setup_linear_flux()

    def setup_sky_flux(self):
        """
        Build the :py:class:`SkyGridFlux` and each container's interpolation
        matrix for a bundle with a sky map.
        """
        e_min, e_max = np.inf, 0.
        for container in self.data:
            e_min = min(e_min, np.min(container['true_energy']))
            e_max = max(e_max, np.max(container['true_energy']))
        # Grid spanning the events, on whole decades
        log_lo, log_hi = np.floor(np.log10(e_min)), np.ceil(np.log10(e_max))
        n_points = int(round((log_hi - log_lo) * self.sky_points_per_decade)) + 1
        energy_grid = np.logspace(log_lo, log_hi, n_points)
        logging.info("daemon_flux: sky %s on %d energies (%.3g-%.3g GeV) x %d cos(zenith)"
                     " nodes per hemisphere", self.sky_site, n_points, energy_grid[0],
                     energy_grid[-1], self.sky_cz_nodes)
        self.sky_grid = SkyGridFlux(
            self.flux_obj, self.sky_site, ['nue', 'antinue', 'numu', 'antinumu'],
            energy_grid, n_cz_half=self.sky_cz_nodes,
        )
        if self.sky_grid.names != self.daemon_names:
            raise ValueError('Parameter order of the sky map differs from the stage')
        self._sky_matrix = {}
        for container in self.data:
            true_energy = np.asarray(container['true_energy'], dtype=np.float64)
            self._sky_matrix[container.name] = (
                self.sky_grid.event_matrix(true_energy, container['true_coszen']),
                true_energy**-3 * 1e4,
            )

    def setup_linear_flux(self):
        """
        Precompute per-event fluxes for `use_linear_flux`.

        Stores, per container, the nue(bar) & numu(bar) fluxes at nominal
        parameter values (shape (n_events, 2)) and their change for a +1 sigma
        shift of each daemonflux parameter (shape (n_params, n_events, 2)).
        """
        logging.info("daemon_flux: precomputing linear flux dependence for"
                     " %d parameters", len(self.daemon_names))
        nominal_params = {name: 0. for name in self.daemon_names}
        self._lin_nominal = self.event_fluxes(params=nominal_params)
        lin_shifts = {name: [] for name in self._lin_nominal}
        for daemon_name in self.daemon_names:
            shifted_params = dict(nominal_params)
            shifted_params[daemon_name] = 1.
            shifted = self.event_fluxes(params=shifted_params)
            for name, flux in shifted.items():
                lin_shifts[name].append(flux - self._lin_nominal[name])
        self._lin_shifts = {}
        for name, shifts in lin_shifts.items():
            self._lin_shifts[name] = np.stack(shifts, axis=0)

    def event_fluxes(self, params):
        """
        Fluxes of nue(bar) and numu(bar) for every event of every container.

        Parameters
        ----------
        params : Dict[str, float]
            daemonflux parameter values (in units of sigma).

        Returns
        -------
        Dict[str, ndarray]
            Container name -> array of shape (n_events, 2) with the nue(bar)
            and numu(bar) fluxes, see :py:meth:`evaluate_flux_map`.
        """
        flux_maps = {}
        for particle in ['numu', 'antinumu', 'nue', 'antinue']:
            flux_maps[particle] = self.make_2d_flux_map(
                particle=particle, params=params
            )

        fluxes = {}
        for container in self.data:
            nubar = container['nubar']
            nu_flux = np.empty((container.size, 2), dtype=FTYPE)
            nu_flux[:, 0] = self.evaluate_flux_map(
                flux_map=flux_maps['antinue'] if nubar < 0 else flux_maps['nue'],
                true_energy=container['true_energy'],
                true_coszen=container['true_coszen']
            )
            nu_flux[:, 1] = self.evaluate_flux_map(
                flux_map=flux_maps['antinumu'] if nubar < 0 else flux_maps['numu'],
                true_energy=container['true_energy'],
                true_coszen=container['true_coszen']
            )
            fluxes[container.name] = nu_flux
        return fluxes

    @profile
    def compute_function(self):
        """
        Compute nominal atmospheric fluxes of nue(bar) and numu(bar).
        """
        # get modified parameters (in units of sigma)
        modif_param_dict = {}
        for i, k in enumerate(self.daemon_params):
            modif_param_dict[self.daemon_names[i]] = self.params[k].value.m_as("dimensionless")

        if self.bundle is not None:
            # flux on the grid once per species, then per event by the
            # precomputed interpolation matrices
            shifts = np.array([modif_param_dict[name] for name in self.daemon_names])
            self.params['daemon_chi2'].value = self.sky_grid.chi2(shifts)
            grid = {}
            for species in ['nue', 'antinue', 'numu', 'antinumu']:
                grid[species] = self.sky_grid.grid_flux(species, shifts).ravel()
            for container in self.data:
                matrix, uconv = self._sky_matrix[container.name]
                bar = 'anti' if container['nubar'] < 0 else ''
                container['nu_flux'][:, 0] = matrix @ grid[bar + 'nue'] * uconv
                container['nu_flux'][:, 1] = matrix @ grid[bar + 'numu'] * uconv
                container.mark_changed("nu_flux")
            return

        # update chi2 parameter
        self.params['daemon_chi2'].value = self.flux_obj.chi2(modif_param_dict)

        if self.linear_flux:
            # flux = nominal + sum_k p_k * (flux change per +1 sigma of k)
            param_values = np.array(
                [modif_param_dict[name] for name in self.daemon_names], dtype=FTYPE
            )
            for container in self.data:
                container['nu_flux'][:] = (
                    self._lin_nominal[container.name]
                    + np.tensordot(param_values, self._lin_shifts[container.name], axes=1)
                )
                container.mark_changed("nu_flux")
            return

        # Obtain modified fluxes using provided parameters at desired points
        # in true_energy and true_coszen now
        fluxes = self.event_fluxes(params=modif_param_dict)
        for container in self.data:
            container['nu_flux'][:] = fluxes[container.name]
            container.mark_changed("nu_flux")


    def make_2d_flux_map(self, particle='numuflux', params=None):
        """
        Create an interpolated 2d (energy-coszen) flux map using daemonflux.

        Parameters
        ----------
        particle : str (default: "numuflux")
            Type of flux to be returned. See `daemonflux.flux.Flux.quantities`.
        params : Dict[str, float], optional
            Dictionary of parameter values for off-baseline shifts.

        Returns
        -------
        fcn : interpolant
            Bivariate spline approximation of flux over energy-coszen grid.
            `scipy.interpolate.RectBivariateSpline` if not `fast_interp`,
            otherwise `fast_interp.interp2d`. See `daemonflux.Flux.flux()` for
            units.
        """
        if params is None:
            params = {}
        #start_t = time.time()
        # Obtain flux from daemonflux: expects ascending zenith angles in deg
        # TODO: Why does flux_obj need these to be handed back to it again?
        flux_ref = self.flux_obj.flux(
            energy=self._egrid,
            zenith_deg=self._icangles_asc,
            quantity=particle,
            params=params
        )
        #stop_t = time.time()
        #logging.info("daemonflux %s flux generation duration (s): %s", particle, f"{stop_t - start_t:.2e}")
        # Now flip zenith angle dimension so we can interpolate in costheta
        # with increasing costheta
        flux_ref_lr = np.fliplr(flux_ref)

        #start_t = time.time()
        # Return interpolant which can be evaluated later
        if not self.fast_interp:
            fcn = interpolate.RectBivariateSpline(
                x=self._egrid, y=self._costheta_angles_asc, z=flux_ref_lr
            )
        else:
            fcn = fast_interp.interp2d(
                a=[min(self._egrid_log), min(self._costheta_angles_asc)],
                b=[max(self._egrid_log), max(self._costheta_angles_asc)],
                h=[self._egrid_log_delta, self._costheta_delta],
                f=flux_ref_lr,
                k=3,
                p=[False, False],
                c=[True, True],
                e=[0, 0]
            )
        #stop_t = time.time()
        #logging.info("splining duration (s): %s", f"{stop_t - start_t:.2e}")
        return fcn


    def evaluate_flux_map(self, flux_map, true_energy, true_coszen):
        """
        Evaluate bivariate spline approximation of flux.

        Parameters
        ----------
        flux_map : scipy.interpolate.RectBivariateSpline
        true_energy : array_like
            True energies in GeV at which to evaluate
        true_coszen : array_like
            True coszens at which to evaluate

        Returns
        -------
        ndarray
            Flux in units of 1/(GeV m² s sr)
        """
        # flux unit conversion factor (see daemonflux.Flux.flux())
        uconv = true_energy**-3 * 1e4
        if not self.fast_interp:
            return flux_map.ev(true_energy, true_coszen) * uconv
        # Remember to transform into log-energy space again before evaluating
        return flux_map(np.log10(true_energy), true_coszen) * uconv


def pisa_param_name(daemon_name):
    """
    PISA parameter name of a daemonflux parameter.

    Parameters
    ----------
    daemon_name : str
        daemonflux name, e.g. "pi+_20T" or "GSF2026_p_1.5PeV".

    Returns
    -------
    str
        "daemon_" + name with charges spelled out and "." -> "p", e.g.
        "daemon_pi_20T", "daemon_antiK_2P", "daemon_GSF2026_p_1p5PeV".
    """
    name = daemon_name.replace('pi+', 'pi').replace('pi-', 'antipi')
    name = name.replace('K+', 'K').replace('K-', 'antiK').replace('.', 'p')
    return 'daemon_' + name


class SkyGridFlux:
    """
    Azimuth-averaged flux of a daemonflux sky map on an (energy, cos-zenith)
    grid, fast for repeated parameter changes.

    daemonflux evaluates each table (one per sky bin, plus the high-energy
    profile) as f0 * (1 + sum_add(J a + C a^2 / 2)) * prod_factor(1 + J a +
    C a^2 / 2) and combines tables with fixed direction weights. Here f0, J
    and C of every table are interpolated once to the energy grid, so each
    call is two matrix-vector products per species plus the direction
    weights. On the grid nodes this is daemonflux's own result to rounding,
    for any parameter values. Events are bilinearly interpolated in
    (ln E, cos zenith) of the E^3-weighted flux, on separate grids for the
    two hemispheres so the horizon cusp is kept.

    Uses daemonflux internals of the sky entry (`_model`, `_params`,
    `_weights`, `_map_tables`, `_he_tables`, `_he`, `_blend`); written for
    daemonflux main f7021c5. :py:func:`test_sky_flux` checks it against
    `Flux.flux`.

    Parameters
    ----------
    flux_obj : daemonflux.Flux
        Flux loaded from a bundle with a sky map.
    site : str
        Sky site, e.g. "SouthPole".
    species_list : list of str
        Species to prepare, e.g. ["nue", "antinue", "numu", "antinumu"]
        (conventional fluxes).
    energy_grid : array_like
        Ascending energies (GeV) of the grid.
    n_cz_half : int
        cos(zenith) nodes per hemisphere.
    """

    def __init__(self, flux_obj, site, species_list, energy_grid, n_cz_half=401):
        from daemonflux.sky import _smoothstep

        sky = flux_obj._get_flux_instance(site)
        self.sky = sky
        model = sky._model
        self.names = list(sky._params.known_parameters)
        """Parameter order of `shifts` (daemonflux known_parameters)"""
        self._index = np.array([model.names.index(n) for n in self.names])
        self._n_model = len(model.names)
        self._base_values = np.array(sky._params.values, dtype=np.float64)
        self._errors = np.array(sky._params.errors, dtype=np.float64)
        self._invcov = np.linalg.inv(np.asarray(sky._params.cov, dtype=np.float64))
        self._factor = np.array(model.factor, dtype=bool)
        self._log = np.array(model.log, dtype=bool)

        self.energy = np.asarray(energy_grid, dtype=np.float64)
        log_e = np.log(self.energy)

        # cos(zenith) nodes; the hemisphere grids end just below and start just
        # above the horizon, where daemonflux's zenith spline is split
        eps = 1e-9
        self.cz_lower = np.linspace(-1., 0., n_cz_half)
        self.cz_upper = np.linspace(0., 1., n_cz_half)
        cz_eval = np.concatenate([self.cz_lower[:-1], [-eps], [eps], self.cz_upper[1:]])
        zenith_deg = np.degrees(np.arccos(np.clip(cz_eval, -1., 1.)))
        w_map, w_he = sky._weights(zenith_deg, None, False)
        self._w_map = np.ascontiguousarray(w_map)
        self._w_he = None if w_he is None else np.ascontiguousarray(w_he)

        # Weight of the high-energy profile at each grid energy
        if sky._he is not None:
            self._blend = _smoothstep(log_e, *sky._blend)
        else:
            self._blend = np.zeros(log_e.size)
        self._map_mask = self._blend < 1.0
        self._he_mask = self._blend > 0.0

        # f0, J, C of every table at the grid energies where it is used
        add = ~self._factor
        self._tables = {}
        for species in species_list:
            parts = []
            for mask, tables in ((self._map_mask, sky._map_tables),
                                 (self._he_mask, sky._he_tables)):
                if not mask.any() or len(tables) == 0:
                    parts.append(None)
                    continue
                f0_list, jac_list, curv_list = [], [], []
                for one_table in tables:
                    f0, jac, curv = one_table[species].interpolate(log_e[mask])
                    f0_list.append(f0)
                    jac_list.append(jac)
                    if curv is None:
                        curv = np.zeros_like(jac)
                    curv_list.append(curv)
                f0 = np.array(f0_list)          # (n_tables, n_E)
                jac = np.array(jac_list)        # (n_tables, n_model, n_E)
                curv = np.array(curv_list)
                n_add = int(add.sum())
                curv_add = curv[:, add, :]
                # Only parameters with a curvature term (none for the sky
                # tables of the r8h6 bundle) need the second product
                c_index = np.flatnonzero(np.any(curv_add != 0., axis=(0, 2)))
                parts.append({
                    'f0': f0,
                    # additive responses as (n_tables * n_E, n_add)
                    'j_add': np.ascontiguousarray(
                        jac[:, add, :].transpose(0, 2, 1).reshape(-1, n_add)),
                    # curvature of the additive parameters in c_index, as
                    # (n_tables * n_E, len(c_index))
                    'c_index': c_index,
                    'c_add': np.ascontiguousarray(
                        curv_add[:, c_index, :].transpose(0, 2, 1).reshape(f0.size, c_index.size)),
                    'j_fac': jac[:, self._factor, :],
                    'c_fac': curv[:, self._factor, :],
                })
            self._tables[species] = parts

    def model_values(self, shifts):
        """
        Model parameter vector for shifts in units of the calibration sigma.

        Parameters
        ----------
        shifts : array_like
            One value per parameter in :py:attr:`names` order.

        Returns
        -------
        ndarray
            Multipliers a(theta) in the daemonflux model order.
        """
        theta = np.zeros(self._n_model)
        theta[self._index] = self._base_values + self._errors * np.asarray(shifts, dtype=np.float64)
        return np.where(self._log, np.expm1(theta), theta)

    def chi2(self, shifts):
        """
        Prior chi2 of the shifts with the calibration covariance, as
        daemonflux's `Flux.chi2`.

        Parameters
        ----------
        shifts : array_like
            One value per parameter in :py:attr:`names` order.

        Returns
        -------
        float
        """
        delta = self._errors * np.asarray(shifts, dtype=np.float64)
        return float(delta @ self._invcov @ delta)

    def _table_flux(self, part, a):
        """Flux of every table, shape (n_tables, n_E)."""
        a_add = a[~self._factor]
        a_fac = a[self._factor]
        n_tables, n_e = part['f0'].shape
        terms = part['j_add'] @ a_add
        if part['c_index'].size > 0:
            a_curv = a_add[part['c_index']]
            terms += 0.5 * (part['c_add'] @ (a_curv * a_curv))
        s_add = 1.0 + terms.reshape(n_tables, n_e)
        if a_fac.size > 0:
            u = (part['j_fac'] * a_fac[None, :, None]
                 + 0.5 * part['c_fac'] * (a_fac * a_fac)[None, :, None])
            s_fac = np.prod(1.0 + u, axis=1)
        else:
            s_fac = 1.0
        return part['f0'] * s_add * s_fac

    def grid_flux(self, species, shifts):
        """
        E^3-weighted flux on the grid (daemonflux units, GeV^2/(cm^2 s sr)).

        Parameters
        ----------
        species : str
            One of the prepared species.
        shifts : array_like
            One value per parameter in :py:attr:`names` order.

        Returns
        -------
        ndarray
            Shape (2 * n_cz_half, n_E): lower hemisphere nodes, then upper.
        """
        a = self.model_values(shifts)
        out = np.zeros((self._w_map.shape[0], self.energy.size))
        map_part, he_part = self._tables[species]
        if map_part is not None:
            flux = self._table_flux(map_part, a)
            out[:, self._map_mask] += ((1.0 - self._blend[self._map_mask])[None, :]
                                       * (self._w_map @ flux))
        if he_part is not None:
            flux = self._table_flux(he_part, a)
            out[:, self._he_mask] += self._blend[self._he_mask][None, :] * (self._w_he @ flux)
        return out

    def event_matrix(self, true_energy, true_coszen):
        """
        Bilinear interpolation weights of events on the grid.

        Parameters
        ----------
        true_energy : array_like
            Event energies (GeV).
        true_coszen : array_like
            Event cos(zenith), -1 (up-going) to 1.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape (n_events, 2 * n_cz_half * n_E); multiply with
            ``grid_flux(...).ravel()``.
        """
        true_energy = np.asarray(true_energy, dtype=np.float64)
        true_coszen = np.asarray(true_coszen, dtype=np.float64)
        log_grid = np.log(self.energy)
        log_e = np.log(true_energy)
        ie = np.clip(np.searchsorted(log_grid, log_e) - 1, 0, log_grid.size - 2)
        we = (log_e - log_grid[ie]) / (log_grid[ie + 1] - log_grid[ie])

        # Each event uses the grid of its own hemisphere
        n_half = self.cz_lower.size
        lower = true_coszen < 0.0
        ic = np.empty(true_coszen.size, dtype=np.int64)
        ic[lower] = np.searchsorted(self.cz_lower, true_coszen[lower]) - 1
        ic[~lower] = np.searchsorted(self.cz_upper, true_coszen[~lower]) - 1
        ic = np.clip(ic, 0, n_half - 2)
        nodes = np.where(lower, 0, 1)
        c0 = np.where(lower, self.cz_lower[ic], self.cz_upper[ic])
        c1 = np.where(lower, self.cz_lower[ic + 1], self.cz_upper[ic + 1])
        wc = np.clip((true_coszen - c0) / (c1 - c0), 0., 1.)
        row0 = nodes * n_half + ic

        n_e = self.energy.size
        n_events = true_energy.size
        rows = np.repeat(np.arange(n_events), 4)
        cols = np.stack([row0 * n_e + ie, row0 * n_e + ie + 1,
                         (row0 + 1) * n_e + ie, (row0 + 1) * n_e + ie + 1], axis=1).ravel()
        vals = np.stack([(1 - wc) * (1 - we), (1 - wc) * we,
                         wc * (1 - we), wc * we], axis=1).ravel()
        return sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_events, 2 * n_half * n_e)
        )


def init_test(**param_kwargs):
    """Initialisation example"""
    param_set = []
    random_state = get_random_state(random_state=666)
    for pname in Flux(location='IceCube', use_calibration=True).params.known_parameters:
        param = Param(
            name='daemon_' + pname.replace('pi+','pi').replace('pi-','antipi')
                 .replace('K+','K').replace('K-','antiK'),
            value=2 * random_state.rand() - 1,
            **param_kwargs
        )
        param_set.append(param)
    param_set = ParamSet(*param_set)
    return daemon_flux(params=param_set)


def test_linear_flux():
    """
    Check that `use_linear_flux=True` gives the same per-event fluxes and
    daemon_chi2 as the full spline calculation, at nominal and at random
    parameter values within +/- 3 sigma.
    """
    random_state = get_random_state(random_state=0)
    n_events = 2000
    # Only rounding differences are expected
    rtol = 1e-10 if FTYPE == np.float64 else 1e-4

    # Events spread over the energy and coszen range of typical samples
    containers = []
    for name, nubar in [('numu_cc', 1), ('numubar_cc', -1)]:
        container = Container(name)
        container.set_aux_data('nubar', nubar)
        container['true_energy'] = np.power(
            10., random_state.uniform(0., 3., n_events)
        ).astype(FTYPE)
        container['true_coszen'] = random_state.uniform(
            -1., 1., n_events
        ).astype(FTYPE)
        containers.append(container)

    services = {}
    for use_linear_flux in [False, True]:
        service = init_test(prior=None, range=None, is_fixed=False)
        service.linear_flux = use_linear_flux
        service.calc_mode = 'events'
        service.data = ContainerSet('data', [c for c in containers])
        service.setup()
        services[use_linear_flux] = service

    for i in range(4):
        for param in services[False].params.free:
            if i == 0:
                value = 0.
            else:
                value = random_state.uniform(-3., 3.)
            for service in services.values():
                service.params[param.name].value = value * ureg.dimensionless
        fluxes = {}
        for use_linear_flux, service in services.items():
            service.compute_function()
            fluxes[use_linear_flux] = {
                c.name: np.array(c['nu_flux'], copy=True) for c in service.data
            }
        for name in fluxes[False]:
            assert np.allclose(
                fluxes[True][name], fluxes[False][name], rtol=rtol, atol=0.
            ), f'linear flux differs from spline flux for {name}'
        assert np.isclose(
            services[True].params['daemon_chi2'].value.m,
            services[False].params['daemon_chi2'].value.m,
        )
    logging.info('<< PASS : test_linear_flux >>')


def test_sky_flux():
    """
    Check the bundle (sky map) path against daemonflux's own `Flux.flux` and
    `Flux.chi2`, at nominal and random parameter values within +/- 2 sigma.

    Needs a bundle with a sky map: set PISA_DAEMONFLUX_TEST_BUNDLE to its
    manifest and PISA_DAEMONFLUX_TEST_SITE to the site (default
    "SouthPole"); optionally PISA_DAEMONFLUX_TEST_CAL to a calibration file.
    Skipped otherwise.
    """
    import os

    bundle = os.environ.get('PISA_DAEMONFLUX_TEST_BUNDLE')
    if bundle is None or Version(daemon_version) < Version(MIN_VERSION_SKY):
        logging.info('<< SKIP : test_sky_flux (no PISA_DAEMONFLUX_TEST_BUNDLE) >>')
        return
    site = os.environ.get('PISA_DAEMONFLUX_TEST_SITE', 'SouthPole')
    cal_file = os.environ.get('PISA_DAEMONFLUX_TEST_CAL')
    random_state = get_random_state(random_state=1)

    flux_obj = Flux(bundle=bundle, cal_file=cal_file)
    sky = flux_obj._get_flux_instance(site)
    names = list(sky._params.known_parameters)
    param_set = ParamSet(*[
        Param(name=pisa_param_name(n), value=0. * ureg.dimensionless,
              prior=None, range=None, is_fixed=False)
        for n in names
    ])
    service = daemon_flux(params=param_set, bundle=bundle, sky_site=site,
                          calibration_file=cal_file)
    service.calc_mode = 'events'

    # Events over 1 GeV - 10 TeV and all directions
    n_events = 300
    containers = []
    for name, nubar in [('numu_cc', 1), ('numubar_cc', -1)]:
        container = Container(name)
        container.set_aux_data('nubar', nubar)
        container['true_energy'] = np.power(
            10., random_state.uniform(0., 4., n_events)).astype(FTYPE)
        container['true_coszen'] = random_state.uniform(-1., 1., n_events).astype(FTYPE)
        containers.append(container)
    service.data = ContainerSet('data', containers)
    service.setup()

    rtol = 1e-3 if FTYPE == np.float64 else 2e-3
    for i in range(3):
        shifts = np.zeros(len(names)) if i == 0 else random_state.uniform(-2., 2., len(names))
        for name, value in zip(names, shifts):
            service.params[pisa_param_name(name)].value = value * ureg.dimensionless
        service.compute_function()
        param_dict = dict(zip(names, shifts))
        assert np.isclose(service.params['daemon_chi2'].value.m,
                          flux_obj.chi2(param_dict, exp=site), rtol=1e-9, atol=1e-9)
        for container in service.data:
            energy = np.asarray(container['true_energy'], dtype=np.float64)
            zenith = np.degrees(np.arccos(np.asarray(container['true_coszen'], dtype=np.float64)))
            bar = 'anti' if container['nubar'] < 0 else ''
            for column, species in [(0, bar + 'nue'), (1, bar + 'numu')]:
                reference = np.array([
                    sky.flux(np.array([e]), np.array([z]), species, params=param_dict)
                    for e, z in zip(energy, zenith)
                ]).ravel() * energy**-3 * 1e4
                assert np.allclose(container['nu_flux'][:, column], reference,
                                   rtol=rtol, atol=0.), \
                    f'sky flux differs from daemonflux for {container.name} {species}'
    logging.info('<< PASS : test_sky_flux >>')
