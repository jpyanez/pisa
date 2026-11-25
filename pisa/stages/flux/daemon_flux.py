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

from pisa import FTYPE, ureg
from pisa.core.binning import OneDimBinning
from pisa.core.container import Container, ContainerSet
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.comparisons import ALLCLOSE_KW
from pisa.utils.log import logging
from pisa.utils.profiler import profile
from pisa.utils.random_numbers import get_random_state

__all__ = ['MIN_VERSION', 'ENERGY_GRID', 'daemon_flux', 'init_test',
           'test_linear_flux']

MIN_VERSION = "0.8.0"
"""Minimum daemonflux package version for correct chi2 prior penalty"""

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
                 use_linear_flux=False, **std_kwargs):
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

        self.flux_obj = Flux(location='IceCube', use_calibration=True,
                             cal_file=self.cal_file)
        """daemonflux flux object"""

        # self.flux_obj.zenith_angles is list of strings of (approx.) uniformly-spaced
        # values in deg between 0° & 180° -> make ascending array first
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

        self.daemon_names = self.flux_obj.params.known_parameters
        """Parameter names from daemonflux"""

        self.daemon_params = ['daemon_' +
            p.replace('pi+','pi').replace('pi-','antipi')
            .replace('K+','K').replace('K-','antiK') for p in self.daemon_names
        ]
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

        if self.linear_flux:
            self.setup_linear_flux()

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

