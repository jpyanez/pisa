"""
Stage to implement getting the contribution to fluxes from astrophysical neutrino sources
"""
import numpy as np

from pisa import FTYPE
from pisa.core.param import Param, ParamSet
from pisa.core.stage import Stage
from pisa.utils.profiler import profile

__all__ = ['astrophysical', 'spectral_index_scale', 'init_test']

PIVOT = FTYPE(100.0e3)  # GeV


class astrophysical(Stage):  # pylint: disable=invalid-name
    """
    Stage to add a power-law astrophysical neutrino flux to the weights.

    The flux of each neutrino type (flavour, neutrino or antineutrino) is

        astro_norm * 0.787e-18 * (E / 100 TeV)**(-2.5 + astro_delta)
        [GeV^-1 cm^-2 s^-1 sr^-1],

    the same for every container, i.e. a 1:1:1 flavour ratio and equal
    neutrino and antineutrino fluxes. The per-flavour neutrino+antineutrino
    normalisation at 100 TeV is therefore 2 * 0.787e-18 = 1.574e-18
    GeV^-1 cm^-2 s^-1 sr^-1 for astro_norm = 1. The flux is isotropic.

    Internally the flux is stored in GeV^-1 m^-2 s^-1 sr^-1 (0.787e-14 at
    100 TeV), the convention of the other PISA flux stages (e.g.
    `flux.daemon_flux`, `flux.honda_ip`), so that it combines with them and
    with effective areas in m^2.

    On apply, `initial_weights * astro_flux` is *added* to `weights`, so that
    every later stage (effective area, absorption, reconstruction) acts on the
    sum of atmospheric and astrophysical contributions. This requires the
    incoming `weights` to carry the atmospheric flux already: place this stage
    after the stage that multiplies the atmospheric flux into the weights
    (e.g. `osc.prob3`) and before the effective-area stage. Earth oscillations
    of the astrophysical flux are neglected (they matter only near 100 GeV,
    where the astrophysical flux is ~1e-3 of the atmospheric one).

    For a pipeline with no atmospheric flux, set `astro_only=True`: the
    weights are then replaced by `initial_weights * astro_flux`.

    Parameters
    ----------
    astro_only : bool
        Replace the weights instead of adding to them.

    params : ParamSet
        Must have parameters::

            astro_delta : quantity (dimensionless)
                Shift of the spectral index; positive values harden the spectrum.
            astro_norm : quantity (dimensionless)
                Scale of the normalisation relative to 0.787e-18.

    Notes
    -----

    Expected container keys are::

        "true_energy", "initial_weights", "weights"

    In binned pipelines (`data.binned_grid`) `initial_weights` is the bin
    volume in GeV sr, so the added term is the flux integrated over the bin
    (evaluated at the bin centre).

    TODO: flavour ratio and neutrino/antineutrino fraction as parameters.
    """

    def __init__(self, astro_only=False, **std_kwargs):
        self._central_gamma = FTYPE(-2.5)
        # Per neutrino type at the pivot: 0.787e-18 GeV^-1 cm^-2 s^-1 sr^-1,
        # converted to the PISA flux unit GeV^-1 m^-2 s^-1 sr^-1
        self._central_norm = FTYPE(0.787e-18 * 1e4)
        self.astro_only = bool(astro_only)

        expected_params = (
            "astro_delta",
            "astro_norm",
        )

        expected_container_keys = (
            'true_energy',
            'initial_weights',
            'weights',
        )

        super().__init__(
            expected_params=expected_params,
            expected_container_keys=expected_container_keys,
            **std_kwargs,
        )

    def setup_function(self):
        """
        Compute the nominal flux at each entry
        """
        for container in self.data:
            true_energy = container["true_energy"]
            container["astro_flux_nominal"] = (
                self._central_norm * np.power(true_energy / PIVOT, self._central_gamma)
            ).astype(FTYPE)
            container["astro_flux"] = np.copy(container["astro_flux_nominal"])
            container.mark_changed("astro_flux_nominal")
            container.mark_changed("astro_flux")

    @profile
    def compute_function(self):
        """
        Apply the normalisation and the spectral-index shift
        """
        delta = self.params.astro_delta.value.m_as("dimensionless")
        norm = self.params.astro_norm.value.m_as("dimensionless")

        for container in self.data:
            spec_scale = spectral_index_scale(container["true_energy"], delta)
            container["astro_flux"] = (
                norm * container["astro_flux_nominal"] * spec_scale
            ).astype(FTYPE)
            container.mark_changed("astro_flux")

    @profile
    def apply_function(self):
        for container in self.data:
            astro_term = container["initial_weights"] * container["astro_flux"]
            if self.astro_only:
                container["weights"] = astro_term
            else:
                container["weights"] = container["weights"] + astro_term
            container.mark_changed("weights")


def spectral_index_scale(true_energy, delta_index):
    """
    Calculate spectral index scale.
    Adjusts the weights for events in an energy dependent way according to a
    shift in spectral index, applied about a user-defined energy pivot.

    Parameters
    ----------
    true_energy : np.ndarray
        Neutrino energies, GeV.
    delta_index : float
        Shift of the spectral index.

    Returns
    -------
    np.ndarray
        (true_energy / PIVOT)**delta_index, same shape as `true_energy`.
    """
    return np.power(true_energy / PIVOT, delta_index)


def init_test(**param_kwargs):
    """Instantiation example"""
    param_set = ParamSet([
        Param(name="astro_norm", value=1.0, **param_kwargs),
        Param(name="astro_delta", value=0.0, **param_kwargs),
    ])

    return astrophysical(params=param_set)
