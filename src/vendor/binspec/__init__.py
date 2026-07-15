"""binspec, vendored verbatim from github.com/kareemelbadry/binspec.

This is the code behind El-Badry et al. 2018b (the APOGEE binary disentangling
paper). It is copied into this repo so our pipeline does not depend on a clone
of the upstream repository at runtime. The three modules below are the upstream
sources copied without algorithmic change; the only edits are:
  - data paths in utils.py resolve relative to THIS package (so no os.chdir into
    the clone is needed), and
  - the imports in spectral_model.py / fitting.py are intra-package
    (`from . import utils` instead of `import utils`).

The trained networks and grids that the modules load live alongside them:
  neural_nets/NN_normalized_spectra.npz    5-label single-star net
                                            (Teff, logg, [Fe/H], [Mg/Fe], v_macro)
  neural_nets/NN_unnormalized_spectra.npz   synthetic flux / continuum net
  neural_nets/NN_radius.npz                 (Teff, logg, [Fe/H]) -> R (Rsun)
  neural_nets/NN_Teff2_logg2.npz            (Teff1, logg1, [Fe/H], q) -> (Teff2, logg2)
  other_data/apogee_wavelength.npz          7214-pixel DR12/13 wavelength grid
  other_data/cannon_cont_pixels_apogee.npz  Cannon continuum-pixel mask

Public API used by our detector (src/physics.py binspec_* helpers):
  utils.read_in_neural_network, utils.load_wavelength_array,
  utils.load_cannon_contpixels, utils.get_apogee_continuum,
  spectral_model.get_spectrum_from_neural_net,
  fitting.fit_normalized_spectrum_single_star_model,
  fitting.fit_normalized_spectrum_binary_model.

Confirmed to import under Python 3.13.
"""

from . import utils  # noqa: F401
from . import spectral_model  # noqa: F401
from . import fitting  # noqa: F401

__all__ = ["utils", "spectral_model", "fitting"]
