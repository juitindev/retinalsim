"""
RetinalSim — Retinal Prosthesis Simulator
==========================================

An open-source implementation of the Beyeler (2019) axon map model
for simulating prosthetic vision in epiretinal implants.

Based on:
  - Beyeler et al. (2019) Sci Rep 9:9199
  - Jansonius et al. (2009) Vision Res 49:2157-2163
  - Jansonius et al. (2012) Exp Eye Res 105:70-78

Example
-------
>>> from retinalsim import AxonMapModel, make_argus_ii, encode_image
>>> model = AxonMapModel(rho=200, axon_lambda=800)
>>> model.build()
>>> electrodes = make_argus_ii(center_x=-1000, rotation_deg=-25)
>>> electrodes = encode_image(image, electrodes)
>>> percept = model.predict(electrodes)
"""

__version__ = "0.1.0"

from .axon_map import (
    AxonMapModel,
    Electrode,
    EncodingStrategy,
    GanglionCell,
    compute_ssim,
    encode_image,
    generate_axon_bundles,
    generate_test_images,
    make_argus_ii,
    make_prima,
    render_axon_map,
)
from .constants import (
    BEYELER_SUBJECTS,
    DEG_TO_UM,
    JANSONIUS_OD_X_DEG,
    JANSONIUS_OD_Y_DEG,
)

__all__ = [
    "AxonMapModel",
    "Electrode",
    "EncodingStrategy",
    "GanglionCell",
    "make_argus_ii",
    "make_prima",
    "encode_image",
    "generate_test_images",
    "compute_ssim",
    "render_axon_map",
    "generate_axon_bundles",
    "DEG_TO_UM",
    "JANSONIUS_OD_X_DEG",
    "JANSONIUS_OD_Y_DEG",
    "BEYELER_SUBJECTS",
]
