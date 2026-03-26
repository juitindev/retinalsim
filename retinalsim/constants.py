"""
Physical and model constants for retinal prosthesis simulation.

References
----------
- Drasdo & Fowler (1974): retinal distance conversion
- Rohrschneider (2004): optic disc location
- Jansonius et al. (2009, 2012): RNFB trajectory model
- Beyeler et al. (2019): axon map model, subject parameters
"""

# Retinal distance conversion (Drasdo & Fowler 1974)
DEG_TO_UM = 301.5  # μm per degree of visual angle

# Optic disc center in Jansonius model coordinates
# (Jansonius 2009 p.3, 2012 p.72)
JANSONIUS_OD_X_DEG = 15.0   # degrees nasal from fovea
JANSONIUS_OD_Y_DEG = 2.0    # degrees superior from fovea

# Optic disc rim radius (Jansonius 2009)
OD_RIM_RADIUS_DEG = 4.0

# Argus II hardware (Beyeler 2019 p.10)
ARGUS_II_ROWS = 6
ARGUS_II_COLS = 10
ARGUS_II_N_ELECTRODES = 60
ARGUS_II_SPACING_UM = 525.0   # center-to-center
ARGUS_II_DIAMETER_UM = 200.0  # electrode diameter
ARGUS_II_RADIUS_UM = 100.0    # electrode radius

# Beyeler 2019 Table 2 + Table 3: fitted subject parameters
# Note: Subject 1 used Argus I (4×4, 16 electrodes) — not compatible
# with Argus II simulation. Subjects 2–4 used Argus II.
BEYELER_SUBJECTS = {
    "S1": {
        "rho": 410, "axon_lambda": 1190,
        "array_x": -651, "array_y": -707, "rotation": -49.3,
        "implant": "Argus I",
        "note": "Argus I (4×4, 16 electrodes) — use S2–S4 for Argus II",
    },
    "S2": {
        "rho": 315, "axon_lambda": 500,
        "array_x": -1331, "array_y": -850, "rotation": -28.4,
        "implant": "Argus II",
    },
    "S3": {
        "rho": 144, "axon_lambda": 1414,
        "array_x": -2142, "array_y": 102, "rotation": -53.9,
        "implant": "Argus II",
    },
    "S4": {
        "rho": 437, "axon_lambda": 1420,
        "array_x": -1807, "array_y": 401, "rotation": -22.1,
        "implant": "Argus II",
    },
}
