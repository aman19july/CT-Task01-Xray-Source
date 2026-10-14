"""
Configuration / parameters for the X-ray SOURCE simulation.

Two different kinds of parameters, kept clearly separate:

1) PHYSICAL parameters: kVp, mA, exposure_time, focal spot, filtration,
   geometry. These describe the real tube/beam.

2) SIMULATION parameter: N_SAMPLES. This is NOT physical -- it is just how
   many representative rays the Monte Carlo simulation uses. It stays fixed
   regardless of mA/exposure_time; physical intensity is represented with
   photon WEIGHTS instead (see weights.py).
"""

import numpy as np

# ---------------------------------------------------------------
# SIMULATION PARAMETER (not physical) -- kept fixed
# ---------------------------------------------------------------
N_SAMPLES = 56,832

# ---------------------------------------------------------------
# PHYSICAL TUBE PARAMETERS (edit these to run different scenarios)
# Reasonable ranges based on Siemens SOMATOM Force reference data:
#   kVp: 70-150 kV, mA: 20-1300 mA, focal spots: 0.4x0.5 / 0.8x1.1 mm
# ---------------------------------------------------------------
KVP = 120.0                  # tube voltage, kV
MA = 200.0                   # tube current, mA
EXPOSURE_TIME = 1          # seconds
AL_FILTRATION_MM = 4.0       # aluminum filtration thickness, mm

FOCAL_SPOT_SMALL = (0.4, 0.5)   # (width_mm, height_mm)
FOCAL_SPOT_LARGE = (0.8, 1.1)
FOCAL_SPOT_SIZE = FOCAL_SPOT_SMALL     # active choice; custom e.g. (0.6, 0.6) allowed

# ---------------------------------------------------------------
# GEOMETRY (normally supplied by the detector/geometry team).
# Example values based on Siemens SOMATOM Force source-isocenter/detector
# distances -- your real values will come from that team, not from here.
# ---------------------------------------------------------------
SOURCE_POSITION = np.array([0.0, 0.0, 0.0])          # mm
DETECTOR_POSITION = np.array([0.0, 0.0, 976.0])      # mm
CONE_ANGLE_DEG = 12.0        # half-angle of the cone beam, degrees

# ---------------------------------------------------------------
# BOWTIE FILTER (simple analytical model, aluminum)
# ---------------------------------------------------------------
BOWTIE_MIN_THICKNESS_MM = 0.5   # thickness at beam center
BOWTIE_MAX_THICKNESS_MM = 8.0   # thickness at edge of the cone

# ---------------------------------------------------------------
# HEEL EFFECT -- OPTIONAL, disabled by default. Core project does not
# depend on this.
# ---------------------------------------------------------------
ENABLE_HEEL_EFFECT = True
ANODE_ANGLE_DEG = 7.0

# ---------------------------------------------------------------
# POISSON NOISE ON SOURCE PHOTON COUNT -- OPTIONAL, see notes in main.py
# ---------------------------------------------------------------
ENABLE_POISSON_ON_WEIGHTS = True

# ---------------------------------------------------------------
# ENERGY GRID for the spectrum model
# ---------------------------------------------------------------
E_MIN_KEV = 10.0
E_STEP_KEV = 1.0
