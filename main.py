"""
X-RAY SOURCE MODULE -- main entry point.

Run:
    python main.py

======================================================================
SOURCE MODULE -> DETECTOR MODULE BOUNDARY
======================================================================
This script produces exactly four arrays (see the bottom of this file):

    photon_positions   shape (N_samples, 3)
    photon_directions  shape (N_samples, 3)
    photon_energies    shape (N_samples,)
    photon_weights     shape (N_samples,)

Everything after that -- detector response, scintillator, electronic
noise, reconstruction -- belongs to the detector team, not this module.
"""

import numpy as np

import config
from spectrum import generate_spectrum, apply_filtration, sample_photon_energies
from geometry import sample_focal_spot, sample_directions
from weights import calculate_photon_weights, apply_bowtie, apply_heel_effect, add_poisson_noise
from flat_field import generate_flat_field
from validate import validate, validate_current_and_exposure_scaling


def run_source_simulation(cfg=config, rng_seed=0):
    rng = np.random.default_rng(rng_seed)

    central_direction = cfg.DETECTOR_POSITION - cfg.SOURCE_POSITION
    central_direction = central_direction / np.linalg.norm(central_direction)

    # 1) Spectrum: unfiltered -> filtered through the Al filtration
    energies_grid, intensity_unfiltered = generate_spectrum(cfg.KVP, cfg.E_MIN_KEV, cfg.E_STEP_KEV)
    intensity_filtered = apply_filtration(energies_grid, intensity_unfiltered, cfg.AL_FILTRATION_MM)

    # 2) Sample each simulated photon's energy from the filtered spectrum
    photon_energies = sample_photon_energies(energies_grid, intensity_filtered, cfg.N_SAMPLES, rng)

    # 3) Sample each simulated photon's starting position across the focal spot
    photon_positions = sample_focal_spot(cfg.SOURCE_POSITION, cfg.FOCAL_SPOT_SIZE,
                                          central_direction, cfg.N_SAMPLES, rng)

    # 4) Sample each simulated photon's direction inside the cone beam
    photon_directions = sample_directions(central_direction, cfg.CONE_ANGLE_DEG, cfg.N_SAMPLES, rng)

    # 5) Base weight per photon, from mA * exposure_time (mAs) -- N_samples never changes
    photon_weights = calculate_photon_weights(cfg.MA, cfg.EXPOSURE_TIME, cfg.N_SAMPLES)

    # 6) Bowtie filter: further reduces weight depending on angle from beam center
    photon_weights, bowtie_thickness, bowtie_transmission = apply_bowtie(
        photon_directions, central_direction, photon_energies, photon_weights,
        cfg.CONE_ANGLE_DEG, cfg.BOWTIE_MIN_THICKNESS_MM, cfg.BOWTIE_MAX_THICKNESS_MM,
    )

    # 7) Optional heel effect (disabled by default; core project does not need it)
    photon_weights = apply_heel_effect(
        photon_directions, central_direction, photon_weights,
        cfg.ANODE_ANGLE_DEG, heel_axis=np.array([1.0, 0.0, 0.0]),
        enable=cfg.ENABLE_HEEL_EFFECT,
    )

    # 8) Optional Poisson statistics on photon NUMBER (not to be confused with
    #    the random ENERGY sampling in step 2). See note below.
    photon_weights = add_poisson_noise(photon_weights, enable=cfg.ENABLE_POISSON_ON_WEIGHTS, rng=rng)

    # 9) No-object / flat-field projection at the detector plane (visualization only)
    flat_field_image = generate_flat_field(
        photon_positions, photon_directions, photon_weights,
        cfg.SOURCE_POSITION, cfg.DETECTOR_POSITION,
    )

    return {
        "energies_grid": energies_grid,
        "intensity_unfiltered": intensity_unfiltered,
        "intensity_filtered": intensity_filtered,
        "photon_positions": photon_positions,
        "photon_directions": photon_directions,
        "photon_energies": photon_energies,
        "photon_weights": photon_weights,
        "bowtie_thickness": bowtie_thickness,
        "bowtie_transmission": bowtie_transmission,
        "flat_field": flat_field_image,
    }


if __name__ == "__main__":
    results = run_source_simulation()

    validate(results, config, save_dir=".")
    validate_current_and_exposure_scaling(calculate_photon_weights, save_dir=".")

    print("\n--- SOURCE MODULE OUTPUT (-> DETECTOR MODULE) ---")
    print("photon_positions :", results["photon_positions"].shape)
    print("photon_directions:", results["photon_directions"].shape)
    print("photon_energies  :", results["photon_energies"].shape)
    print("photon_weights   :", results["photon_weights"].shape)

    # These four arrays are everything the detector team needs from this module.
    photon_positions = results["photon_positions"]
    photon_directions = results["photon_directions"]
    photon_energies = results["photon_energies"]
    photon_weights = results["photon_weights"]

    
    # Save all four arrays to a CSV file

    data = np.column_stack((
        photon_positions,
        photon_directions,
        photon_energies,
        photon_weights
    ))

    np.savetxt(
        "source_output.csv",
        data,
        delimiter=",",
        header="pos_x,pos_y,pos_z,dir_x,dir_y,dir_z,energy,weight",
        comments=""
    )

    print("\nCSV file saved as source_output.csv")

