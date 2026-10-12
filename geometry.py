"""
Focal spot sampling and cone-beam direction sampling.
"""

import numpy as np


def orthonormal_basis(central_direction):
    """
    Build two unit vectors (u, v) perpendicular to central_direction so
    that (u, v, central_direction) form a right-handed orthonormal basis.
    """
    central_direction = central_direction / np.linalg.norm(central_direction)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, central_direction)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    u = np.cross(central_direction, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(central_direction, u)   # already unit length
    return u, v


def sample_focal_spot(source_position, focal_spot_size_mm, central_direction, n_samples, rng=None):
    """
    Sample photon starting positions across a rectangular focal spot,
    treated as a small flat rectangle centered at source_position, lying
    in the plane perpendicular to central_direction. (Simplification of
    the real angled-anode focal spot -- fine for beam geometry purposes.)
    """
    if rng is None:
        rng = np.random.default_rng()

    u, v = orthonormal_basis(central_direction)
    w_mm, h_mm = focal_spot_size_mm

    du = rng.uniform(-w_mm / 2, w_mm / 2, size=n_samples)
    dv = rng.uniform(-h_mm / 2, h_mm / 2, size=n_samples)

    positions = source_position + du[:, None] * u + dv[:, None] * v
    return positions


def sample_directions(central_direction, cone_angle_deg, n_samples, rng=None):
    """
    Sample N unit direction vectors UNIFORMLY OVER THE SOLID ANGLE of a
    cone of half-angle cone_angle_deg around central_direction.

    Correct method (uniform over solid angle, not uniform in angle):
        cos(theta) = 1 - u*(1 - cos(cone_angle))   for u ~ Uniform(0,1)
        phi        = 2*pi*v                        for v ~ Uniform(0,1)
    """
    if rng is None:
        rng = np.random.default_rng()

    central_direction = central_direction / np.linalg.norm(central_direction)
    u_vec, v_vec = orthonormal_basis(central_direction)

    cone_angle_rad = np.deg2rad(cone_angle_deg)
    u = rng.uniform(0, 1, size=n_samples)
    phi = rng.uniform(0, 2 * np.pi, size=n_samples)

    cos_theta = 1 - u * (1 - np.cos(cone_angle_rad))
    sin_theta = np.sqrt(1 - cos_theta ** 2)

    directions = (sin_theta * np.cos(phi))[:, None] * u_vec \
        + (sin_theta * np.sin(phi))[:, None] * v_vec \
        + cos_theta[:, None] * central_direction

    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    return directions
