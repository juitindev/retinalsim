"""
Axon Map Model for Retinal Prosthesis Simulation
=================================================
Implements:
  - Jansonius et al. (2009) Vision Res 49:2157-2163
  - Jansonius et al. (2012) Exp Eye Res 105:70-78
  - Beyeler et al. (2019) Sci Rep 9:9199

All coordinates use a fovea-centered retinal coordinate system
(right eye, fundus view): x+ = nasal, y+ = superior, units = μm.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import logging

from .constants import (
    DEG_TO_UM,
    JANSONIUS_OD_X_DEG,
    JANSONIUS_OD_Y_DEG,
    OD_RIM_RADIUS_DEG,
    ARGUS_II_ROWS,
    ARGUS_II_COLS,
    ARGUS_II_SPACING_UM,
    ARGUS_II_RADIUS_UM,
)

logger = logging.getLogger(__name__)


class EncodingStrategy(Enum):
    """Image-to-current encoding strategies."""
    DIRECT = "direct"
    EDGE = "edge"
    CONTRAST = "contrast"
    SALIENCY = "saliency"


# ═══════════════════════════════════════════════════════════════
# Jansonius Coordinate Transform (2009, Appendix A, Eq.7-10)
# ═══════════════════════════════════════════════════════════════

def _fovea_to_jansonius_polar(x_deg: float, y_deg: float):
    """
    Transform from fovea-centered Cartesian (x, y) in degrees
    to Jansonius modified polar (r, phi) centered on OD.

    Jansonius 2009, Appendix A:
      x' = x - 15                       (Eq.7)
      y' = y - 2*(x/15)^2  for x > 0   (Eq.8a)
      y' = y                otherwise   (Eq.8b)
      r  = sqrt(x'^2 + y'^2)            (Eq.9)
      phi = arctan(y'/x')               (Eq.10)
    """
    xp = x_deg - JANSONIUS_OD_X_DEG
    if x_deg > 0:
        yp = y_deg - JANSONIUS_OD_Y_DEG * (x_deg / JANSONIUS_OD_X_DEG) ** 2
    else:
        yp = y_deg

    r = np.sqrt(xp ** 2 + yp ** 2)
    phi = np.degrees(np.arctan2(yp, xp))
    return r, phi


def _jansonius_polar_to_fovea(r_deg: float, phi_deg: float):
    """
    Inverse transform: Jansonius polar (r, phi) → fovea Cartesian (x, y).
    """
    phi_rad = np.radians(phi_deg)
    xp = r_deg * np.cos(phi_rad)
    yp = r_deg * np.sin(phi_rad)

    x_deg = xp + JANSONIUS_OD_X_DEG

    if x_deg > 0:
        y_deg = yp + JANSONIUS_OD_Y_DEG * (x_deg / JANSONIUS_OD_X_DEG) ** 2
    else:
        y_deg = yp

    return x_deg, y_deg


# ═══════════════════════════════════════════════════════════════
# Jansonius Nerve Fiber Bundle Model
# ═══════════════════════════════════════════════════════════════

def _jansonius_c(phi0: float) -> float:
    """
    Curvature location parameter c as function of φ₀.

    Superior (φ₀ ≥ 0): c = 1.9 + 1.4·tanh{(φ₀ - 121)/14}
        Source: Jansonius 2012 Eq.2; Beyeler 2019 Eq.8 (corrected sign)

    Inferior (φ₀ < 0): c = 1.0 + 0.5·tanh{(-φ₀ - 90)/25}
        Source: Jansonius 2009 Eq.4; 2012 Eq.3
    """
    if phi0 >= 0:
        return 1.9 + 1.4 * np.tanh((phi0 - 121.0) / 14.0)
    else:
        return 1.0 + 0.5 * np.tanh((-phi0 - 90.0) / 25.0)


def _jansonius_b(phi0: float) -> float:
    """
    Curvature amount parameter b as function of φ₀.

    Superior-temporal (φ₀ ≥ 60°):
        ln(b) = -1.9 + 3.9·tanh{-(φ₀ - 121)/14}
        Source: Jansonius 2009 Eq.5; Beyeler 2019 Eq.7 upper branch

    Inferior-temporal (φ₀ ≤ -60°):
        ln(-b) = 0.7 + 1.5·tanh{-(-φ₀ - 90)/25}
        Source: Jansonius 2009 Eq.6; Beyeler 2019 Eq.7 lower branch

    Nasal (-60° < φ₀ < 60°):
        b = 0.00083·φ₀² + 0.020·φ₀ - 2.65
        Source: Jansonius 2012 Eq.8
    """
    if phi0 >= 60.0:
        # Superior-temporal
        return np.exp(-1.9 + 3.9 * np.tanh(-(phi0 - 121.0) / 14.0))
    elif phi0 <= -60.0:
        # Inferior-temporal
        return -np.exp(0.7 + 1.5 * np.tanh(-(-phi0 - 90.0) / 25.0))
    else:
        # Nasal (Jansonius 2012 Eq.8)
        return 0.00083 * phi0 ** 2 + 0.020 * phi0 - 2.65


def _trace_single_axon(
    phi0: float,
    r0: float = OD_RIM_RADIUS_DEG,
    r_max: float = 45.0,
    n_steps: int = 500
) -> np.ndarray:
    """
    Trace one axon bundle from OD outward using Jansonius 2009 Eq.1:

        φ(φ₀, r) = φ₀ + b(φ₀) · (r - r₀)^c(φ₀)

    Returns path in fovea-centered retinal μm, from OD toward periphery.

    Includes raphe termination: fibers are truncated if they cross y=0
    in the temporal retina (x < 0). This fixes the known Jansonius 2012
    papillomacular bundle over-curvature (Section 4.1: φ₀ < -165°).
    """
    b = _jansonius_b(phi0)
    c = _jansonius_c(phi0)

    radii = np.linspace(r0, r_max, n_steps)
    phi_values = phi0 + b * (radii - r0) ** c

    # Convert each (r, phi) point to fovea-centered (x, y) in μm
    path = np.zeros((n_steps, 2))
    valid_len = n_steps  # may be shortened by raphe termination

    # Raphe tolerance: allow small crossing near OD where fibers converge.
    # Only enforce in temporal retina (x < 0) where raphe is well-defined at y=0.
    raphe_tol_um = 200.0  # ~0.7 deg

    for i in range(n_steps):
        x_deg, y_deg = _jansonius_polar_to_fovea(radii[i], phi_values[i])
        x_um = x_deg * DEG_TO_UM
        y_um = y_deg * DEG_TO_UM
        path[i, 0] = x_um
        path[i, 1] = y_um

        # Raphe check: only in temporal retina (x < 0), skip first few points near OD
        if i > 10 and x_um < 0:
            if phi0 < 0 and y_um > raphe_tol_um:
                # Inferior fiber crossed into superior retina → truncate
                valid_len = i
                break
            elif phi0 > 0 and y_um < -raphe_tol_um:
                # Superior fiber crossed into inferior retina → truncate
                valid_len = i
                break

    return path[:valid_len]


def generate_axon_bundles(
    n_bundles: int = 400,
    phi_range: Tuple[float, float] = (-180.0, 180.0),
    r0: float = OD_RIM_RADIUS_DEG,
    r_max: float = 45.0,
    n_steps: int = 500
) -> Tuple[List[np.ndarray], List[float]]:
    """Generate axon bundles covering the full retina.

    Returns
    -------
    bundles : list of (N_i, 2) arrays — axon paths in retinal μm
    phi0s : list of float — starting angle for each bundle (hemifield tag)
    """
    phi_angles = np.linspace(phi_range[0], phi_range[1], n_bundles,
                             endpoint=False)
    bundles = []
    phi0s = []
    for phi0 in phi_angles:
        try:
            path = _trace_single_axon(phi0, r0, r_max, n_steps)
            if len(path) > 2:  # discard degenerate bundles
                bundles.append(path)
                phi0s.append(phi0)
        except (ValueError, RuntimeWarning):
            continue
    return bundles, phi0s


# ═══════════════════════════════════════════════════════════════
# Ganglion Cell + Axon Path Assignment
# ═══════════════════════════════════════════════════════════════

@dataclass
class GanglionCell:
    x: float  # soma x [μm]
    y: float  # soma y [μm]
    axon_path: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    cumulative_lengths: np.ndarray = field(default_factory=lambda: np.empty(0))


def _assign_axon_paths_batch(
    soma_positions: np.ndarray,
    axon_bundles: List[np.ndarray],
    bundle_phi0s: List[float],
    max_dist_um: float = 1500.0
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Batch-assign axon paths for all somas at once.
    Much faster than per-cell loop by vectorizing the nearest-bundle search.

    Includes hemifield constraint: somas in the inferior retina (y < 0)
    are only matched to inferior bundles (φ₀ < 0), and vice versa.
    This prevents cross-raphe assignment — fixes the Jansonius 2012
    papillomacular bundle issue where fibers near ±180° cross the raphe.

    Parameters
    ----------
    soma_positions : (N, 2) array of soma (x, y) in μm
    axon_bundles : list of (M_i, 2) arrays
    bundle_phi0s : list of float, φ₀ for each bundle

    Returns
    -------
    paths : list of (K_i, 2) arrays, soma→OD direction
    cum_lengths : list of (K_i,) arrays
    """
    n_cells = len(soma_positions)

    # Stack all bundle points into one big array for fast lookup
    # Build an index: for each point, which bundle and which index within it
    all_points = []
    bundle_ids = []
    point_ids = []
    for bi, bundle in enumerate(axon_bundles):
        all_points.append(bundle)
        bundle_ids.append(np.full(len(bundle), bi, dtype=np.int32))
        point_ids.append(np.arange(len(bundle), dtype=np.int32))

    all_points = np.vstack(all_points).astype(np.float32)  # (total_pts, 2)
    bundle_ids = np.concatenate(bundle_ids)                 # (total_pts,)
    point_ids = np.concatenate(point_ids)                   # (total_pts,)

    # Precompute hemifield mask per bundle point
    # True = this point belongs to a superior bundle (φ₀ ≥ 0)
    phi0_arr = np.array(bundle_phi0s)
    is_sup_bundle = phi0_arr >= 0  # (n_bundles,)
    is_sup_point = is_sup_bundle[bundle_ids]  # (total_pts,)

    # Raphe blend width: controls the transition zone where both
    # hemifields are allowed. Uses sigmoid for smooth transition.
    # At |y| >> blend_half: strong hemifield preference
    # At |y| << blend_half: both hemifields equally accessible
    # 800μm ≈ 2.7° gives a wide enough zone for smooth visual output.
    raphe_blend_half_um = 800.0

    paths = []
    cum_lens = []

    # Process in chunks to avoid memory explosion
    chunk_size = 200
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        somas = soma_positions[start:end].astype(np.float32)  # (chunk, 2)

        # Distance from each soma to every bundle point
        # (chunk, 1, 2) - (1, total_pts, 2) → (chunk, total_pts)
        dx = somas[:, 0:1] - all_points[None, :, 0]  # broadcast trick
        dy = somas[:, 1:2] - all_points[None, :, 1]
        d_sq = dx ** 2 + dy ** 2  # (chunk, total_pts)

        # ── Hemifield constraint (sigmoid blend) ──
        # Instead of hard cutoff at a fixed y threshold, apply a smooth
        # distance penalty that gradually increases for cross-hemifield
        # bundles. At the raphe (y≈0), both hemifields are allowed.
        big_val = np.float32(1e12)
        for i in range(len(somas)):
            soma_y = float(somas[i, 1])

            # Sigmoid: 0 (deep inferior) → 0.5 (raphe) → 1 (deep superior)
            sup_pref = 1.0 / (1.0 + np.exp(-soma_y / raphe_blend_half_um))

            # Smooth ramp: 0 near raphe, 1 at extremes.
            # Uses linear ramp with soft saturation (power 1.5) for
            # a wider smooth zone than power 2.
            cross_inf = max(0.0, 2.0 * sup_pref - 1.0) ** 1.5
            cross_sup = max(0.0, 1.0 - 2.0 * sup_pref) ** 1.5

            if cross_inf > 1e-6:
                d_sq[i, ~is_sup_point] += np.float32(big_val * cross_inf)
            if cross_sup > 1e-6:
                d_sq[i, is_sup_point] += np.float32(big_val * cross_sup)

        # Find nearest point for each soma
        min_idx = np.argmin(d_sq, axis=1)  # (chunk,)
        min_dist = np.sqrt(np.minimum(d_sq[np.arange(len(somas)), min_idx], 1e10))

        for i in range(len(somas)):
            if min_dist[i] > max_dist_um:
                paths.append(np.array([somas[i]]))
                cum_lens.append(np.array([0.0]))
                continue

            bi = bundle_ids[min_idx[i]]
            pi = point_ids[min_idx[i]]
            bundle = axon_bundles[bi]

            # Bundle: OD(0) → periphery(N). Take [0:pi+1], reverse → soma→OD
            path = bundle[:pi + 1][::-1].copy()
            path[0] = somas[i]  # snap to soma

            if len(path) < 2:
                paths.append(path)
                cum_lens.append(np.array([0.0]))
                continue

            seg = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
            cl = np.zeros(len(path))
            cl[1:] = np.cumsum(seg)
            paths.append(path)
            cum_lens.append(cl)

    return paths, cum_lens


# ═══════════════════════════════════════════════════════════════
# Electrode Arrays
# ═══════════════════════════════════════════════════════════════

@dataclass
class Electrode:
    x: float
    y: float
    current: float = 0.0
    radius: float = 100.0


def _apply_array_transform(
    electrodes: List[Electrode],
    center_x: float = 0.0,
    center_y: float = 0.0,
    rotation_deg: float = 0.0
) -> List[Electrode]:
    """
    Apply translation and rotation to an electrode array.

    Parameters
    ----------
    center_x, center_y : float
        Array center position in retinal μm (fovea = 0,0).
        Beyeler 2019 Table 3 examples:
          Subject 1: (-651, -707)
          Subject 2: (-1331, -850)
          Subject 4: (-1807, 401)
    rotation_deg : float
        Array rotation relative to horizontal raphe [degrees].
        Beyeler 2019 Table 3 examples:
          Subject 1: -49.3°
          Subject 2: -28.4°
          Subject 4: -22.1°
    """
    if center_x == 0.0 and center_y == 0.0 and rotation_deg == 0.0:
        return electrodes

    theta = np.radians(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for e in electrodes:
        # Rotate around (0,0) then translate
        rx = e.x * cos_t - e.y * sin_t
        ry = e.x * sin_t + e.y * cos_t
        e.x = rx + center_x
        e.y = ry + center_y

    return electrodes


def make_argus_ii(
    center_x: float = 0.0,
    center_y: float = 0.0,
    rotation_deg: float = 0.0
) -> List[Electrode]:
    """
    Argus II: 6×10 grid, 200μm diameter, 525μm spacing.
    (Beyeler 2019 p.10)

    Parameters
    ----------
    center_x, center_y : array center in retinal μm
    rotation_deg : rotation relative to horizontal raphe
    """
    electrodes = []
    spacing = ARGUS_II_SPACING_UM
    rows, cols = ARGUS_II_ROWS, ARGUS_II_COLS
    x_off = -(cols - 1) * spacing / 2
    y_off = -(rows - 1) * spacing / 2
    for r in range(rows):
        for c in range(cols):
            electrodes.append(Electrode(
                x=x_off + c * spacing,
                y=y_off + r * spacing,
                radius=ARGUS_II_RADIUS_UM,
            ))
    return _apply_array_transform(electrodes, center_x, center_y, rotation_deg)


def make_prima(
    n_electrodes: int = 378,
    center_x: float = 0.0,
    center_y: float = 0.0,
    rotation_deg: float = 0.0
) -> List[Electrode]:
    """PRIMA: hexagonal, 100μm pitch, 378 pixels."""
    electrodes = []
    pitch = 100.0
    side = int(np.ceil(np.sqrt(n_electrodes)))
    count = 0
    for row in range(side * 2):
        y = (row - side) * pitch * np.sqrt(3) / 2
        x_shift = (pitch / 2) if (row % 2) else 0
        for col in range(side):
            x = (col - side // 2) * pitch + x_shift
            electrodes.append(Electrode(x=x, y=y, radius=35.0))
            count += 1
            if count >= n_electrodes:
                return _apply_array_transform(electrodes, center_x, center_y, rotation_deg)
    return _apply_array_transform(electrodes, center_x, center_y, rotation_deg)


# ═══════════════════════════════════════════════════════════════
# AxonMapModel — Core Simulator
# ═══════════════════════════════════════════════════════════════

@dataclass
class AxonMapModel:
    """
    Beyeler 2019 Axon Map Model.

    Parameters (from Beyeler 2019 Table 2):
        rho:         144-437 μm (radial decay)
        axon_lambda: 500-1420 μm (axonal decay)
    """
    rho: float = 200.0
    axon_lambda: float = 800.0
    grid_spacing: float = 50.0
    x_range: Tuple[float, float] = (-2500.0, 2500.0)
    y_range: Tuple[float, float] = (-2500.0, 2500.0)
    axon_n_bundles: int = 400
    sensitivity_threshold: float = 1e-4

    ganglion_cells: List[GanglionCell] = field(default_factory=list)
    axon_bundles: List[np.ndarray] = field(default_factory=list)
    _bundle_phi0s: List[float] = field(default_factory=list)
    _grid_shape: Tuple[int, int] = (0, 0)
    _built: bool = False

    def build(self):
        logger.info(f"Building AxonMapModel: rho={self.rho}, lambda={self.axon_lambda}")

        self.axon_bundles, self._bundle_phi0s = generate_axon_bundles(
            n_bundles=self.axon_n_bundles
        )
        logger.info(f"  Generated {len(self.axon_bundles)} axon bundles")

        xs = np.arange(self.x_range[0], self.x_range[1] + 1, self.grid_spacing)
        ys = np.arange(self.y_range[0], self.y_range[1] + 1, self.grid_spacing)
        self._grid_shape = (len(ys), len(xs))

        xx, yy = np.meshgrid(xs, ys)
        n_cells = xx.size
        logger.info(f"  Grid: {len(xs)}x{len(ys)} = {n_cells} cells")

        # Assign axon paths — batch vectorized with hemifield constraint
        soma_pos = np.column_stack([xx.ravel(), yy.ravel()])
        logger.info(f"  Assigning axon paths for {n_cells} cells (batch)...")
        paths, cum_lens = _assign_axon_paths_batch(
            soma_pos, self.axon_bundles, self._bundle_phi0s
        )

        # ── Precompute packed arrays for vectorized sensitivity ──
        # Find max path length for padding
        max_len = max(len(p) for p in paths)
        logger.info(f"  Packing axon segments: max_len={max_len}")

        # Padded arrays: (n_cells, max_len, 2) for coords
        # and (n_cells, max_len) for cumulative lengths
        # Mask: True where valid data exists
        self._axon_xy = np.zeros((n_cells, max_len, 2), dtype=np.float32)
        self._axon_l_sq = np.zeros((n_cells, max_len), dtype=np.float32)
        self._axon_mask = np.zeros((n_cells, max_len), dtype=bool)

        for i, (ap, cl) in enumerate(zip(paths, cum_lens)):
            n = len(ap)
            self._axon_xy[i, :n, :] = ap
            self._axon_l_sq[i, :n] = cl ** 2
            self._axon_mask[i, :n] = True

        # Precompute axonal decay (doesn't depend on electrode position)
        two_lam_sq = np.float32(2.0 * self.axon_lambda ** 2)
        self._axonal_weight = np.where(
            self._axon_mask,
            np.exp(-self._axon_l_sq / two_lam_sq),
            0.0
        ).astype(np.float32)

        self._built = True
        logger.info(f"  Built: {n_cells} cells, grid {self._grid_shape}")

    def compute_sensitivity(self, electrode: Electrode, gc_index: int) -> float:
        """Single cell sensitivity (for diagnostics)."""
        e = np.array([electrode.x, electrode.y], dtype=np.float32)
        d_sq = np.sum((self._axon_xy[gc_index] - e) ** 2, axis=1)
        radial = np.exp(-d_sq / (2.0 * self.rho ** 2))
        return float(np.sum(radial * self._axonal_weight[gc_index]))

    def compute_sensitivity_matrix(self, electrodes: List[Electrode]) -> np.ndarray:
        """
        Vectorized sensitivity matrix: S[g, e].

        Strategy: for each electrode, broadcast distance computation
        across ALL cells × ALL segments in one numpy operation.
        """
        assert self._built, "Call .build() first"
        n_gc = self._axon_xy.shape[0]
        n_el = len(electrodes)
        S = np.zeros((n_gc, n_el), dtype=np.float32)

        two_rho_sq = np.float32(2.0 * self.rho ** 2)

        for ei, elec in enumerate(electrodes):
            if ei % 20 == 0:
                logger.info(f"  Electrode {ei}/{n_el}")

            e = np.array([elec.x, elec.y], dtype=np.float32)

            # (n_cells, max_len, 2) - (2,) → (n_cells, max_len, 2)
            delta = self._axon_xy - e
            # (n_cells, max_len)
            d_sq = delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2
            radial = np.exp(-d_sq / two_rho_sq)

            # Multiply by precomputed axonal weights, zero out padding
            # Then sum along segments → (n_cells,)
            S[:, ei] = np.sum(radial * self._axonal_weight, axis=1)

        logger.info(f"  Sensitivity matrix complete.")
        return S

    def simulate(
        self,
        electrodes: List[Electrode],
        sensitivity_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute percept from electrode currents.

        Parameters
        ----------
        electrodes : list of Electrode
            Electrodes with ``.current`` already set.
        sensitivity_matrix : (n_cells, n_electrodes) array, optional
            Precomputed sensitivity matrix. Computed if not provided.

        Returns
        -------
        percept : (rows, cols) array, values in [0, 1]
        """
        assert self._built
        if sensitivity_matrix is None:
            sensitivity_matrix = self.compute_sensitivity_matrix(electrodes)
        currents = np.array([e.current for e in electrodes])
        brightness = sensitivity_matrix @ currents
        percept = brightness.reshape(self._grid_shape)

        # ── Raphe smoothing ──
        # The raphe (y=0) is a physical transition where nerve fiber
        # directions reverse. Axon path assignment creates a discrete
        # switch at this boundary. We smooth the percept along the y-axis
        # in a narrow band around y=0 to approximate the gradual fiber
        # density transition that occurs anatomically.
        percept = self._smooth_raphe(percept)

        if percept.max() > 0:
            percept = percept / percept.max()
        return percept

    def _smooth_raphe(self, percept: np.ndarray) -> np.ndarray:
        """Apply 1D Gaussian smoothing along y near the raphe (y=0).

        The smoothing kernel width adapts to ρ: larger ρ means the
        electrode activates more distant axons, amplifying the raphe
        discontinuity, so more smoothing is needed.
        """
        rows, cols = percept.shape
        ys = np.arange(self.y_range[0], self.y_range[1] + 1, self.grid_spacing)
        if len(ys) != rows:
            ys = np.linspace(self.y_range[0], self.y_range[1], rows)

        # Smoothing kernel width: 2–4 grid cells, scaled by ρ
        # At ρ=100: σ ≈ 1.5 cells. At ρ=500: σ ≈ 3.5 cells.
        sigma_cells = max(1.5, min(4.0, self.rho / 150.0))
        kernel_half = int(np.ceil(sigma_cells * 2.5))

        # Build 1D Gaussian kernel
        kx = np.arange(-kernel_half, kernel_half + 1, dtype=np.float64)
        kernel = np.exp(-kx**2 / (2 * sigma_cells**2))
        kernel /= kernel.sum()

        # Find the raphe row (y closest to 0)
        raphe_row = np.argmin(np.abs(ys))

        # Blend zone: rows within ±(kernel_half + margin) of raphe
        blend_margin = kernel_half + 2
        r_lo = max(0, raphe_row - blend_margin)
        r_hi = min(rows, raphe_row + blend_margin + 1)

        # Apply 1D convolution along y for each column in the blend zone
        smoothed = percept.copy()
        for c in range(cols):
            col_data = percept[:, c]
            for r in range(r_lo, r_hi):
                # Weighted average of nearby rows
                total = 0.0
                weight_sum = 0.0
                for k in range(-kernel_half, kernel_half + 1):
                    rr = r + k
                    if 0 <= rr < rows:
                        w = kernel[k + kernel_half]
                        total += col_data[rr] * w
                        weight_sum += w
                if weight_sum > 0:
                    smoothed[r, c] = total / weight_sum

        return smoothed

    def predict(
        self,
        image: np.ndarray,
        electrodes: Optional[List[Electrode]] = None,
        encoding: str = "direct",
        current_amplitude: float = 1.0,
        array_type: str = "argus_ii",
        center_x: float = -1000.0,
        center_y: float = 0.0,
        rotation_deg: float = -25.0,
    ) -> np.ndarray:
        """End-to-end prediction: image → percept.

        Convenience method that creates electrodes, encodes the image,
        and runs the simulation in one call.

        Parameters
        ----------
        image : 2D array
            Grayscale input image, values in [0, 1] or [0, 255].
        electrodes : list of Electrode, optional
            Pre-configured electrodes. If None, creates from array_type.
        encoding : str
            Encoding strategy: 'direct', 'edge', 'contrast', 'saliency'.
        current_amplitude : float
            Maximum current scaling factor.
        array_type : str
            'argus_ii' or 'prima'. Ignored if electrodes provided.
        center_x, center_y : float
            Array center in retinal μm. Ignored if electrodes provided.
        rotation_deg : float
            Array rotation in degrees. Ignored if electrodes provided.

        Returns
        -------
        percept : (rows, cols) array, values in [0, 1]

        Example
        -------
        >>> model = AxonMapModel(rho=200, axon_lambda=800)
        >>> model.build()
        >>> percept = model.predict(image, encoding='edge')
        """
        assert self._built, "Call .build() first"

        if electrodes is None:
            if array_type == "prima":
                electrodes = make_prima(center_x, center_y, rotation_deg)
            else:
                electrodes = make_argus_ii(center_x, center_y, rotation_deg)

        strategy = EncodingStrategy(encoding)
        electrodes = encode_image(
            image, electrodes, strategy,
            current_amplitude=current_amplitude,
        )
        return self.simulate(electrodes)


# ═══════════════════════════════════════════════════════════════
# Encoding Strategies
# ═══════════════════════════════════════════════════════════════

def encode_image(
    image: np.ndarray,
    electrodes: List[Electrode],
    strategy: EncodingStrategy = EncodingStrategy.DIRECT,
    image_fov_um: float = 5000.0,
    current_amplitude: float = 1.0
) -> List[Electrode]:
    """
    Map image to electrode currents.
    Uses area averaging (not single pixel) to avoid aliasing artifacts.
    Each electrode samples a patch proportional to its radius.

    The image FOV is centered on the electrode array center (not fovea),
    matching real prosthesis behavior where the camera image maps to
    the array's coverage area.
    """
    import cv2

    h, w = image.shape[:2]
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    elif image.max() > 1.0:
        image = image.astype(float) / 255.0

    if strategy == EncodingStrategy.DIRECT:
        encoded = image
    elif strategy == EncodingStrategy.EDGE:
        edges = cv2.Canny((image * 255).astype(np.uint8), 50, 150)
        encoded = edges.astype(float) / 255.0
    elif strategy == EncodingStrategy.CONTRAST:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((image * 255).astype(np.uint8))
        encoded = enhanced.astype(float) / 255.0
    elif strategy == EncodingStrategy.SALIENCY:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        encoded = mag / (mag.max() + 1e-8)
    else:
        encoded = image

    # Center image FOV on the electrode array center
    arr_cx = np.mean([e.x for e in electrodes])
    arr_cy = np.mean([e.y for e in electrodes])
    half_fov = image_fov_um / 2.0

    # Patch radius in pixels: half the electrode radius (100μm)
    # This averages over the electrode's physical area, not more
    patch_radius_um = 100.0  # sample a 200μm diameter patch = electrode size
    patch_r_px = max(1, int(patch_radius_um / image_fov_um * w))

    for elec in electrodes:
        # Map electrode position relative to array center → image pixel
        px = ((elec.x - arr_cx) + half_fov) / image_fov_um * (w - 1)
        py = (half_fov - (elec.y - arr_cy)) / image_fov_um * (h - 1)

        # Area average: sample a square patch around (px, py)
        x0 = int(np.clip(px - patch_r_px, 0, w - 1))
        x1 = int(np.clip(px + patch_r_px + 1, 0, w))
        y0 = int(np.clip(py - patch_r_px, 0, h - 1))
        y1 = int(np.clip(py + patch_r_px + 1, 0, h))

        if x1 > x0 and y1 > y0:
            elec.current = float(np.mean(encoded[y0:y1, x0:x1])) * current_amplitude
        else:
            elec.current = 0.0

    return electrodes


# ═══════════════════════════════════════════════════════════════
# SSIM
# ═══════════════════════════════════════════════════════════════

def compute_ssim(original: np.ndarray, simulated: np.ndarray) -> float:
    import cv2
    if original.shape != simulated.shape:
        simulated = cv2.resize(simulated, (original.shape[1], original.shape[0]))
    img1 = original.astype(np.float64)
    img2 = simulated.astype(np.float64)
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1 ** 2
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2 ** 2
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


# ═══════════════════════════════════════════════════════════════
# Test Images
# ═══════════════════════════════════════════════════════════════

def generate_test_images(size: int = 256) -> Dict[str, np.ndarray]:
    patterns = {}

    # Letter E — moderate stroke, clear gaps between bars
    img = np.zeros((size, size))
    t = size // 8  # moderate stroke (~32px at 256)
    left = size // 3
    right = 2 * size // 3
    # Three horizontal bars with clear spacing
    top_y = size // 5
    mid_y = size // 2 - t // 2
    bot_y = 4 * size // 5 - t
    img[top_y:top_y + t, left:right] = 1.0      # top bar
    img[mid_y:mid_y + t, left:right] = 1.0      # middle bar
    img[bot_y:bot_y + t, left:right] = 1.0      # bottom bar
    img[top_y:bot_y + t, left:left + t] = 1.0   # vertical bar
    patterns['letter_E'] = img

    # Grating
    x = np.linspace(0, 8 * np.pi, size)
    patterns['grating'] = (np.sin(x[None, :].repeat(size, 0)) + 1) / 2

    # Dot superior — single bright dot in upper retina
    img = np.zeros((size, size))
    yy, xx = np.mgrid[:size, :size]
    cx, cy = size // 2, size // 4  # center x, upper quarter
    radius = size // 12
    dot = ((xx - cx)**2 + (yy - cy)**2) < radius**2
    img[dot] = 1.0
    patterns['dot_superior'] = img

    # Dot inferior — single bright dot in lower retina
    img = np.zeros((size, size))
    cx, cy = size // 2, 3 * size // 4  # center x, lower quarter
    dot = ((xx - cx)**2 + (yy - cy)**2) < radius**2
    img[dot] = 1.0
    patterns['dot_inferior'] = img

    return patterns


# ═══════════════════════════════════════════════════════════════
# Axon Map Visualization
# ═══════════════════════════════════════════════════════════════

def render_axon_map(
    axon_bundles: List[np.ndarray],
    bundle_phi0s: List[float],
    electrodes: Optional[List[Electrode]] = None,
    figsize: Tuple[float, float] = (5, 5),
    view_range_um: float = 6000.0,
) -> np.ndarray:
    """
    Render axon bundle paths + electrode positions as an image array.

    Uses the model's actual computed bundles — same data used in simulation.

    Returns
    -------
    RGBA image as uint8 numpy array (for conversion to base64 PNG)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse
    import io

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='#111111')
    ax.set_facecolor('#111111')
    ax.set_xlim(-view_range_um, view_range_um)
    ax.set_ylim(-view_range_um, view_range_um)
    ax.set_aspect('equal')
    ax.axis('off')

    # ── Draw axon bundles colored by fiber type ──
    for bundle, phi0 in zip(axon_bundles, bundle_phi0s):
        if len(bundle) < 2:
            continue
        abs_phi = abs(phi0)

        if abs_phi > 160:
            color = (0.75, 0.55, 0.9, 0.5)   # purple — papillomacular
            lw = 0.6
        elif phi0 >= 60:
            color = (0.9, 0.45, 0.2, 0.3)    # orange — superior arcade
            lw = 0.4
        elif phi0 <= -60:
            color = (0.2, 0.6, 0.9, 0.3)     # blue — inferior arcade
            lw = 0.4
        else:
            color = (0.55, 0.55, 0.5, 0.2)   # gray — nasal
            lw = 0.3

        ax.plot(bundle[:, 0], bundle[:, 1], color=color, linewidth=lw,
                solid_capstyle='round')

    # ── OD ──
    od_x = JANSONIUS_OD_X_DEG * DEG_TO_UM
    od_y = JANSONIUS_OD_Y_DEG * DEG_TO_UM
    od_r = OD_RIM_RADIUS_DEG * DEG_TO_UM
    od = Ellipse((od_x, od_y), od_r * 2, od_r * 1.5,
                 facecolor=(0.9, 0.8, 0.5, 0.2),
                 edgecolor=(0.8, 0.65, 0.3, 0.8), linewidth=1.5)
    ax.add_patch(od)
    ax.text(od_x, od_y, 'OD', color='#cccccc', fontsize=10,
            ha='center', va='center', fontweight='bold')

    # ── Fovea ──
    ax.plot(0, 0, 'o', color='white', markersize=4, zorder=10)
    ax.text(0, -300, 'fovea', color='#cccccc', fontsize=8,
            ha='center', va='top')

    # ── Electrodes ──
    if electrodes:
        xs = [e.x for e in electrodes]
        ys = [e.y for e in electrodes]

        # Array bounding box
        margin = 200
        ax.plot([min(xs) - margin, max(xs) + margin, max(xs) + margin,
                 min(xs) - margin, min(xs) - margin],
                [min(ys) - margin, min(ys) - margin, max(ys) + margin,
                 max(ys) + margin, min(ys) - margin],
                color=(0, 1, 0.7, 0.4), linewidth=1)

        # Individual electrodes
        for e in electrodes:
            circ = Circle((e.x, e.y), e.radius,
                          facecolor=(0, 1, 0.7, 0.5),
                          edgecolor='none')
            ax.add_patch(circ)

        ax.text(min(xs) - margin, max(ys) + margin + 150,
                'Argus II', color=(0, 1, 0.7, 0.9), fontsize=9,
                fontweight='bold')

    # ── Labels ──
    ax.text(0, view_range_um - 200, 'superior', color='#888888',
            fontsize=9, ha='center', va='top')
    ax.text(0, -view_range_um + 200, 'inferior', color='#888888',
            fontsize=9, ha='center', va='bottom')
    ax.text(-view_range_um + 100, 0, 'temporal', color='#888888',
            fontsize=9, ha='left', va='center')
    ax.text(view_range_um - 100, 0, 'nasal', color='#888888',
            fontsize=9, ha='right', va='center')

    # ── Legend ──
    legend_items = [
        ((0.9, 0.45, 0.2), 'Superior arcade'),
        ((0.2, 0.6, 0.9), 'Inferior arcade'),
        ((0.75, 0.55, 0.9), 'Papillomacular'),
        ((0.55, 0.55, 0.5), 'Nasal'),
    ]
    for i, (c, label) in enumerate(legend_items):
        y_pos = view_range_um - 400 - i * 400
        ax.plot([-view_range_um + 200, -view_range_um + 600],
                [y_pos, y_pos], color=c, linewidth=2)
        ax.text(-view_range_um + 700, y_pos, label,
                color='#aaaaaa', fontsize=7, va='center')

    fig.tight_layout(pad=0.3)

    # Convert to base64 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1,
                dpi=100, facecolor='#111111')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()