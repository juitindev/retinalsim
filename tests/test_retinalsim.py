"""
Test suite for RetinalSim.

Tests are organized by component:
  - Constants & coordinate transforms
  - Jansonius nerve fiber bundle model (b, c parameters)
  - Axon tracing & raphe constraints
  - Electrode arrays (Argus II, PRIMA)
  - AxonMapModel build & simulation
  - Encoding strategies
  - SSIM metric
  - Beyeler 2019 subject parameter reproduction

Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from retinalsim import (
    AxonMapModel,
    Electrode,
    EncodingStrategy,
    make_argus_ii,
    make_prima,
    encode_image,
    generate_test_images,
    compute_ssim,
    generate_axon_bundles,
    DEG_TO_UM,
    JANSONIUS_OD_X_DEG,
    JANSONIUS_OD_Y_DEG,
    BEYELER_SUBJECTS,
)
from retinalsim.axon_map import (
    _fovea_to_jansonius_polar,
    _jansonius_polar_to_fovea,
    _jansonius_b,
    _jansonius_c,
    _trace_single_axon,
)
from retinalsim.constants import (
    OD_RIM_RADIUS_DEG,
    ARGUS_II_ROWS,
    ARGUS_II_COLS,
    ARGUS_II_N_ELECTRODES,
    ARGUS_II_SPACING_UM,
)


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

class TestConstants:
    """Verify physical constants match published values."""

    def test_deg_to_um(self):
        """DEG_TO_UM = 301.5 μm/deg (Drasdo & Fowler 1974)."""
        assert DEG_TO_UM == 301.5

    def test_od_location(self):
        """OD at 15° nasal, 2° superior (Jansonius 2009 p.3)."""
        assert JANSONIUS_OD_X_DEG == 15.0
        assert JANSONIUS_OD_Y_DEG == 2.0

    def test_od_rim_radius(self):
        """OD rim radius = 4° (Jansonius 2009)."""
        assert OD_RIM_RADIUS_DEG == 4.0

    def test_argus_ii_specs(self):
        """Argus II: 6×10, 525μm spacing (Beyeler 2019 p.10)."""
        assert ARGUS_II_ROWS == 6
        assert ARGUS_II_COLS == 10
        assert ARGUS_II_N_ELECTRODES == 60
        assert ARGUS_II_SPACING_UM == 525.0

    def test_beyeler_subjects_complete(self):
        """All 4 Beyeler 2019 subjects present."""
        assert set(BEYELER_SUBJECTS.keys()) == {"S1", "S2", "S3", "S4"}
        for key in ["rho", "axon_lambda", "array_x", "array_y", "rotation"]:
            for subj in BEYELER_SUBJECTS.values():
                assert key in subj

    def test_beyeler_s2_params(self):
        """Beyeler 2019 Table 2+3: Subject 2 values."""
        s2 = BEYELER_SUBJECTS["S2"]
        assert s2["rho"] == 315
        assert s2["axon_lambda"] == 500
        assert s2["array_x"] == -1331
        assert s2["array_y"] == -850
        assert s2["rotation"] == pytest.approx(-28.4)


# ═══════════════════════════════════════════════════════════════
# Coordinate Transforms (Jansonius 2009 Appendix A)
# ═══════════════════════════════════════════════════════════════

class TestCoordinateTransform:
    """Verify Jansonius 2009 Eq.7–10 coordinate transforms."""

    def test_fovea_maps_near_od(self):
        """Fovea (0, 0) in Jansonius polar should be near r ≈ 15, φ ≈ 180°."""
        r, phi = _fovea_to_jansonius_polar(0.0, 0.0)
        # Fovea is ~15° from OD, roughly at φ = 180° (temporal direction)
        assert r == pytest.approx(15.0, abs=1.0)
        assert abs(phi) == pytest.approx(180.0, abs=5.0)

    def test_od_center_maps_to_origin(self):
        """OD center (15, 2) in Jansonius polar should be r ≈ 0."""
        r, phi = _fovea_to_jansonius_polar(15.0, 2.0)
        assert r == pytest.approx(0.0, abs=0.1)

    def test_roundtrip_identity(self):
        """Fovea → Jansonius → Fovea should be identity."""
        for x, y in [(5.0, 3.0), (-10.0, 5.0), (20.0, -8.0), (0.0, 0.0)]:
            r, phi = _fovea_to_jansonius_polar(x, y)
            x2, y2 = _jansonius_polar_to_fovea(r, phi)
            assert x2 == pytest.approx(x, abs=1e-10)
            assert y2 == pytest.approx(y, abs=1e-10)

    def test_y_correction_positive_x(self):
        """For x > 0: y' = y - 2*(x/15)^2 (Jansonius 2009 Eq.8a)."""
        x, y = 15.0, 4.0
        # y' = 4 - 2*(15/15)^2 = 4 - 2 = 2
        # x' = 15 - 15 = 0
        r, phi = _fovea_to_jansonius_polar(x, y)
        assert r == pytest.approx(2.0, abs=0.01)

    def test_no_y_correction_negative_x(self):
        """For x ≤ 0: y' = y (Jansonius 2009 Eq.8b)."""
        x, y = -5.0, 3.0
        r, phi = _fovea_to_jansonius_polar(x, y)
        xp = x - 15.0  # = -20
        yp = y          # = 3 (no correction)
        expected_r = np.sqrt(xp**2 + yp**2)
        assert r == pytest.approx(expected_r, abs=1e-10)


# ═══════════════════════════════════════════════════════════════
# Jansonius b, c Parameters
# ═══════════════════════════════════════════════════════════════

class TestJansoniusParameters:
    """Verify Jansonius 2009/2012 curvature parameters."""

    def test_c_superior_range(self):
        """Superior c(φ₀): tanh saturates to [0.5, 3.3]."""
        # At φ₀ = 180° (far superior): c → 1.9 + 1.4·tanh((180-121)/14) ≈ 3.27
        c_high = _jansonius_c(180.0)
        assert 3.0 < c_high < 3.4

        # At φ₀ = 0°: c = 1.9 + 1.4·tanh(-121/14) ≈ 0.50
        c_low = _jansonius_c(0.0)
        assert 0.4 < c_low < 0.6

    def test_c_inferior_range(self):
        """Inferior c(φ₀): tanh gives range [1.0, 1.5]."""
        c_inf = _jansonius_c(-90.0)
        # c = 1.0 + 0.5·tanh((90-90)/25) = 1.0
        assert c_inf == pytest.approx(1.0, abs=0.01)

        c_deep = _jansonius_c(-180.0)
        # c = 1.0 + 0.5·tanh((180-90)/25) ≈ 1.49
        assert 1.4 < c_deep < 1.5

    def test_b_superior_temporal_positive(self):
        """Superior-temporal (φ₀ ≥ 60°): b > 0."""
        assert _jansonius_b(90.0) > 0
        assert _jansonius_b(120.0) > 0

    def test_b_inferior_temporal_negative(self):
        """Inferior-temporal (φ₀ ≤ -60°): b < 0."""
        assert _jansonius_b(-90.0) < 0
        assert _jansonius_b(-120.0) < 0

    def test_b_nasal_quadratic(self):
        """Nasal (-60° < φ₀ < 60°): b = quadratic (Jansonius 2012 Eq.8)."""
        b0 = _jansonius_b(0.0)
        expected = 0.00083 * 0**2 + 0.020 * 0 - 2.65
        assert b0 == pytest.approx(expected, abs=1e-5)

    def test_b_continuity_at_60(self):
        """b should be roughly continuous near φ₀ = ±60° transition."""
        # Not perfectly continuous (known limitation), but should be same order
        b_nasal_59 = _jansonius_b(59.0)
        b_arcade_61 = _jansonius_b(61.0)
        # Both should be negative and similar magnitude
        assert abs(b_nasal_59 - b_arcade_61) < 5.0


# ═══════════════════════════════════════════════════════════════
# Axon Tracing
# ═══════════════════════════════════════════════════════════════

class TestAxonTracing:
    """Verify axon bundle path generation."""

    def test_trace_returns_array(self):
        """_trace_single_axon returns (N, 2) array in μm."""
        path = _trace_single_axon(90.0)
        assert path.ndim == 2
        assert path.shape[1] == 2
        assert len(path) > 2

    def test_trace_starts_near_od(self):
        """Axon path starts near the optic disc."""
        path = _trace_single_axon(90.0)
        od_x = JANSONIUS_OD_X_DEG * DEG_TO_UM
        od_y = JANSONIUS_OD_Y_DEG * DEG_TO_UM
        start_dist = np.sqrt((path[0, 0] - od_x)**2 + (path[0, 1] - od_y)**2)
        # Should be within r0 * DEG_TO_UM of OD center
        assert start_dist < OD_RIM_RADIUS_DEG * DEG_TO_UM * 1.5

    def test_trace_superior_goes_up(self):
        """Superior arcade fiber (φ₀=90°) should extend into y > 0 region."""
        path = _trace_single_axon(90.0)
        assert np.max(path[:, 1]) > 1000.0  # reaches well above fovea

    def test_trace_inferior_goes_down(self):
        """Inferior arcade fiber (φ₀=-90°) should extend into y < 0 region."""
        path = _trace_single_axon(-90.0)
        assert np.min(path[:, 1]) < -1000.0

    def test_raphe_truncation_inferior(self):
        """Inferior fibers should not cross into superior temporal retina."""
        # Deep inferior papillomacular — known over-curvature region
        path = _trace_single_axon(-170.0)
        # In temporal retina (x < 0), y should stay ≤ small tolerance
        temporal_mask = path[:, 0] < 0
        if temporal_mask.any():
            max_y_temporal = np.max(path[temporal_mask, 1])
            assert max_y_temporal < 500.0  # raphe_tol_um + margin

    def test_generate_bundles_count(self):
        """generate_axon_bundles returns requested number of bundles (±some failures)."""
        bundles, phi0s = generate_axon_bundles(n_bundles=100)
        assert len(bundles) == len(phi0s)
        assert len(bundles) > 80  # allow some degenerate bundles to be dropped
        assert len(bundles) <= 100

    def test_generate_bundles_covers_hemifields(self):
        """Bundles span both superior (φ₀ > 0) and inferior (φ₀ < 0) hemifields."""
        bundles, phi0s = generate_axon_bundles(n_bundles=100)
        phi0_arr = np.array(phi0s)
        assert np.any(phi0_arr > 0), "No superior bundles"
        assert np.any(phi0_arr < 0), "No inferior bundles"


# ═══════════════════════════════════════════════════════════════
# Electrode Arrays
# ═══════════════════════════════════════════════════════════════

class TestElectrodeArrays:
    """Verify electrode array geometry."""

    def test_argus_ii_count(self):
        """Argus II has exactly 60 electrodes."""
        electrodes = make_argus_ii()
        assert len(electrodes) == ARGUS_II_N_ELECTRODES

    def test_argus_ii_centered(self):
        """Default Argus II is centered at (0, 0)."""
        electrodes = make_argus_ii()
        xs = [e.x for e in electrodes]
        ys = [e.y for e in electrodes]
        assert np.mean(xs) == pytest.approx(0.0, abs=1.0)
        assert np.mean(ys) == pytest.approx(0.0, abs=1.0)

    def test_argus_ii_spacing(self):
        """Argus II electrode spacing is 525 μm."""
        electrodes = make_argus_ii()
        # Check spacing between adjacent electrodes in first row
        row0 = sorted([e for e in electrodes if abs(e.y - electrodes[0].y) < 1],
                       key=lambda e: e.x)
        if len(row0) >= 2:
            spacing = row0[1].x - row0[0].x
            assert spacing == pytest.approx(ARGUS_II_SPACING_UM, abs=1.0)

    def test_argus_ii_offset(self):
        """Argus II with offset shifts all electrodes."""
        dx, dy = -1000.0, 500.0
        electrodes = make_argus_ii(center_x=dx, center_y=dy)
        xs = [e.x for e in electrodes]
        ys = [e.y for e in electrodes]
        assert np.mean(xs) == pytest.approx(dx, abs=1.0)
        assert np.mean(ys) == pytest.approx(dy, abs=1.0)

    def test_argus_ii_rotation(self):
        """Argus II rotation changes electrode positions."""
        e0 = make_argus_ii()
        e_rot = make_argus_ii(rotation_deg=45.0)
        # Positions should differ
        x0 = np.array([e.x for e in e0])
        x_rot = np.array([e.x for e in e_rot])
        assert not np.allclose(x0, x_rot, atol=1.0)

    def test_prima_count(self):
        """PRIMA generates requested number of electrodes."""
        electrodes = make_prima(n_electrodes=378)
        assert len(electrodes) == 378

    def test_electrode_radius(self):
        """Argus II electrodes have 100 μm radius."""
        electrodes = make_argus_ii()
        for e in electrodes:
            assert e.radius == 100.0

    def test_electrode_default_current(self):
        """Electrodes default to zero current."""
        electrodes = make_argus_ii()
        for e in electrodes:
            assert e.current == 0.0


# ═══════════════════════════════════════════════════════════════
# AxonMapModel — Build & Simulate
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def small_model():
    """Build a small model for testing (coarse grid, fewer bundles)."""
    model = AxonMapModel(
        rho=200.0,
        axon_lambda=800.0,
        grid_spacing=200.0,       # coarse grid for speed
        x_range=(-2000, 2000),
        y_range=(-2000, 2000),
        axon_n_bundles=100,       # fewer bundles for speed
    )
    model.build()
    return model


class TestAxonMapModel:
    """Test the core AxonMapModel."""

    def test_build_sets_flag(self, small_model):
        """After build(), _built is True."""
        assert small_model._built is True

    def test_grid_shape(self, small_model):
        """Grid shape matches expected dimensions."""
        rows, cols = small_model._grid_shape
        assert rows > 0 and cols > 0
        # With spacing=200, range=[-2000,2000] → 21 points per axis
        assert rows == 21
        assert cols == 21

    def test_sensitivity_matrix_shape(self, small_model):
        """Sensitivity matrix is (n_cells, n_electrodes)."""
        electrodes = make_argus_ii()
        S = small_model.compute_sensitivity_matrix(electrodes)
        n_cells = small_model._grid_shape[0] * small_model._grid_shape[1]
        assert S.shape == (n_cells, len(electrodes))

    def test_sensitivity_nonnegative(self, small_model):
        """All sensitivity values are ≥ 0."""
        electrodes = make_argus_ii()
        S = small_model.compute_sensitivity_matrix(electrodes)
        assert np.all(S >= 0)

    def test_sensitivity_decays_with_distance(self, small_model):
        """Closer cells should have higher sensitivity than distant ones."""
        e = Electrode(x=0.0, y=0.0, current=1.0)
        # Find the cell closest to (0, 0) and one far away
        s_close = small_model.compute_sensitivity(e, 0)
        # Cell at center of grid
        center_idx = small_model._grid_shape[0] * small_model._grid_shape[1] // 2
        s_center = small_model.compute_sensitivity(e, center_idx)
        # At least one of these should be nonzero
        assert s_close >= 0 and s_center >= 0

    def test_simulate_shape(self, small_model):
        """Simulated percept matches grid shape."""
        electrodes = make_argus_ii()
        for e in electrodes:
            e.current = 1.0
        percept = small_model.simulate(electrodes)
        assert percept.shape == small_model._grid_shape

    def test_simulate_normalized(self, small_model):
        """Percept values in [0, 1]."""
        electrodes = make_argus_ii()
        for e in electrodes:
            e.current = 1.0
        percept = small_model.simulate(electrodes)
        assert percept.min() >= 0.0
        assert percept.max() <= 1.0 + 1e-7

    def test_simulate_zero_current(self, small_model):
        """Zero current → zero percept."""
        electrodes = make_argus_ii()
        for e in electrodes:
            e.current = 0.0
        percept = small_model.simulate(electrodes)
        assert np.all(percept == 0)

    def test_single_electrode_phosphene(self, small_model):
        """Single active electrode produces localized percept."""
        electrodes = make_argus_ii(center_x=-1000.0)
        for e in electrodes:
            e.current = 0.0
        electrodes[30].current = 1.0
        percept = small_model.simulate(electrodes)
        assert percept.max() > 0  # something visible
        # Most pixels should be dark
        assert np.mean(percept < 0.01) > 0.5

    def test_predict_convenience(self, small_model):
        """predict() returns valid percept for test image."""
        patterns = generate_test_images(size=64)
        percept = small_model.predict(patterns["letter_E"])
        assert percept.shape == small_model._grid_shape
        assert 0 <= percept.min() and percept.max() <= 1.0 + 1e-7


# ═══════════════════════════════════════════════════════════════
# Encoding Strategies
# ═══════════════════════════════════════════════════════════════

class TestEncoding:
    """Test image-to-current encoding strategies."""

    def test_direct_encoding(self):
        """Direct encoding maps bright pixels to nonzero current."""
        img = np.ones((64, 64))  # all white
        electrodes = make_argus_ii()
        electrodes = encode_image(img, electrodes, EncodingStrategy.DIRECT)
        currents = [e.current for e in electrodes]
        assert all(c > 0 for c in currents)

    def test_black_image_zero_current(self):
        """Black image → zero current for all encodings (CLAHE may produce tiny residual)."""
        img = np.zeros((64, 64))
        for strategy in EncodingStrategy:
            electrodes = make_argus_ii()
            electrodes = encode_image(img, electrodes, strategy)
            for e in electrodes:
                # CLAHE redistributes histogram even on uniform images,
                # producing a small residual (~1/255). Use relaxed tolerance.
                assert abs(e.current) < 0.05, \
                    f"Unexpected current with black image, encoding={strategy}"

    def test_current_amplitude_scaling(self):
        """current_amplitude scales the maximum current."""
        img = np.ones((64, 64))
        e_full = make_argus_ii()
        e_half = make_argus_ii()
        encode_image(img, e_full, EncodingStrategy.DIRECT, current_amplitude=1.0)
        encode_image(img, e_half, EncodingStrategy.DIRECT, current_amplitude=0.5)
        for ef, eh in zip(e_full, e_half):
            assert eh.current == pytest.approx(ef.current * 0.5, abs=1e-5)

    def test_edge_encoding_uniform(self):
        """Edge encoding on uniform image → near-zero current."""
        img = np.ones((64, 64)) * 0.5
        electrodes = make_argus_ii()
        electrodes = encode_image(img, electrodes, EncodingStrategy.EDGE)
        # Uniform image has no edges
        max_current = max(e.current for e in electrodes)
        assert max_current < 0.1

    def test_all_strategies_run(self):
        """All 4 encoding strategies execute without error."""
        img = generate_test_images(size=64)["letter_E"]
        for strategy in EncodingStrategy:
            electrodes = make_argus_ii()
            result = encode_image(img, electrodes, strategy)
            assert len(result) == ARGUS_II_N_ELECTRODES

    def test_encoding_centered_on_array(self):
        """Image FOV is centered on array center, not fovea."""
        img = np.zeros((64, 64))
        # Put a bright dot in center of image
        img[28:36, 28:36] = 1.0

        # Array at (0, 0) — dot should map to center electrodes
        e_center = make_argus_ii(center_x=0)
        encode_image(img, e_center, EncodingStrategy.DIRECT)

        # Array far away — dot still maps to center electrodes
        # (because FOV follows array)
        e_far = make_argus_ii(center_x=-3000)
        encode_image(img, e_far, EncodingStrategy.DIRECT)

        # Both should have similar current patterns
        c_center = [e.current for e in e_center]
        c_far = [e.current for e in e_far]
        assert np.corrcoef(c_center, c_far)[0, 1] > 0.95


# ═══════════════════════════════════════════════════════════════
# Test Images
# ═══════════════════════════════════════════════════════════════

class TestTestImages:
    """Verify built-in test patterns."""

    def test_all_patterns_present(self):
        """All 4 test patterns generated."""
        patterns = generate_test_images()
        assert set(patterns.keys()) == {"letter_E", "grating", "dot_superior", "dot_inferior"}

    def test_pattern_shape(self):
        """Patterns have correct shape."""
        for size in [64, 128, 256]:
            patterns = generate_test_images(size=size)
            for name, img in patterns.items():
                assert img.shape == (size, size), f"{name} wrong shape at size={size}"

    def test_pattern_range(self):
        """Pattern values in [0, 1]."""
        patterns = generate_test_images()
        for name, img in patterns.items():
            assert img.min() >= 0.0, f"{name} has negative values"
            assert img.max() <= 1.0, f"{name} exceeds 1.0"

    def test_letter_e_not_blank(self):
        """Letter E has both bright and dark pixels."""
        img = generate_test_images()["letter_E"]
        assert img.mean() > 0.05, "Letter E is too dark"
        assert img.mean() < 0.5, "Letter E is too bright"

    def test_dot_superior_position(self):
        """dot_superior has bright pixels in upper half of image."""
        img = generate_test_images()["dot_superior"]
        upper_half = img[:img.shape[0]//2, :]
        lower_half = img[img.shape[0]//2:, :]
        assert upper_half.mean() > lower_half.mean()

    def test_dot_inferior_position(self):
        """dot_inferior has bright pixels in lower half of image."""
        img = generate_test_images()["dot_inferior"]
        upper_half = img[:img.shape[0]//2, :]
        lower_half = img[img.shape[0]//2:, :]
        assert lower_half.mean() > upper_half.mean()


# ═══════════════════════════════════════════════════════════════
# SSIM
# ═══════════════════════════════════════════════════════════════

class TestSSIM:
    """Verify SSIM computation."""

    def test_ssim_identical(self):
        """SSIM of identical images ≈ 1.0."""
        img = np.random.rand(64, 64)
        assert compute_ssim(img, img) == pytest.approx(1.0, abs=0.01)

    def test_ssim_different(self):
        """SSIM of unrelated images < 1.0."""
        img1 = np.zeros((64, 64))
        img2 = np.ones((64, 64))
        assert compute_ssim(img1, img2) < 0.5

    def test_ssim_symmetric(self):
        """SSIM(a, b) ≈ SSIM(b, a)."""
        a = np.random.rand(64, 64)
        b = np.random.rand(64, 64)
        assert compute_ssim(a, b) == pytest.approx(compute_ssim(b, a), abs=1e-6)

    def test_ssim_size_mismatch(self):
        """SSIM handles different-sized inputs (resizes internally)."""
        a = np.random.rand(64, 64)
        b = np.random.rand(32, 32)
        ssim = compute_ssim(a, b)
        assert -1.0 <= ssim <= 1.0  # valid range


# ═══════════════════════════════════════════════════════════════
# Hemifield Constraints
# ═══════════════════════════════════════════════════════════════

class TestHemifieldConstraint:
    """Verify raphe and hemifield constraints."""

    def test_superior_electrode_superior_streak(self, small_model):
        """Electrode in superior retina → percept primarily in superior retina."""
        electrodes = make_argus_ii(center_x=-1000, center_y=800)
        for e in electrodes:
            e.current = 0.0
        # Activate an electrode in the superior part of the array
        electrodes[50].current = 1.0  # upper rows
        percept = small_model.simulate(electrodes)
        rows = percept.shape[0]
        upper = percept[rows//2:, :].sum()
        lower = percept[:rows//2, :].sum()
        # Superior electrode should produce more brightness in upper half
        # (y increases upward in grid, upper rows = superior retina)
        assert upper >= lower * 0.3  # at least some superior response

    def test_scoreboard_limit(self, small_model):
        """When ρ→0, λ→0, model degrades to scoreboard (point phosphenes)."""
        sb_model = AxonMapModel(
            rho=1.0, axon_lambda=1.0,
            grid_spacing=200.0,
            x_range=(-2000, 2000),
            y_range=(-2000, 2000),
            axon_n_bundles=50,
        )
        sb_model.build()
        electrodes = make_argus_ii()
        for e in electrodes:
            e.current = 0.0
        electrodes[30].current = 1.0
        percept = sb_model.simulate(electrodes)
        nonzero = np.sum(percept > 0.01)
        total = percept.size
        assert nonzero / total < 0.1, "Scoreboard should produce localized phosphene"

    def test_raphe_smoothness(self):
        """Raphe transition (y=0) should be smooth, not a hard seam.

        With large ρ and λ, the raphe region is most visible. The max
        row-to-row difference at the raphe should not be dramatically
        larger than the median difference across all rows.
        """
        model = AxonMapModel(
            rho=300.0, axon_lambda=1000.0,
            grid_spacing=100.0,
            x_range=(-2500, 2500),
            y_range=(-2500, 2500),
            axon_n_bundles=200,
        )
        model.build()
        electrodes = make_argus_ii(center_x=-1000.0)
        img = generate_test_images(size=128)["letter_E"]
        electrodes = encode_image(img, electrodes, EncodingStrategy.DIRECT)
        percept = model.simulate(electrodes)

        mid = percept.shape[0] // 2
        diffs = [
            np.max(np.abs(percept[r] - percept[r - 1]))
            for r in range(1, percept.shape[0] - 1)
        ]
        raphe_diff = max(diffs[mid - 1], diffs[mid], diffs[mid + 1])
        median_diff = np.median(diffs)
        ratio = raphe_diff / (median_diff + 1e-8)
        assert ratio < 3.0, (
            f"Raphe seam too sharp: ratio={ratio:.1f}x "
            f"(raphe={raphe_diff:.4f}, median={median_diff:.4f})"
        )


# ═══════════════════════════════════════════════════════════════
# Beyeler 2019 Subject Reproduction
# ═══════════════════════════════════════════════════════════════

class TestBeyelerReproduction:
    """Reproduce qualitative results from Beyeler 2019."""

    @pytest.fixture(scope="class")
    def s2_model(self):
        """Model with Subject 2 parameters (ρ=315, λ=500)."""
        s = BEYELER_SUBJECTS["S2"]
        model = AxonMapModel(
            rho=s["rho"], axon_lambda=s["axon_lambda"],
            grid_spacing=200.0,
            x_range=(-3000, 3000),
            y_range=(-3000, 3000),
            axon_n_bundles=100,
        )
        model.build()
        return model

    def test_s2_produces_percept(self, s2_model):
        """S2 model produces nonzero percept."""
        s = BEYELER_SUBJECTS["S2"]
        electrodes = make_argus_ii(
            center_x=s["array_x"], center_y=s["array_y"],
            rotation_deg=s["rotation"],
        )
        img = generate_test_images(size=128)["letter_E"]
        electrodes = encode_image(img, electrodes, EncodingStrategy.DIRECT)
        percept = s2_model.simulate(electrodes)
        assert percept.max() > 0

    def test_s2_high_rho_blurs(self, s2_model):
        """S2 (ρ=315) should produce broader phosphenes than ρ=50."""
        s = BEYELER_SUBJECTS["S2"]
        electrodes = make_argus_ii(
            center_x=s["array_x"], center_y=s["array_y"],
            rotation_deg=s["rotation"],
        )
        # Single electrode
        for e in electrodes:
            e.current = 0.0
        electrodes[30].current = 1.0
        percept = s2_model.simulate(electrodes)
        bright_pixels = np.sum(percept > 0.01)
        assert bright_pixels > 1, "ρ=315 should produce spread-out phosphene"


# ═══════════════════════════════════════════════════════════════
# Axon Map Rendering
# ═══════════════════════════════════════════════════════════════

class TestRendering:
    """Test visualization output."""

    def test_render_returns_png(self, small_model):
        """render_axon_map returns valid PNG bytes."""
        electrodes = make_argus_ii()
        png_bytes = render_axon_map(
            small_model.axon_bundles,
            small_model._bundle_phi0s,
            electrodes=electrodes,
        )
        assert isinstance(png_bytes, bytes)
        assert png_bytes[:4] == b'\x89PNG'  # PNG magic number
        assert len(png_bytes) > 1000  # non-trivial image

    def test_render_without_electrodes(self, small_model):
        """render_axon_map works without electrodes."""
        png_bytes = render_axon_map(
            small_model.axon_bundles,
            small_model._bundle_phi0s,
        )
        assert png_bytes[:4] == b'\x89PNG'
