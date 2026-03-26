"""
Benchmark: RetinalSim vs pulse2percept
=======================================
Compares single-electrode phosphene shapes between RetinalSim and
pulse2percept (Beyeler et al. 2017) using identical parameters from
Beyeler 2019 Table 2+3 (Subjects S2, S3, S4).

Requirements:
    pip install pulse2percept retinalsim matplotlib numpy

Usage:
    python benchmark_vs_p2p.py

Outputs:
    benchmark_single_electrode.png — side-by-side phosphene comparison
    benchmark_letter_e.png        — full image simulation comparison
    benchmark_results.txt         — numerical comparison metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
import sys
import os

# ── Add parent dir to path so retinalsim can be imported ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Imports ──
try:
    from retinalsim import (
        AxonMapModel, make_argus_ii, encode_image,
        EncodingStrategy, generate_test_images, BEYELER_SUBJECTS,
    )
    print("✓ RetinalSim imported")
except ImportError:
    print("✗ RetinalSim not found. Run: pip install -e .")
    sys.exit(1)

try:
    import pulse2percept as p2p
    print(f"✓ pulse2percept {p2p.__version__} imported")
except ImportError:
    print("✗ pulse2percept not found. Run: pip install pulse2percept")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SUBJECTS = {
    "S2": BEYELER_SUBJECTS["S2"],
    "S3": BEYELER_SUBJECTS["S3"],
    "S4": BEYELER_SUBJECTS["S4"],
}

# Test electrodes: center (#30), edge (#5), corner (#55)
TEST_ELECTRODES = [30, 5, 55]

# Grid parameters (must match for fair comparison)
GRID_SPACING = 80  # μm
X_RANGE = (-5000, 5000)
Y_RANGE = (-5000, 5000)

OUTPUT_DIR = os.path.dirname(__file__) or "."


# ═══════════════════════════════════════════════════════════════
# Helper: Run RetinalSim
# ═══════════════════════════════════════════════════════════════

def run_retinalsim_single(rho, axon_lambda, array_x, array_y, rotation, elec_idx):
    """Run single electrode phosphene in RetinalSim."""
    model = AxonMapModel(
        rho=rho, axon_lambda=axon_lambda,
        grid_spacing=GRID_SPACING,
        x_range=X_RANGE, y_range=Y_RANGE,
        axon_n_bundles=200,
    )
    model.build()

    electrodes = make_argus_ii(
        center_x=array_x, center_y=array_y, rotation_deg=rotation,
    )
    for e in electrodes:
        e.current = 0.0
    electrodes[elec_idx].current = 1.0

    percept = model.simulate(electrodes)
    return percept, model


def run_retinalsim_image(model, image, array_x, array_y, rotation):
    """Run full image simulation in RetinalSim."""
    electrodes = make_argus_ii(
        center_x=array_x, center_y=array_y, rotation_deg=rotation,
    )
    electrodes = encode_image(image, electrodes, EncodingStrategy.DIRECT)
    return model.simulate(electrodes)


# ═══════════════════════════════════════════════════════════════
# Helper: Run pulse2percept
# ═══════════════════════════════════════════════════════════════

def run_p2p_single(rho, axon_lambda, array_x, array_y, rotation, elec_idx):
    """Run single electrode phosphene in pulse2percept."""
    model = p2p.models.AxonMapModel(
        rho=rho, axlambda=axon_lambda,
        xrange=X_RANGE, yrange=Y_RANGE,
        xystep=GRID_SPACING,
        n_axons=200,
    )
    model.build()

    implant = p2p.implants.ArgusII(
        x=array_x, y=array_y, rot=np.radians(rotation),
    )

    # Single electrode stimulus
    stim_dict = {}
    elec_names = implant.electrode_names
    for i, name in enumerate(elec_names):
        stim_dict[name] = 1.0 if i == elec_idx else 0.0
    implant.stim = p2p.stimuli.Stimulus(stim_dict)

    percept = model.predict_percept(implant)
    # Extract 2D array from percept
    percept_arr = percept.data.squeeze()
    return percept_arr, model


def run_p2p_image(model_p2p, image, array_x, array_y, rotation):
    """Run full image simulation in pulse2percept."""
    implant = p2p.implants.ArgusII(
        x=array_x, y=array_y, rot=np.radians(rotation),
    )

    # Map image to electrode currents (manual, matching RetinalSim's direct encoding)
    from retinalsim import make_argus_ii as rs_make_argus, encode_image as rs_encode
    rs_elecs = rs_make_argus(center_x=array_x, center_y=array_y, rotation_deg=rotation)
    rs_elecs = rs_encode(image, rs_elecs, EncodingStrategy.DIRECT)
    currents = [e.current for e in rs_elecs]

    # Apply same currents to p2p
    stim_dict = {}
    for i, name in enumerate(implant.electrode_names):
        stim_dict[name] = currents[i]
    implant.stim = p2p.stimuli.Stimulus(stim_dict)

    percept = model_p2p.predict_percept(implant)
    return percept.data.squeeze()


# ═══════════════════════════════════════════════════════════════
# Helper: Compute similarity metrics
# ═══════════════════════════════════════════════════════════════

def compute_metrics(rs_percept, p2p_percept):
    """Compare two percept arrays."""
    # Resize to same shape if needed
    if rs_percept.shape != p2p_percept.shape:
        import cv2
        p2p_percept = cv2.resize(
            p2p_percept,
            (rs_percept.shape[1], rs_percept.shape[0]),
        )

    # Normalize both to [0, 1]
    rs = rs_percept.copy()
    p2 = p2p_percept.copy()
    if rs.max() > 0:
        rs = rs / rs.max()
    if p2.max() > 0:
        p2 = p2 / p2.max()

    # Pearson correlation
    rs_flat = rs.ravel()
    p2_flat = p2.ravel()
    corr = np.corrcoef(rs_flat, p2_flat)[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((rs_flat - p2_flat) ** 2))

    # Structural similarity (simplified)
    from retinalsim import compute_ssim
    ssim = compute_ssim(rs, p2)

    return {"correlation": corr, "rmse": rmse, "ssim": ssim}


# ═══════════════════════════════════════════════════════════════
# Benchmark 1: Single Electrode Phosphenes
# ═══════════════════════════════════════════════════════════════

def benchmark_single_electrode():
    """Compare single electrode phosphenes across S2, S3, S4."""
    print("\n" + "=" * 60)
    print("BENCHMARK 1: Single Electrode Phosphenes")
    print("=" * 60)

    n_subj = len(SUBJECTS)
    n_elec = len(TEST_ELECTRODES)

    fig, axes = plt.subplots(
        n_subj, n_elec * 2 + 1,  # +1 for labels column
        figsize=(4 * (n_elec * 2 + 1), 4 * n_subj),
    )

    # Better layout: subjects as rows, electrode pairs as columns
    fig2, axes2 = plt.subplots(
        n_subj * n_elec, 2,
        figsize=(8, 4 * n_subj * n_elec),
    )

    results = []
    row = 0

    for si, (subj_name, params) in enumerate(SUBJECTS.items()):
        rho = params["rho"]
        lam = params["axon_lambda"]
        ax = params["array_x"]
        ay = params["array_y"]
        rot = params["rotation"]

        print(f"\n--- {subj_name}: ρ={rho}, λ={lam} ---")

        for ei, elec_idx in enumerate(TEST_ELECTRODES):
            print(f"  Electrode #{elec_idx}...", end=" ", flush=True)

            # RetinalSim
            t0 = time.time()
            rs_percept, rs_model = run_retinalsim_single(
                rho, lam, ax, ay, rot, elec_idx,
            )
            rs_time = time.time() - t0

            # pulse2percept
            t0 = time.time()
            p2p_percept, p2p_model = run_p2p_single(
                rho, lam, ax, ay, rot, elec_idx,
            )
            p2p_time = time.time() - t0

            # Metrics
            metrics = compute_metrics(rs_percept, p2p_percept)

            result_line = (
                f"{subj_name} elec#{elec_idx}: "
                f"corr={metrics['correlation']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, "
                f"SSIM={metrics['ssim']:.4f}, "
                f"RS={rs_time:.1f}s, P2P={p2p_time:.1f}s"
            )
            print(result_line)
            results.append(result_line)

            # Plot side by side
            axes2[row, 0].imshow(rs_percept, cmap="gray", origin="lower")
            axes2[row, 0].set_title(
                f"RetinalSim\n{subj_name} #{elec_idx}",
                fontsize=10,
            )
            axes2[row, 0].axis("off")

            axes2[row, 1].imshow(p2p_percept, cmap="gray", origin="lower")
            axes2[row, 1].set_title(
                f"pulse2percept\ncorr={metrics['correlation']:.3f}",
                fontsize=10,
            )
            axes2[row, 1].axis("off")

            row += 1

    fig2.suptitle(
        "Single Electrode Phosphene Comparison\n"
        "RetinalSim vs pulse2percept",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig2.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "benchmark_single_electrode.png")
    fig2.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"\nSaved: {outpath}")

    return results


# ═══════════════════════════════════════════════════════════════
# Benchmark 2: Full Image (Letter E)
# ═══════════════════════════════════════════════════════════════

def benchmark_letter_e():
    """Compare letter E percept across S2, S3, S4."""
    print("\n" + "=" * 60)
    print("BENCHMARK 2: Letter E Full Simulation")
    print("=" * 60)

    image = generate_test_images(size=256)["letter_E"]

    fig, axes = plt.subplots(len(SUBJECTS), 3, figsize=(12, 4 * len(SUBJECTS)))
    # Columns: Input | RetinalSim | pulse2percept

    results = []

    for si, (subj_name, params) in enumerate(SUBJECTS.items()):
        rho = params["rho"]
        lam = params["axon_lambda"]
        ax = params["array_x"]
        ay = params["array_y"]
        rot = params["rotation"]

        print(f"\n--- {subj_name}: ρ={rho}, λ={lam} ---")

        # Build RetinalSim model
        print("  RetinalSim...", end=" ", flush=True)
        t0 = time.time()
        rs_model = AxonMapModel(
            rho=rho, axon_lambda=lam,
            grid_spacing=GRID_SPACING,
            x_range=X_RANGE, y_range=Y_RANGE,
            axon_n_bundles=200,
        )
        rs_model.build()
        rs_percept = run_retinalsim_image(rs_model, image, ax, ay, rot)
        rs_time = time.time() - t0
        print(f"{rs_time:.1f}s")

        # Build pulse2percept model
        print("  pulse2percept...", end=" ", flush=True)
        t0 = time.time()
        p2p_model = p2p.models.AxonMapModel(
            rho=rho, axlambda=lam,
            xrange=X_RANGE, yrange=Y_RANGE,
            xystep=GRID_SPACING,
            n_axons=200,
        )
        p2p_model.build()
        p2p_percept = run_p2p_image(p2p_model, image, ax, ay, rot)
        p2p_time = time.time() - t0
        print(f"{p2p_time:.1f}s")

        # Metrics
        metrics = compute_metrics(rs_percept, p2p_percept)
        result_line = (
            f"{subj_name} Letter E: "
            f"corr={metrics['correlation']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"SSIM={metrics['ssim']:.4f}"
        )
        print(f"  {result_line}")
        results.append(result_line)

        # Plot
        axes[si, 0].imshow(image, cmap="gray", origin="upper")
        axes[si, 0].set_title("Input" if si == 0 else "", fontsize=11)
        axes[si, 0].set_ylabel(
            f"{subj_name}\nρ={rho} λ={lam}",
            fontsize=11, fontweight="bold",
        )
        axes[si, 0].axis("off")

        axes[si, 1].imshow(rs_percept, cmap="gray", origin="lower")
        axes[si, 1].set_title(
            "RetinalSim" if si == 0 else "",
            fontsize=11,
        )
        axes[si, 1].axis("off")

        axes[si, 2].imshow(p2p_percept, cmap="gray", origin="lower")
        axes[si, 2].set_title(
            f"pulse2percept" if si == 0 else "",
            fontsize=11,
        )
        axes[si, 2].axis("off")

        # Add correlation annotation
        axes[si, 2].text(
            0.95, 0.05, f"r={metrics['correlation']:.3f}",
            transform=axes[si, 2].transAxes,
            fontsize=9, color="white", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )

    fig.suptitle(
        "Letter E Percept Comparison: RetinalSim vs pulse2percept",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "benchmark_letter_e.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("RetinalSim vs pulse2percept Benchmark")
    print("=" * 60)
    print(f"Grid: {X_RANGE} × {Y_RANGE}, spacing={GRID_SPACING}μm")
    print(f"Subjects: {list(SUBJECTS.keys())}")
    print(f"Test electrodes: {TEST_ELECTRODES}")

    all_results = []

    # Run benchmarks
    results1 = benchmark_single_electrode()
    all_results.extend(results1)

    results2 = benchmark_letter_e()
    all_results.extend(results2)

    # Save text results
    outpath = os.path.join(OUTPUT_DIR, "benchmark_results.txt")
    with open(outpath, "w") as f:
        f.write("RetinalSim vs pulse2percept Benchmark Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Grid: {X_RANGE} x {Y_RANGE}, spacing={GRID_SPACING}um\n")
        f.write(f"Subjects: {list(SUBJECTS.keys())}\n")
        f.write(f"Electrodes: {TEST_ELECTRODES}\n\n")
        for line in all_results:
            f.write(line + "\n")

    print(f"\nSaved: {outpath}")
    print("\n" + "=" * 60)
    print("DONE. Check output files:")
    print(f"  benchmark_single_electrode.png")
    print(f"  benchmark_letter_e.png")
    print(f"  benchmark_results.txt")
    print("=" * 60)
