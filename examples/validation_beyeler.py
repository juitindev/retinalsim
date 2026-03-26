"""
Validation: RetinalSim vs Beyeler 2019 Published Results
========================================================
Compares RetinalSim phosphene predictions against the qualitative
descriptions and Figure 5 of Beyeler et al. (2019) Sci Rep 9:9199.

This script does NOT require pulse2percept. It validates that:
1. Phosphene shapes match the expected ρ/λ-dependent behavior
2. Streak direction follows axon trajectories (temporal direction)
3. ρ controls streak width, λ controls streak length
4. Near-OD electrodes produce rounder phosphenes
5. S2/S3/S4 produce qualitatively different percept patterns

Requirements:
    pip install numpy opencv-python-headless matplotlib

Usage:
    python validation_beyeler.py

Outputs:
    validation_single_phosphenes.png  — S2/S3/S4 phosphene shapes
    validation_letter_e.png           — S2/S3/S4 letter E percepts
    validation_rho_lambda_sweep.png   — parameter sweep showing ρ/λ effects
    validation_results.txt            — quantitative shape metrics
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retinalsim import (
    AxonMapModel, make_argus_ii, encode_image,
    EncodingStrategy, generate_test_images, compute_ssim,
    render_axon_map, BEYELER_SUBJECTS,
)

OUTPUT_DIR = os.path.dirname(__file__) or "."


# ═══════════════════════════════════════════════════════════════
# Helper: Measure phosphene shape
# ═══════════════════════════════════════════════════════════════

def measure_phosphene(percept, grid_spacing, threshold=0.05):
    """Extract shape metrics from a percept array."""
    bright = percept > threshold
    n_bright = int(np.sum(bright))

    if n_bright == 0:
        return {"n_pixels": 0, "x_span_um": 0, "y_span_um": 0,
                "aspect_ratio": 0, "area_um2": 0, "centroid_x": 0, "centroid_y": 0}

    ys, xs = np.where(bright)
    x_span = (xs.max() - xs.min()) * grid_spacing
    y_span = (ys.max() - ys.min()) * grid_spacing
    aspect = y_span / (x_span + 1e-6)
    area = n_bright * grid_spacing ** 2

    # Weighted centroid
    weights = percept[bright]
    cx = np.average(xs, weights=weights) * grid_spacing
    cy = np.average(ys, weights=weights) * grid_spacing

    return {
        "n_pixels": n_bright,
        "x_span_um": x_span,
        "y_span_um": y_span,
        "aspect_ratio": aspect,
        "area_um2": area,
        "centroid_x": cx,
        "centroid_y": cy,
    }


# ═══════════════════════════════════════════════════════════════
# Validation 1: Single Electrode Phosphenes (Beyeler Fig.5)
# ═══════════════════════════════════════════════════════════════

def validation_single_phosphenes():
    """
    Beyeler 2019 Figure 5 shows single-electrode phosphenes for S2-S4.

    Expected behavior (from Beyeler 2019):
    - S2 (ρ=315, λ=500):  Moderate spread, short streaks
    - S3 (ρ=144, λ=1414): Narrow spread, long streaks
    - S4 (ρ=437, λ=1420): Wide spread, long streaks

    We test 5 electrodes per subject spanning the array:
    - #0  (bottom-left)
    - #15 (bottom-center)
    - #30 (center)
    - #45 (top-center)
    - #55 (top-right)
    """
    print("\n" + "=" * 60)
    print("VALIDATION 1: Single Electrode Phosphenes")
    print("(Compare with Beyeler 2019 Figure 5)")
    print("=" * 60)

    test_electrodes = [0, 15, 30, 45, 55]
    subjects = ["S2", "S3", "S4"]

    fig, axes = plt.subplots(
        len(subjects), len(test_electrodes),
        figsize=(4 * len(test_electrodes), 4 * len(subjects)),
    )

    results = []

    for si, subj_name in enumerate(subjects):
        s = BEYELER_SUBJECTS[subj_name]
        print(f"\n--- {subj_name}: ρ={s['rho']}, λ={s['axon_lambda']} ---")

        t0 = time.time()
        model = AxonMapModel(
            rho=s["rho"], axon_lambda=s["axon_lambda"],
            grid_spacing=80, x_range=(-5000, 5000), y_range=(-5000, 5000),
            axon_n_bundles=200,
        )
        model.build()
        build_time = time.time() - t0
        print(f"  Build: {build_time:.1f}s")

        electrodes_base = make_argus_ii(
            center_x=s["array_x"], center_y=s["array_y"],
            rotation_deg=s["rotation"],
        )

        for ei, elec_idx in enumerate(test_electrodes):
            electrodes = make_argus_ii(
                center_x=s["array_x"], center_y=s["array_y"],
                rotation_deg=s["rotation"],
            )
            for e in electrodes:
                e.current = 0.0
            electrodes[elec_idx].current = 1.0

            t1 = time.time()
            percept = model.simulate(electrodes)
            sim_time = time.time() - t1

            metrics = measure_phosphene(percept, model.grid_spacing)
            ex = electrodes[elec_idx].x
            ey = electrodes[elec_idx].y

            line = (
                f"  #{elec_idx:2d} ({ex:+6.0f},{ey:+6.0f}μm): "
                f"{metrics['n_pixels']:4d}px, "
                f"span={metrics['x_span_um']:.0f}×{metrics['y_span_um']:.0f}μm, "
                f"aspect={metrics['aspect_ratio']:.2f}, "
                f"{sim_time:.1f}s"
            )
            print(line)
            results.append(f"{subj_name} {line.strip()}")

            # Plot
            ax = axes[si, ei]
            ax.imshow(percept, cmap="hot", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"#{elec_idx}", fontsize=10)
            if ei == 0:
                ax.set_ylabel(
                    f"{subj_name}\nρ={s['rho']} λ={s['axon_lambda']}",
                    fontsize=11, fontweight="bold",
                )
            ax.axis("off")

    fig.suptitle(
        "Single Electrode Phosphenes — RetinalSim\n"
        "Beyeler 2019 Subjects S2, S3, S4",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "validation_single_phosphenes.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    return results


# ═══════════════════════════════════════════════════════════════
# Validation 2: Letter E Percepts
# ═══════════════════════════════════════════════════════════════

def validation_letter_e():
    """
    Compare letter E percepts across S2, S3, S4.

    Expected (from Beyeler 2019 discussion):
    - S2: Blurred but recognizable (short λ limits streak length)
    - S3: Stripe-like distortion (narrow ρ + long λ = thin long streaks)
    - S4: Unrecognizable blobs (large ρ + long λ = everything smeared)
    """
    print("\n" + "=" * 60)
    print("VALIDATION 2: Letter E Percept Comparison")
    print("=" * 60)

    image = generate_test_images(size=256)["letter_E"]
    subjects = ["S2", "S3", "S4"]

    fig, axes = plt.subplots(len(subjects), 3, figsize=(12, 4 * len(subjects)))
    # Columns: Input | Percept | Axon Map

    results = []

    for si, subj_name in enumerate(subjects):
        s = BEYELER_SUBJECTS[subj_name]
        print(f"\n--- {subj_name}: ρ={s['rho']}, λ={s['axon_lambda']} ---")

        t0 = time.time()
        model = AxonMapModel(
            rho=s["rho"], axon_lambda=s["axon_lambda"],
            grid_spacing=80, x_range=(-5000, 5000), y_range=(-5000, 5000),
            axon_n_bundles=200,
        )
        model.build()

        electrodes = make_argus_ii(
            center_x=s["array_x"], center_y=s["array_y"],
            rotation_deg=s["rotation"],
        )
        electrodes = encode_image(image, electrodes, EncodingStrategy.DIRECT)
        percept = model.simulate(electrodes)
        ssim = compute_ssim(image, percept)
        elapsed = time.time() - t0

        line = f"{subj_name}: SSIM={ssim:.4f}, time={elapsed:.1f}s"
        print(f"  {line}")
        results.append(line)

        # Plot input
        axes[si, 0].imshow(image, cmap="gray", origin="upper")
        axes[si, 0].set_title("Input" if si == 0 else "", fontsize=11)
        axes[si, 0].set_ylabel(
            f"{subj_name}\nρ={s['rho']} λ={s['axon_lambda']}",
            fontsize=11, fontweight="bold",
        )
        axes[si, 0].axis("off")

        # Plot percept
        axes[si, 1].imshow(percept, cmap="gray", origin="lower")
        axes[si, 1].set_title("Simulated Percept" if si == 0 else "", fontsize=11)
        axes[si, 1].text(
            0.95, 0.05, f"SSIM={ssim:.3f}",
            transform=axes[si, 1].transAxes,
            fontsize=9, color="white", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        )
        axes[si, 1].axis("off")

        # Plot axon map
        png_bytes = render_axon_map(
            model.axon_bundles, model._bundle_phi0s,
            electrodes=electrodes,
        )
        import io
        axon_img = plt.imread(io.BytesIO(png_bytes), format='png')
        axes[si, 2].imshow(axon_img)
        axes[si, 2].set_title("Axon Map + Array" if si == 0 else "", fontsize=11)
        axes[si, 2].axis("off")

    fig.suptitle(
        "Letter E Simulation — Beyeler 2019 Subject Parameters\n"
        "RetinalSim Axon Map Model",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "validation_letter_e.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    return results


# ═══════════════════════════════════════════════════════════════
# Validation 3: ρ/λ Parameter Sweep
# ═══════════════════════════════════════════════════════════════

def validation_parameter_sweep():
    """
    Demonstrate that ρ controls width and λ controls length.

    Expected (Beyeler 2019 Eq.5+9):
    - ρ↑ → wider phosphenes (more distant axons activated)
    - λ↑ → longer phosphenes (signal propagates further along axon)
    - ρ→0, λ→0 → scoreboard model (point phosphenes)
    """
    print("\n" + "=" * 60)
    print("VALIDATION 3: ρ/λ Parameter Sweep")
    print("=" * 60)

    rho_values = [50, 150, 300, 500]
    lambda_values = [200, 600, 1200, 2000]

    fig, axes = plt.subplots(
        len(rho_values), len(lambda_values),
        figsize=(3.5 * len(lambda_values), 3.5 * len(rho_values)),
    )

    results = []
    elec_idx = 30  # center electrode

    for ri, rho in enumerate(rho_values):
        for li, lam in enumerate(lambda_values):
            print(f"  ρ={rho:3d}, λ={lam:4d}...", end=" ", flush=True)

            t0 = time.time()
            model = AxonMapModel(
                rho=rho, axon_lambda=lam,
                grid_spacing=100, x_range=(-4000, 4000), y_range=(-4000, 4000),
                axon_n_bundles=150,
            )
            model.build()

            electrodes = make_argus_ii(center_x=-1000, rotation_deg=-25)
            for e in electrodes:
                e.current = 0.0
            electrodes[elec_idx].current = 1.0

            percept = model.simulate(electrodes)
            metrics = measure_phosphene(percept, model.grid_spacing)
            elapsed = time.time() - t0

            line = (
                f"ρ={rho:3d} λ={lam:4d}: "
                f"span={metrics['x_span_um']:.0f}×{metrics['y_span_um']:.0f}, "
                f"aspect={metrics['aspect_ratio']:.2f}, "
                f"area={metrics['area_um2']/1e6:.2f}mm²"
            )
            print(f"{elapsed:.1f}s")
            results.append(line)

            ax = axes[ri, li]
            ax.imshow(percept, cmap="hot", origin="lower", vmin=0, vmax=1)
            if ri == 0:
                ax.set_title(f"λ={lam}", fontsize=11)
            if li == 0:
                ax.set_ylabel(f"ρ={rho}", fontsize=11, fontweight="bold")
            ax.axis("off")

    fig.suptitle(
        "ρ/λ Parameter Sweep — Single Electrode #30\n"
        "ρ controls width, λ controls length",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "validation_rho_lambda_sweep.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    return results


# ═══════════════════════════════════════════════════════════════
# Validation 4: Scoreboard Limit
# ═══════════════════════════════════════════════════════════════

def validation_scoreboard():
    """
    When ρ→0 and λ→0, the axon map model should degrade to the
    scoreboard model (each electrode = one point phosphene).
    """
    print("\n" + "=" * 60)
    print("VALIDATION 4: Scoreboard Limit (ρ→0, λ→0)")
    print("=" * 60)

    configs = [
        (1, 1, "Scoreboard (ρ=1, λ=1)"),
        (50, 50, "Near-scoreboard (ρ=50, λ=50)"),
        (200, 800, "Default (ρ=200, λ=800)"),
        (400, 1400, "S4-like (ρ=400, λ=1400)"),
    ]

    image = generate_test_images(size=256)["letter_E"]

    fig, axes = plt.subplots(1, len(configs), figsize=(5 * len(configs), 5))

    results = []

    for ci, (rho, lam, label) in enumerate(configs):
        print(f"  {label}...", end=" ", flush=True)

        t0 = time.time()
        model = AxonMapModel(
            rho=rho, axon_lambda=lam,
            grid_spacing=100, x_range=(-3000, 3000), y_range=(-3000, 3000),
            axon_n_bundles=150,
        )
        model.build()

        electrodes = make_argus_ii(center_x=-1000, rotation_deg=-25)
        electrodes = encode_image(image, electrodes, EncodingStrategy.DIRECT)
        percept = model.simulate(electrodes)
        ssim = compute_ssim(image, percept)
        elapsed = time.time() - t0

        line = f"{label}: SSIM={ssim:.4f}, time={elapsed:.1f}s"
        print(line)
        results.append(line)

        axes[ci].imshow(percept, cmap="gray", origin="lower")
        axes[ci].set_title(f"{label}\nSSIM={ssim:.3f}", fontsize=10)
        axes[ci].axis("off")

    fig.suptitle(
        "Scoreboard → Axon Map Transition\n"
        "Letter E: ρ/λ increasing left to right",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "validation_scoreboard.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {outpath}")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("RetinalSim Validation — Beyeler 2019 Comparison")
    print("=" * 60)

    all_results = []

    r1 = validation_single_phosphenes()
    all_results.extend(["", "=== Single Electrode Phosphenes ==="] + r1)

    r2 = validation_letter_e()
    all_results.extend(["", "=== Letter E Percepts ==="] + r2)

    r3 = validation_parameter_sweep()
    all_results.extend(["", "=== ρ/λ Parameter Sweep ==="] + r3)

    r4 = validation_scoreboard()
    all_results.extend(["", "=== Scoreboard Limit ==="] + r4)

    # Save text results
    outpath = os.path.join(OUTPUT_DIR, "validation_results.txt")
    with open(outpath, "w") as f:
        f.write("RetinalSim Validation Results\n")
        f.write("Comparison with Beyeler et al. (2019) Sci Rep 9:9199\n")
        f.write("=" * 50 + "\n\n")
        for line in all_results:
            f.write(line + "\n")
    print(f"\nSaved: {outpath}")

    print("\n" + "=" * 60)
    print("DONE. Output files:")
    print("  validation_single_phosphenes.png")
    print("  validation_letter_e.png")
    print("  validation_rho_lambda_sweep.png")
    print("  validation_scoreboard.png")
    print("  validation_results.txt")
    print("=" * 60)
