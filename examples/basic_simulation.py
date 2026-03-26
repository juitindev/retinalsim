"""
Basic Simulation Example
========================
Demonstrates how to use RetinalSim to simulate prosthetic vision
with the Argus II electrode array.

This example:
1. Builds the axon map model
2. Generates a test pattern (letter E)
3. Simulates the percept with different encoding strategies
4. Saves output images for comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from retinalsim import (
    AxonMapModel,
    make_argus_ii,
    encode_image,
    generate_test_images,
    compute_ssim,
    render_axon_map,
    EncodingStrategy,
)

# ── 1. Build model ──
print("Building axon map model...")
model = AxonMapModel(
    rho=200.0,          # radial current spread [μm]
    axon_lambda=800.0,  # axonal signal decay [μm]
    grid_spacing=50.0,  # simulation resolution [μm]
)
model.build()
print(f"  Grid shape: {model._grid_shape}")
print(f"  Axon bundles: {len(model.axon_bundles)}")

# ── 2. Create electrode array ──
electrodes = make_argus_ii(
    center_x=-1000.0,    # 1mm temporal from fovea
    center_y=0.0,
    rotation_deg=-25.0,  # typical surgical rotation
)
print(f"  Electrodes: {len(electrodes)}")

# ── 3. Simulate with different encodings ──
image = generate_test_images(size=256)["letter_E"]

fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# Input
axes[0].imshow(image, cmap="gray", origin="upper")
axes[0].set_title("Input: Letter E")
axes[0].axis("off")

for i, strategy in enumerate(EncodingStrategy):
    # Fresh electrodes each time (encode_image modifies in place)
    elecs = make_argus_ii(center_x=-1000.0, rotation_deg=-25.0)
    elecs = encode_image(image, elecs, strategy)
    percept = model.simulate(elecs)
    ssim = compute_ssim(image, percept)

    axes[i + 1].imshow(percept, cmap="gray", origin="lower")
    axes[i + 1].set_title(f"{strategy.value}\nSSIM={ssim:.3f}")
    axes[i + 1].axis("off")

plt.suptitle("RetinalSim: Encoding Strategy Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("encoding_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: encoding_comparison.png")

# ── 4. Render axon map ──
png_bytes = render_axon_map(
    model.axon_bundles,
    model._bundle_phi0s,
    electrodes=electrodes,
)
with open("axon_map.png", "wb") as f:
    f.write(png_bytes)
print("Saved: axon_map.png")
