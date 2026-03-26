"""
Beyeler 2019 Subject Reproduction
==================================
Reproduces the phosphene predictions for Subjects 2–4 from
Beyeler et al. (2019) Sci Rep 9:9199, Table 2 + Table 3.

Each subject has unique ρ, λ, implant position, and rotation
fitted from their drawing data. This example shows how these
parameters dramatically change the simulated percept.

Expected outcomes (from Beyeler 2019):
- S2 (ρ=315, λ=500):  Moderate spread, short streaks → blurred but recognizable
- S3 (ρ=144, λ=1414): Narrow spread, very long streaks → stripe-like distortion
- S4 (ρ=437, λ=1420): Wide spread, very long streaks → unrecognizable blobs
"""

import matplotlib.pyplot as plt
from retinalsim import (
    AxonMapModel,
    BEYELER_SUBJECTS,
    make_argus_ii,
    encode_image,
    generate_test_images,
    compute_ssim,
    EncodingStrategy,
)

image = generate_test_images(size=256)["letter_E"]

# Only S2–S4 used Argus II (S1 used Argus I)
subjects = ["S2", "S3", "S4"]

fig, axes = plt.subplots(2, len(subjects), figsize=(15, 10))

for col, subj_name in enumerate(subjects):
    s = BEYELER_SUBJECTS[subj_name]
    print(f"\n{subj_name}: ρ={s['rho']}, λ={s['axon_lambda']}, "
          f"pos=({s['array_x']}, {s['array_y']}), rot={s['rotation']}°")

    # Build model with subject-specific ρ and λ
    model = AxonMapModel(
        rho=s["rho"],
        axon_lambda=s["axon_lambda"],
        grid_spacing=80.0,
    )
    model.build()

    # Create array with subject-specific placement
    electrodes = make_argus_ii(
        center_x=s["array_x"],
        center_y=s["array_y"],
        rotation_deg=s["rotation"],
    )

    # Simulate
    electrodes = encode_image(image, electrodes, EncodingStrategy.DIRECT)
    percept = model.simulate(electrodes)
    ssim = compute_ssim(image, percept)
    print(f"  SSIM = {ssim:.4f}")

    # Plot input (top row)
    axes[0, col].imshow(image, cmap="gray", origin="upper")
    axes[0, col].set_title(f"{subj_name} Input", fontsize=12)
    axes[0, col].axis("off")

    # Plot percept (bottom row)
    axes[1, col].imshow(percept, cmap="gray", origin="lower")
    axes[1, col].set_title(
        f"{subj_name} Percept\n"
        f"ρ={s['rho']}μm, λ={s['axon_lambda']}μm\n"
        f"SSIM={ssim:.3f}",
        fontsize=11,
    )
    axes[1, col].axis("off")

plt.suptitle(
    "Beyeler 2019 Subject Reproduction — Letter E",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
plt.savefig("beyeler_subjects.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: beyeler_subjects.png")
