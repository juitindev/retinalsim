# RetinalSim

[![Tests](https://github.com/juit/retinalsim/actions/workflows/tests.yml/badge.svg)](https://github.com/juit/retinalsim/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JOSS](https://joss.theoj.org/papers/STATUS_BADGE_URL/status.svg)](https://joss.theoj.org/papers/STATUS_BADGE_URL)

An open-source retinal prosthesis simulator based on the Beyeler (2019) axon map model. Predicts the elongated phosphene shapes ("streaks") perceived by patients with epiretinal implants like the Argus II.

**Live demo:** [retinalsim.com](http://retinalsim.com)

![RetinalSim screenshot](docs/screenshot.png)

## Features

- **Axon map model** — full implementation of Beyeler et al. (2019) with Jansonius (2009, 2012) nerve fiber bundle trajectories
- **Vectorized computation** — NumPy-optimized sensitivity matrix (< 2s on cached model)
- **4 encoding strategies** — direct luminance, edge detection, local contrast (CLAHE), gradient saliency
- **Interactive web UI** — real-time parameter adjustment with FastAPI backend
- **Argus II & PRIMA** electrode array models
- **Anatomical constraints** — raphe termination + hemifield assignment
- **Beyeler subject presets** — reproduce published patient outcomes (S2–S4)
- **Docker deployment** — one-command setup for institutional or classroom use

## Quick start

### Install

```bash
pip install retinalsim
```

Or from source:

```bash
git clone https://github.com/juit/retinalsim.git
cd retinalsim
pip install -e ".[dev]"
```

### Python API

```python
import numpy as np
from retinalsim import AxonMapModel, make_argus_ii, generate_test_images

# Build model
model = AxonMapModel(rho=200, axon_lambda=800)
model.build()

# Simulate
image = generate_test_images()["letter_E"]
percept = model.predict(image, center_x=-1000, rotation_deg=-25)

# Or with more control:
from retinalsim import encode_image, EncodingStrategy

electrodes = make_argus_ii(center_x=-1000, rotation_deg=-25)
electrodes = encode_image(image, electrodes, EncodingStrategy.EDGE)
S = model.compute_sensitivity_matrix(electrodes)
percept = model.simulate(electrodes, S)
```

### Reproduce Beyeler 2019 subjects

```python
from retinalsim import AxonMapModel, BEYELER_SUBJECTS, make_argus_ii

s2 = BEYELER_SUBJECTS["S2"]
model = AxonMapModel(rho=s2["rho"], axon_lambda=s2["axon_lambda"])
model.build()

electrodes = make_argus_ii(
    center_x=s2["array_x"],
    center_y=s2["array_y"],
    rotation_deg=s2["rotation"],
)
percept = model.predict(image, electrodes=electrodes)
```

### Web interface

```bash
# Install server dependencies
pip install retinalsim[server]

# Run locally
cd retinalsim
python -m retinalsim.server
# Open http://localhost:8000
```

### Docker deployment

```bash
cd deploy
docker compose up -d --build
# Open http://localhost
```

## Model overview

The axon map model predicts that epiretinal stimulation produces elongated percepts because electrical current activates not only the ganglion cells directly beneath each electrode, but also the axons of more distant cells that pass through the electrode's vicinity. Each activated axon propagates the signal to the brain, which interprets it as if the corresponding soma location were stimulated.

The percept shape is governed by two patient-specific parameters:
- **ρ (rho)** — radial current spread [μm], controls streak width
- **λ (lambda)** — axonal signal decay [μm], controls streak length

When ρ → 0 and λ → 0, the model degrades to the "scoreboard" model where each electrode produces a point phosphene.

### Key equations

**Jansonius (2009) Eq.1 — axon trajectory:**
```
φ(φ₀, r) = φ₀ + b(φ₀) · (r − r₀)^c(φ₀)
```

**Beyeler (2019) Eq.5+9 — electrode sensitivity:**
```
S = Σ_k exp(−d_k² / 2ρ²) × exp(−l_k² / 2λ²)
```

See the [JOSS paper](paper.md) for full details.

## Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `rho` | 50–1000 μm | 200 | Radial current spread |
| `axon_lambda` | 100–2000 μm | 800 | Axonal signal decay |
| `grid_spacing` | 20–200 μm | 50 | Simulation grid resolution |
| `axon_n_bundles` | 50–500 | 400 | Number of traced axon bundles |
| `center_x` | −3000–1000 μm | −1000 | Array X offset from fovea |
| `center_y` | −2000–2000 μm | 0 | Array Y offset from fovea |
| `rotation_deg` | −90°–90° | −25° | Array rotation |

### Beyeler 2019 fitted parameters (Table 2 + Table 3)

| Subject | ρ (μm) | λ (μm) | X (μm) | Y (μm) | Rotation (°) | Implant |
|---------|--------|--------|---------|---------|---------------|---------|
| S1 | 410 | 1190 | −651 | −707 | −49.3 | Argus I |
| S2 | 315 | 500 | −1331 | −850 | −28.4 | Argus II |
| S3 | 144 | 1414 | −2142 | 102 | −53.9 | Argus II |
| S4 | 437 | 1420 | −1807 | 401 | −22.1 | Argus II |

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v --cov=retinalsim
```

## Project structure

```
retinalsim/
├── retinalsim/
│   ├── __init__.py          # Public API
│   ├── __main__.py          # python -m retinalsim entry point
│   ├── axon_map.py          # Core model (Jansonius + Beyeler)
│   ├── constants.py         # Physical constants & subject presets
│   └── server.py            # FastAPI web server
├── tests/
│   └── test_retinalsim.py   # Comprehensive test suite
├── frontend/
│   └── index.html           # Interactive web UI
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── nginx.conf
│   └── requirements.txt
├── examples/
│   ├── basic_simulation.py
│   └── beyeler_subjects.py
├── paper.md                 # JOSS paper
├── paper.bib                # References
├── pyproject.toml           # Package configuration
├── LICENSE                  # MIT License
├── CITATION.cff             # Citation metadata
└── CONTRIBUTING.md          # Contribution guidelines
```

## Coordinate system

RetinalSim uses a fovea-centered retinal coordinate system (right eye, fundus view):

- **x+** = nasal (toward optic disc)
- **x−** = temporal
- **y+** = superior
- **y−** = inferior
- **Origin** = fovea center
- **Units** = micrometers (μm)
- **Conversion** = 301.5 μm/deg (Drasdo & Fowler 1974)

## Citation

If you use RetinalSim in your research, please cite:

```bibtex
@article{retinalsim2026,
  title     = {RetinalSim: An open-source retinal prosthesis simulator based on the axon map model},
  author    = {Juit},
  journal   = {Journal of Open Source Software},
  year      = {2026},
  doi       = {10.21105/joss.XXXXX},
}
```

And the underlying model:

```bibtex
@article{Beyeler2019,
  author  = {Beyeler, Michael and Boynton, Geoffrey M. and Fine, Ione and Rokem, Ariel},
  title   = {Model of Ganglion Axon Pathways Accounts for Percepts Elicited by Retinal Implants},
  journal = {Scientific Reports},
  volume  = {9},
  pages   = {9199},
  year    = {2019},
  doi     = {10.1038/s41598-019-45416-4},
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Related projects

- [pulse2percept](https://github.com/pulse2percept/pulse2percept) — comprehensive prosthetic vision simulation framework
- [Beyeler et al. (2019)](https://doi.org/10.1038/s41598-019-45416-4) — original axon map model paper
