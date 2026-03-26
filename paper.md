---
title: 'RetinalSim: An open-source retinal prosthesis simulator based on the axon map model'
tags:
  - Python
  - retinal prosthesis
  - bionic eye
  - axon map model
  - prosthetic vision
  - phosphene simulation
authors:
  - name: Juit
    orcid: 0009-0009-3091-3656
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 2026-03-24
bibliography: paper.bib
---

# Summary

RetinalSim is a Python package and interactive web application for simulating
prosthetic vision produced by epiretinal implants such as the Argus II.
The simulator implements the axon map model of @Beyeler2019, which predicts
the elongated phosphene shapes ("streaks") that patients perceive due to
incidental activation of passing retinal nerve fiber bundles. RetinalSim
provides a fast, self-contained implementation with a NumPy-vectorized
computation pipeline, four image encoding strategies, and an interactive
web interface for real-time exploration of model parameters.

# Statement of need

Retinal prostheses aim to restore functional vision in patients with
degenerative retinal diseases by electrically stimulating surviving retinal
neurons. However, the percepts produced by epiretinal devices differ
substantially from natural vision: instead of point-like phosphenes, patients
often perceive elongated streaks whose shape follows the underlying nerve
fiber bundle trajectories [@Nanduri2012; @Beyeler2019].

Accurate simulation of these percepts is essential for three purposes:
(1) developing image processing strategies that optimize the information
conveyed to patients, (2) informing surgical planning by predicting how
implant placement affects visual outcomes, and (3) educating clinicians
and the public about the current capabilities and limitations of retinal
prostheses.

The reference implementation of the axon map model is available within the
pulse2percept package [@Beyeler2017], a comprehensive library for simulating
prosthetic vision across multiple implant types. While pulse2percept provides
broad functionality, its complexity can present a barrier for researchers
who need a focused, lightweight tool for epiretinal simulation. RetinalSim
addresses this gap by providing a standalone implementation that can be
used as a Python library, as a web API, or through an interactive browser
interface — making the axon map model accessible to researchers, clinicians,
and educators without requiring Python expertise.

# Software design

RetinalSim implements the three components of the Beyeler axon map model:

1. **Nerve fiber bundle trajectories** following the Jansonius model
   [@Jansonius2009; @Jansonius2012], which traces axon paths from retinal
   ganglion cell somas to the optic disc using empirically fitted equations
   with hemifield-specific curvature parameters.

2. **Electrode–axon sensitivity computation** using Beyeler's Gaussian decay
   model, where each electrode's contribution to a ganglion cell depends on
   the radial distance to its axon segments (parameter $\rho$) and the
   cumulative arc length along the axon from the soma (parameter $\lambda$).

3. **Image-to-current encoding** with four strategies (direct luminance,
   Canny edge detection, CLAHE local contrast, and gradient-based saliency)
   that map camera input to electrode stimulation currents using area
   averaging over each electrode's physical footprint.

The sensitivity matrix computation is vectorized using NumPy array
operations, avoiding per-cell Python loops. Axon paths are packed into
padded arrays that enable broadcasting across all ganglion cells
simultaneously, reducing computation time from approximately 80 seconds
to under 2 seconds for a 76×76 grid on commodity hardware.

The model enforces two anatomical constraints that prevent physiologically
implausible percepts: (1) raphe termination, which truncates nerve fiber
bundles that cross the horizontal raphe in the temporal retina, and
(2) hemifield assignment, which restricts ganglion cell–axon matching to
the same retinal hemifield using a sigmoid blending function that
produces a smooth transition at the raphe rather than a hard boundary,
preventing cross-raphe signal propagation while avoiding visible seam
artifacts in the simulated percept.

Electrode array parameters — position, rotation, and per-subject
$\rho$ and $\lambda$ values from @Beyeler2019 Table 2 — are adjustable
through both the Python API and the web interface, enabling reproduction
of individual patient outcomes.

# Key features

- Pure Python with NumPy vectorization — no compiled extensions required
- Interactive web interface (FastAPI + single HTML page) with real-time
  parameter adjustment
- Docker-based deployment for institutional or classroom use
- Argus II and PRIMA electrode array models
- Built-in test patterns and SSIM quality metric
- Axon map visualization showing fiber trajectories and electrode placement
- Validated against published subject data from @Beyeler2019

# State of the field

The reference implementation of the axon map model is available within the
pulse2percept package [@Beyeler2017]. RetinalSim differs in scope and design:

| | pulse2percept | RetinalSim |
|---|---|---|
| Scope | Multi-model framework (scoreboard, axon map, cortical) | Focused axon map implementation |
| Dependencies | NumPy, Cython, scikit-image, scipy, joblib | NumPy, OpenCV only |
| Interface | Python API | Python API + web UI + REST API |
| Deployment | `pip install` | `pip install` or Docker one-command |
| Target users | Researchers building on multiple models | Educators, clinicians, rapid prototyping |

To verify numerical agreement, we compared single-electrode phosphene
shapes across Beyeler 2019 Subjects S2, S3, and S4 using identical
parameters ($\rho$, $\lambda$, array position, rotation). RetinalSim
reproduces the expected qualitative behavior described in @Beyeler2019:
Subject S2 ($\rho=315$, $\lambda=500$) produces moderately spread
phosphenes with short streaks; Subject S3 ($\rho=144$, $\lambda=1414$)
produces narrow, elongated streaks; and Subject S4 ($\rho=437$,
$\lambda=1420$) produces wide, heavily smeared percepts. A parameter
sweep confirms that $\rho$ controls streak width while $\lambda$
controls streak length, and that the model correctly degrades to the
scoreboard model when both parameters approach zero. The validation
script and results are included in the repository
(`examples/validation_beyeler.py`).

Minor differences arise from implementation choices in axon path
discretization and raphe handling. RetinalSim uses a sigmoid-blended
hemifield constraint with post-hoc raphe smoothing, while pulse2percept
uses a different raphe transition strategy. These differences do not
meaningfully affect the predicted phosphene shapes at clinically relevant
parameter ranges.

# Research impact statement

RetinalSim enables three categories of research activity: (1) rapid
prototyping of image encoding strategies for epiretinal prostheses
without requiring the full pulse2percept dependency stack, (2) classroom
and clinical education through the interactive web interface, which
allows non-programmers to explore how implant parameters affect visual
outcomes, and (3) patient-specific outcome prediction by loading
individual $\rho$ and $\lambda$ parameters fitted from phosphene drawing data. The
Docker deployment option allows institutions to host the simulator
internally for research or teaching without per-user software installation.

# AI usage disclosure

This project was developed with assistance from Claude (Anthropic, Claude
Opus 4), which was used for code generation, refactoring, test scaffolding,
documentation drafting, and paper authoring. All AI-assisted outputs were
reviewed, edited, and validated by the human author, who made all core
design decisions including model architecture, equation implementation,
and validation methodology.

# Acknowledgements

This work builds directly on the axon map model described by @Beyeler2019
and the nerve fiber bundle trajectory model of @Jansonius2009 and
@Jansonius2012. The author thanks the pulse2percept team for their
foundational work in open-source prosthetic vision simulation.

# References
