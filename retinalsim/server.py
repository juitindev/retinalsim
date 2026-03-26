"""
FastAPI Backend for Retinal Prosthesis Simulator
================================================
Endpoints:
  GET  /                     → health check
  GET  /test_patterns        → list available test patterns
  POST /simulate             → run simulation, return percept image
  POST /simulate_upload      → run simulation with uploaded image
  POST /single_phosphene     → diagnostic: one electrode phosphene shape

Run directly:
  python -m retinalsim.server

Or with uvicorn:
  uvicorn retinalsim.server:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .axon_map import (
    AxonMapModel,
    EncodingStrategy,
    compute_ssim,
    encode_image,
    generate_test_images,
    make_argus_ii,
    make_prima,
    render_axon_map,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ──
app = FastAPI(title="Retinal Prosthesis Simulator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model cache ──
_model_cache: dict = {}


def _get_or_build_model(
    rho: float, axon_lambda: float,
    grid_spacing: float = 80.0,
    x_range: tuple = (-3000, 3000),
    y_range: tuple = (-3000, 3000)
) -> AxonMapModel:
    """Cache model by (rho, lambda, grid_spacing) to avoid rebuilding.
    Only keeps 1 cached model to limit memory on small servers.
    """
    key = (rho, axon_lambda, grid_spacing)
    if key not in _model_cache:
        _model_cache.clear()
        import gc; gc.collect()

        logger.info(f"Building new model: rho={rho}, lambda={axon_lambda}, grid={grid_spacing}")
        model = AxonMapModel(
            rho=rho,
            axon_lambda=axon_lambda,
            grid_spacing=grid_spacing,
            x_range=x_range,
            y_range=y_range,
            axon_n_bundles=200,
        )
        model.build()
        _model_cache[key] = model
        logger.info("Model built and cached.")
    return _model_cache[key]


def _ndarray_to_base64_png(arr: np.ndarray, origin: str = 'lower') -> str:
    """Convert a [0,1] float array to base64-encoded PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(arr, cmap='gray', vmin=0, vmax=1, origin=origin)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _render_axon_map_base64(model: AxonMapModel, electrodes) -> str:
    """Generate axon map visualization as base64 PNG."""
    png_bytes = render_axon_map(
        axon_bundles=model.axon_bundles,
        bundle_phi0s=model._bundle_phi0s,
        electrodes=electrodes,
    )
    return base64.b64encode(png_bytes).decode('utf-8')


# ── Request / Response Models ──

class SimulateRequest(BaseModel):
    rho: float = 200.0
    axon_lambda: float = 800.0
    encoding: str = "direct"
    current_amplitude: float = 1.0
    array_type: str = "argus_ii"
    test_pattern: Optional[str] = None
    grid_spacing: float = 80.0
    array_x: float = -1000.0
    array_y: float = 0.0
    array_rotation: float = -25.0


class SimulateResponse(BaseModel):
    percept_base64: str
    input_base64: str
    axon_map_base64: str
    ssim: float
    elapsed_seconds: float
    grid_shape: list


class PhospheneRequest(BaseModel):
    rho: float = 200.0
    axon_lambda: float = 800.0
    electrode_index: int = 30
    grid_spacing: float = 80.0
    array_x: float = -1000.0
    array_y: float = 0.0
    array_rotation: float = -25.0


# ── Endpoints ──

@app.get("/")
def health():
    return {"status": "ok", "model_cache_size": len(_model_cache)}


@app.get("/test_patterns")
def list_test_patterns():
    return {"patterns": ["letter_E", "grating", "dot_superior", "dot_inferior"]}


@app.get("/preview_pattern/{name}")
def preview_pattern(name: str):
    """Return a test pattern image as base64 PNG for preview."""
    patterns = generate_test_images()
    if name not in patterns:
        raise HTTPException(404, f"Unknown pattern: {name}")
    return {"image_base64": _ndarray_to_base64_png(patterns[name], 'upper')}


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    t0 = time.time()
    model = _get_or_build_model(req.rho, req.axon_lambda, req.grid_spacing)

    electrodes = (make_prima(center_x=req.array_x, center_y=req.array_y, rotation_deg=req.array_rotation)
                  if req.array_type == "prima"
                  else make_argus_ii(center_x=req.array_x, center_y=req.array_y, rotation_deg=req.array_rotation))

    if not req.test_pattern:
        raise HTTPException(400, "Provide test_pattern name, or use /simulate_upload for images")

    patterns = generate_test_images()
    if req.test_pattern not in patterns:
        raise HTTPException(400, f"Unknown pattern: {req.test_pattern}")
    input_img = patterns[req.test_pattern]

    try:
        strategy = EncodingStrategy(req.encoding)
    except ValueError:
        raise HTTPException(400, f"Unknown encoding: {req.encoding}")

    electrodes = encode_image(input_img, electrodes, strategy,
                              current_amplitude=req.current_amplitude)
    S = model.compute_sensitivity_matrix(electrodes)
    percept = model.simulate(electrodes, S)
    ssim = compute_ssim(input_img, percept)
    elapsed = time.time() - t0
    logger.info(f"Simulation done in {elapsed:.1f}s, SSIM={ssim:.4f}")

    return SimulateResponse(
        percept_base64=_ndarray_to_base64_png(percept),
        input_base64=_ndarray_to_base64_png(input_img, 'upper'),
        axon_map_base64=_render_axon_map_base64(model, electrodes),
        ssim=ssim, elapsed_seconds=round(elapsed, 2),
        grid_shape=list(model._grid_shape),
    )


@app.post("/simulate_upload", response_model=SimulateResponse)
async def simulate_upload(
    image: UploadFile = File(...),
    rho: float = Form(200.0),
    axon_lambda: float = Form(800.0),
    encoding: str = Form("direct"),
    current_amplitude: float = Form(1.0),
    array_type: str = Form("argus_ii"),
    grid_spacing: float = Form(80.0),
    array_x: float = Form(-1000.0),
    array_y: float = Form(0.0),
    array_rotation: float = Form(-25.0),
):
    t0 = time.time()
    model = _get_or_build_model(rho, axon_lambda, grid_spacing)

    electrodes = (make_prima(center_x=array_x, center_y=array_y, rotation_deg=array_rotation)
                  if array_type == "prima"
                  else make_argus_ii(center_x=array_x, center_y=array_y, rotation_deg=array_rotation))

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    input_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise HTTPException(400, "Could not decode uploaded image")
    input_img = input_img.astype(float) / 255.0

    try:
        strategy = EncodingStrategy(encoding)
    except ValueError:
        raise HTTPException(400, f"Unknown encoding: {encoding}")

    electrodes = encode_image(input_img, electrodes, strategy,
                              current_amplitude=current_amplitude)
    S = model.compute_sensitivity_matrix(electrodes)
    percept = model.simulate(electrodes, S)
    ssim = compute_ssim(input_img, percept)
    elapsed = time.time() - t0
    logger.info(f"Upload sim done in {elapsed:.1f}s, SSIM={ssim:.4f}")

    return SimulateResponse(
        percept_base64=_ndarray_to_base64_png(percept),
        input_base64=_ndarray_to_base64_png(input_img, 'upper'),
        axon_map_base64=_render_axon_map_base64(model, electrodes),
        ssim=ssim, elapsed_seconds=round(elapsed, 2),
        grid_shape=list(model._grid_shape),
    )


@app.post("/single_phosphene")
async def single_phosphene(req: PhospheneRequest):
    """Diagnostic: show phosphene shape for a single electrode."""
    t0 = time.time()

    model = _get_or_build_model(req.rho, req.axon_lambda, req.grid_spacing)
    electrodes = make_argus_ii(
        center_x=req.array_x,
        center_y=req.array_y,
        rotation_deg=req.array_rotation,
    )

    if req.electrode_index >= len(electrodes):
        raise HTTPException(400, f"Electrode index {req.electrode_index} out of range (max {len(electrodes)-1})")

    for e in electrodes:
        e.current = 0.0
    electrodes[req.electrode_index].current = 1.0

    S = model.compute_sensitivity_matrix(electrodes)
    percept = model.simulate(electrodes, S)

    elapsed = time.time() - t0

    return {
        "percept_base64": _ndarray_to_base64_png(percept),
        "axon_map_base64": _render_axon_map_base64(model, electrodes),
        "electrode_x": electrodes[req.electrode_index].x,
        "electrode_y": electrodes[req.electrode_index].y,
        "electrode_index": req.electrode_index,
        "elapsed_seconds": round(elapsed, 2),
    }


# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
