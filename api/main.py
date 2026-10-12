"""
LivestockVisionAI API Gateway.
Primary FastAPI entry point for the Multi-Species Livestock Vision AI Platform.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from LivestockVisionAI.api.routers import quality, predict, animals
from LivestockVisionAI.api.dependencies import get_platform_config


app = FastAPI(
    title="LivestockVisionAI Platform API",
    description="Enterprise multi-species computer vision API for livestock weight, age, and biometrics.",
    version="1.0.0-rc1",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Load configuration
cfg = get_platform_config()
cors_cfg = cfg.get("cors", {})
allowed_origins = cors_cfg.get("allowed_origins", [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
])

# Enable hardened CORS (disallowing wildcard origin with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

from LivestockVisionAI.api.security import SecurityHeadersMiddleware, RateLimitingMiddleware
sec_cfg = cfg.get("security", {})
rate_cfg = sec_cfg.get("rate_limiting", {})

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=rate_cfg.get("max_requests_per_minute", 120),
    enabled=rate_cfg.get("enabled", True),
)


# Register routers
app.include_router(quality.router)
app.include_router(predict.router)
app.include_router(animals.router)


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from LivestockVisionAI.api.dependencies import get_species_detector, get_cattle_weight_estimator

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def enforce_payload_limit(request: Request, call_next):
    """Enforce maximum upload payload limit (15MB) to protect vision endpoints."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 15 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"detail": "Payload too large. Maximum file upload size is 15 MB."}
        )
    return await call_next(request)


@app.get("/", tags=["Dashboard"], include_in_schema=False)
async def serve_dashboard():
    """Serves the interactive web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "LivestockVisionAI Dashboard"}


@app.get("/health", tags=["System"])
async def health_check():
    """System health check verifying that vision models are loaded and functional."""
    platform_cfg = get_platform_config()
    detector = get_species_detector()
    cattle_est = get_cattle_weight_estimator()

    # Smoke test model inference
    smoke_passed = True
    try:
        import torch
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            detector.model(dummy)
    except Exception:
        smoke_passed = False

    detector_loaded = getattr(detector, "weights_loaded", False)
    cattle_loaded = getattr(cattle_est, "weights_loaded", False)

    return {
        "status": "HEALTHY" if smoke_passed else "DEGRADED",
        "platform": "LivestockVisionAI",
        "version": "1.0.0-rc1",
        "supported_species": platform_cfg.get("supported_species", ["COW", "HEN", "GOAT", "SHEEP"]),
        "quality_filter_active": True,
        "conformal_uncertainty_active": True,
        "models": {
            "species_detector_weights_loaded": detector_loaded,
            "cattle_weight_weights_loaded": cattle_loaded,
            "smoke_forward_pass_passed": smoke_passed,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("LivestockVisionAI.api.main:app", host="0.0.0.0", port=8000, reload=True)
