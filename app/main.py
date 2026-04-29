from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.analysis import AnalysisService, read_image
from app.config import DATA_DIR, POSTER_TEMPLATE_DIR, UPLOAD_DIR
from app.database import fetch_filter_options, fetch_latest_by_roi, fetch_results, init_db
from app.roi_store import ConfigStore
from app.schemas import ROI


app = FastAPI(title="MVP1 Franchise Quality Monitor")
templates = Jinja2Templates(directory="app/templates")
config_store = ConfigStore()
analysis_service = AnalysisService()

init_db()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")


def save_upload(upload: UploadFile, target_dir: Path, fallback_stem: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".bin"
    target_path = target_dir / f"{fallback_stem}{suffix}"
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return target_path


def config_payload(config_id: str | None) -> dict[str, Any] | None:
    if not config_id:
        return None
    try:
        return config_store.load(config_id).to_dict()
    except FileNotFoundError:
        return None


def with_human_clear_ratio(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    if "visible_ratio" in payload and "human_clear_ratio" not in payload:
        payload["human_clear_ratio"] = payload["visible_ratio"]
    return payload


@app.get("/")
def dashboard(request: Request) -> Any:
    configs = config_store.list_configs()
    stores = config_store.list_store_names()
    latest_pop = [with_human_clear_ratio(row) for row in fetch_latest_by_roi("POP")]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "title": "Dashboard",
            "configs": configs,
            "stores": stores,
            "latest_pop": latest_pop,
            "absent_pop_count": len([row for row in latest_pop if row["decision"] == "Absent"]),
        },
    )


@app.get("/stores")
def stores_page(request: Request) -> Any:
    store_summaries = config_store.list_store_summaries()
    return templates.TemplateResponse(
        request,
        "stores.html",
        {
            "request": request,
            "title": "Stores",
            "stores": store_summaries,
        },
    )


@app.get("/setup")
def setup_page(
    request: Request,
    config_id: str | None = Query(default=None),
    store_name: str | None = Query(default=None),
) -> Any:
    store_summaries = config_store.list_store_summaries()
    selected = config_payload(config_id)
    selected_store_name = selected["store_name"] if selected else (store_name or "")
    selected_store_configs = config_store.list_configs_by_store(selected_store_name) if selected_store_name else []
    return templates.TemplateResponse(
        request,
        "setup.html",
        {
            "request": request,
            "title": "ROI Setup",
            "stores": store_summaries,
            "selected_store_name": selected_store_name,
            "selected_store_configs": selected_store_configs,
            "selected_config": selected,
        },
    )


@app.post("/stores")
async def create_store(store_name: str = Form(...)) -> RedirectResponse:
    try:
        normalized = store_name.strip()
        config_store.add_store_name(normalized)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url=f"/stores?store_name={quote_plus(normalized)}", status_code=303)


@app.post("/setup/save")
async def save_setup(
    store_name: str = Form(...),
    cctv_nickname: str = Form(...),
    rois_json: str = Form(...),
    existing_reference_path: str = Form(default=""),
    reference_image: UploadFile | None = File(default=None),
) -> RedirectResponse:
    try:
        raw_rois = json.loads(rois_json)
        rois = [ROI.from_dict(item) for item in raw_rois]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid roi payload: {exc}") from exc

    temp_path: Path | None = None
    if reference_image and reference_image.filename:
        suffix = Path(reference_image.filename).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(reference_image.file, temp_file)
            temp_path = Path(temp_file.name)

    try:
        config = config_store.save_config(
            store_name=store_name,
            cctv_nickname=cctv_nickname,
            rois=rois,
            reference_image_source=temp_path,
            existing_reference_path=existing_reference_path or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

    return RedirectResponse(
        url=f"/setup?store_name={quote_plus(config.store_name)}&config_id={config.config_id}",
        status_code=303,
    )


@app.get("/api/configs")
def list_configs() -> list[dict[str, Any]]:
    return [config.to_dict() for config in config_store.list_configs()]


@app.get("/api/stores")
def list_stores() -> list[dict[str, int | str]]:
    return config_store.list_store_summaries()


@app.get("/api/configs/{config_id}")
def get_config(config_id: str) -> dict[str, Any]:
    try:
        return config_store.load(config_id).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc


@app.get("/api/configs/{config_id}/rois/{roi_name}")
def get_roi(config_id: str, roi_name: str) -> dict[str, Any]:
    try:
        return config_store.get_roi(config_id, roi_name).to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc


@app.get("/analyze")
def analyze_page(request: Request, config_id: str | None = Query(default=None)) -> Any:
    configs = config_store.list_configs()
    poster_templates = sorted(POSTER_TEMPLATE_DIR.glob("*"))
    selected = config_payload(config_id)
    return templates.TemplateResponse(
        request,
        "analyze.html",
        {
            "request": request,
            "title": "Analysis",
            "configs": configs,
            "configs_payload": [config.to_dict() for config in configs],
            "poster_templates": [path.name for path in poster_templates],
            "selected_config": selected,
        },
    )


@app.post("/analyze")
async def analyze_upload(
    request: Request,
    config_id: str = Form(...),
    roi_name: str = Form(...),
    sensor_brightness: float | None = Form(default=None),
    enable_sensor_match: bool = Form(default=False),
    media_file: UploadFile = File(...),
    poster_template_upload: UploadFile | None = File(default=None),
    poster_template_name: str = Form(default=""),
) -> Any:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    media_path = save_upload(media_file, UPLOAD_DIR, f"{config_id}_{roi_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}")

    if poster_template_upload and poster_template_upload.filename:
        poster_path = save_upload(
            poster_template_upload,
            POSTER_TEMPLATE_DIR,
            f"{config_id}_{roi_name}_poster_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        )
    elif poster_template_name:
        poster_path = POSTER_TEMPLATE_DIR / poster_template_name
        if not poster_path.exists():
            raise HTTPException(status_code=400, detail="poster template not found")
    else:
        raise HTTPException(status_code=400, detail="poster template is required")

    try:
        result = analysis_service.analyze_media(
            config=config,
            roi=roi,
            media_path=media_path,
            poster_template_path=poster_path,
            sensor_brightness=sensor_brightness,
            enable_sensor_match=enable_sensor_match,
        )
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "GEMINI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc

    configs = config_store.list_configs()
    return templates.TemplateResponse(
        request,
        "analyze.html",
        {
            "request": request,
            "title": "Analysis",
            "configs": configs,
            "configs_payload": [item.to_dict() for item in configs],
            "poster_templates": [path.name for path in sorted(POSTER_TEMPLATE_DIR.glob("*"))],
            "selected_config": config.to_dict(),
            "analysis_result": result,
        },
    )


@app.post("/api/validate-image")
async def validate_image_api(
    config_id: str = Form(...),
    roi_name: str = Form(...),
    image_file: UploadFile = File(...),
) -> JSONResponse:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    image_path = save_upload(image_file, UPLOAD_DIR, f"{config_id}_{roi_name}_validate_image")
    try:
        validation = analysis_service.validator.validate_image(read_image(image_path), roi, source_path=image_path)
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "GEMINI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return JSONResponse(
        {
            "is_valid": validation.is_valid,
            "human_clear_ratio": round(validation.visible_ratio, 3),
            "occlusion_level": validation.occlusion_level,
            "summary": validation.summary,
            "reject_reason": validation.reject_reason,
        }
    )


@app.post("/api/validate-video")
async def validate_video_api(
    config_id: str = Form(...),
    roi_name: str = Form(...),
    sensor_brightness: float | None = Form(default=None),
    enable_sensor_match: bool = Form(default=False),
    video_file: UploadFile = File(...),
) -> JSONResponse:
    try:
        config = config_store.load(config_id)
        roi = config_store.get_roi(config_id, roi_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="config not found") from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="roi not found") from exc

    video_path = save_upload(video_file, UPLOAD_DIR, f"{config_id}_{roi_name}_validate_video")
    try:
        validation = analysis_service.validator.validate_video(
            video_path,
            roi,
            sensor_brightness=sensor_brightness,
            enable_sensor_match=enable_sensor_match,
        )
    except RuntimeError as exc:
        detail = str(exc)
        status_code = 503 if "GEMINI_API_KEY" in detail else 502
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return JSONResponse(
        {
            "is_valid": validation.is_valid,
            "human_clear_ratio": round(validation.visible_ratio, 3),
            "occlusion_duration": round(validation.occlusion_duration, 2),
            "brightness_mismatch_duration": round(validation.brightness_mismatch_duration, 2),
            "occlusion_level": validation.occlusion_level,
            "summary": validation.summary,
            "reject_reason": validation.reject_reason,
        }
    )


@app.get("/reports")
def reports_page(
    request: Request,
    store_name: str | None = Query(default=None),
    cctv_id: str | None = Query(default=None),
    roi_name: str | None = Query(default=None),
    decision: str | None = Query(default=None),
    item_type: str | None = Query(default=None),
) -> Any:
    filters = {
        "store_name": store_name,
        "cctv_id": cctv_id,
        "roi_name": roi_name,
        "decision": decision,
        "item_type": item_type,
    }
    records = [with_human_clear_ratio(row) for row in fetch_results(filters)]
    latest_pop = [with_human_clear_ratio(row) for row in fetch_latest_by_roi(roi_name if roi_name else "POP")]
    return templates.TemplateResponse(
        request,
        "reports.html",
        {
            "request": request,
            "title": "Reports",
            "records": records,
            "latest_pop": latest_pop,
            "options": fetch_filter_options(),
            "filters": filters,
        },
    )
