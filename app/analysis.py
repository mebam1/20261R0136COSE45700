from __future__ import annotations

import json
import mimetypes
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from google import genai
from google.genai import types

from app.config import (
    ANALYSIS_CROP_DIR,
    BRIGHTNESS_MISMATCH_THRESHOLD,
    DARKNESS_THRESHOLD,
    GEMINI_API_KEY,
    GEMINI_FILE_POLL_INTERVAL_SECONDS,
    GEMINI_FILE_POLL_TIMEOUT_SECONDS,
    GEMINI_MODEL,
    MAX_VISIBILITY_SAMPLE_STEP_FRAMES,
    OCCLUSION_THRESHOLD,
    PERSISTENT_MISMATCH_SECONDS,
    UNKNOWN_CONFIDENCE_THRESHOLD,
    VISIBILITY_SAMPLE_SECONDS,
    VISIBILITY_THRESHOLD,
)
from app.database import insert_result, truncate_to_hour
from app.schemas import CCTVConfig, ROI


def emit_console_alert(alert_type: str, payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "type": alert_type,
                "payload": payload,
            },
            ensure_ascii=False,
        )
    )


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    extension = path.suffix.lower() or ".png"
    if extension == ".jpeg":
        extension = ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError(f"cannot encode image for {path}")
    encoded.tofile(str(path))


def guess_mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        return guessed
    if path.suffix.lower() == ".mp4":
        return "video/mp4"
    if path.suffix.lower() == ".png":
        return "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def open_video_capture(video_path: Path) -> tuple[cv2.VideoCapture, Path | None]:
    temp_copy: Path | None = None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=video_path.suffix, prefix="mvp1_video_")
        temp_copy = Path(temp_file.name)
        temp_file.close()
        shutil.copy2(video_path, temp_copy)
        capture = cv2.VideoCapture(str(temp_copy))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    return capture, temp_copy


def ordered_points(roi: ROI) -> np.ndarray:
    points = np.array(roi.point_pairs(), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def warp_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    source = ordered_points(roi)

    width_top = np.linalg.norm(source[1] - source[0])
    width_bottom = np.linalg.norm(source[2] - source[3])
    height_right = np.linalg.norm(source[2] - source[1])
    height_left = np.linalg.norm(source[3] - source[0])

    target_width = max(int(round(max(width_top, width_bottom))), 8)
    target_height = max(int(round(max(height_left, height_right))), 8)

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (target_width, target_height))


def crop_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    return warp_roi(image, roi)


def mean_brightness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)


def visibility_sample_step(fps: float) -> int:
    return max(1, min(int(round(fps * VISIBILITY_SAMPLE_SECONDS)), MAX_VISIBILITY_SAMPLE_STEP_FRAMES))


def sample_video_frames(video_path: Path) -> tuple[list[np.ndarray], float]:
    capture, temp_copy = open_video_capture(video_path)

    fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
    step = visibility_sample_step(fps)
    frames: list[np.ndarray] = []
    index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if index % step == 0:
                frames.append(frame)
            index += 1
    finally:
        capture.release()
        if temp_copy and temp_copy.exists():
            temp_copy.unlink(missing_ok=True)
    return frames, max(step / fps, 1.0 / fps)


def build_cropped_image_file(image: np.ndarray, roi: ROI) -> Path:
    crop = crop_roi(image, roi)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=f"mvp1_{roi.name.lower()}_")
    temp_path = Path(temp_file.name)
    temp_file.close()
    write_image(temp_path, crop)
    return temp_path


def save_analysis_crop(image: np.ndarray, roi: ROI, stem: str) -> Path:
    crop = crop_roi(image, roi)
    output_path = ANALYSIS_CROP_DIR / f"{stem}.png"
    write_image(output_path, crop)
    return output_path


def build_cropped_video_file(video_path: Path, roi: ROI) -> Path:
    capture, temp_copy = open_video_capture(video_path)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", prefix=f"mvp1_{roi.name.lower()}_")
    output_path = Path(temp_file.name)
    temp_file.close()

    writer: cv2.VideoWriter | None = None
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 1.0
        success, frame = capture.read()
        if not success:
            raise ValueError(f"video contains no readable frames: {video_path}")

        first_crop = crop_roi(frame, roi)
        height, width = first_crop.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(f"cannot create cropped video: {output_path}")

        writer.write(first_crop)
        while True:
            success, frame = capture.read()
            if not success:
                break
            writer.write(crop_roi(frame, roi))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if temp_copy and temp_copy.exists():
            temp_copy.unlink(missing_ok=True)

    return output_path


@dataclass
class ValidationResult:
    media_type: str
    is_valid: bool
    visible_ratio: float
    occlusion_duration: float
    brightness_mismatch_duration: float
    occlusion_level: str
    average_brightness: float
    average_blur: float
    sampled_frame_count: int
    representative_frame: np.ndarray | None
    summary: str
    reject_reason: str | None


VISIBILITY_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "human_body_ratio": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Estimated proportion of the ROI area occupied by visible human body parts only.",
        },
        "summary": {
            "type": "string",
            "description": "Short explanation grounded only in visible human body occlusion.",
        },
    },
    "required": ["human_body_ratio", "summary"],
    "propertyOrdering": ["human_body_ratio", "summary"],
}


@dataclass
class GeminiVisibilityAssessment:
    human_body_ratio: float
    summary: str


class GeminiFileClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        client: genai.Client | None = None,
    ) -> None:
        self.api_key = api_key or GEMINI_API_KEY
        self.model = model
        self.client = client
        self.poll_interval_seconds = GEMINI_FILE_POLL_INTERVAL_SECONDS
        self.poll_timeout_seconds = GEMINI_FILE_POLL_TIMEOUT_SECONDS

    def _get_client(self) -> genai.Client:
        if self.client is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is required for Gemini-based analysis.")
            self.client = genai.Client(api_key=self.api_key)
        return self.client

    def _upload_and_prepare_file(self, path: Path) -> types.File:
        client = self._get_client()
        uploaded = client.files.upload(
            file=path,
            config=types.UploadFileConfig(
                mime_type=guess_mime_type(path),
                display_name=path.name,
            ),
        )
        return self._wait_until_active(uploaded)

    def _wait_until_active(self, uploaded_file: types.File) -> types.File:
        client = self._get_client()
        deadline = time.monotonic() + self.poll_timeout_seconds
        current = uploaded_file

        while True:
            state = getattr(current, "state", None)
            state_name = getattr(state, "name", None) if state is not None else None
            if state_name in (None, "ACTIVE"):
                return current
            if state_name == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {current.name}.")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for Gemini file processing: {current.name}")
            time.sleep(self.poll_interval_seconds)
            current = client.files.get(name=current.name)

    def _delete_file_quietly(self, uploaded_file: types.File | None) -> None:
        if uploaded_file is None or not getattr(uploaded_file, "name", None):
            return
        try:
            self._get_client().files.delete(name=uploaded_file.name)
        except Exception:
            pass


class GeminiVisibilityClient(GeminiFileClient):
    def analyze_visibility(
        self,
        *,
        cropped_media_path: Path,
        media_type: str,
        roi_name: str,
        source_path: Path | None = None,
    ) -> GeminiVisibilityAssessment:
        _ = source_path
        client = self._get_client()
        media_file: types.File | None = None

        prompt = (
            "You are validating whether a cropped store ROI is blocked by people.\n"
            f"The provided file is a cropped ROI {media_type} for ROI named {roi_name}.\n"
            "Estimate only how much of the ROI area is occupied by visible human body parts.\n"
            "Count head, face, hair, torso, arms, hands, legs, feet, and clothing worn by a person.\n"
            "Do not count posters, POP materials, shelves, products, counters, walls, reflections, or any static fixture as occlusion.\n"
            "A poster fully covering the ROI but with no person in front of it should still yield a human_body_ratio near 0.\n"
            "Return only JSON."
        )

        try:
            media_file = self._upload_and_prepare_file(cropped_media_path)
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, media_file],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_json_schema=VISIBILITY_ANALYSIS_SCHEMA,
                ),
            )

            payload = response.parsed
            if payload is None:
                payload = json.loads(response.text)
            elif hasattr(payload, "model_dump"):
                payload = payload.model_dump()

            human_body_ratio = float(np.clip(float(payload["human_body_ratio"]), 0.0, 1.0))
            summary = str(payload["summary"])
            return GeminiVisibilityAssessment(human_body_ratio=human_body_ratio, summary=summary)
        finally:
            self._delete_file_quietly(media_file)


class VideoValidator:
    def __init__(self, visibility_client: GeminiVisibilityClient | None = None) -> None:
        self.visibility_threshold = VISIBILITY_THRESHOLD
        self.occlusion_threshold = OCCLUSION_THRESHOLD
        self.darkness_threshold = DARKNESS_THRESHOLD
        self.brightness_mismatch_threshold = BRIGHTNESS_MISMATCH_THRESHOLD
        self.persistent_mismatch_seconds = PERSISTENT_MISMATCH_SECONDS
        self.visibility_client = visibility_client or GeminiVisibilityClient()

    def _assess_frame_visibility(
        self,
        current_crop: np.ndarray,
        roi: ROI,
        source_path: Path | None = None,
    ) -> GeminiVisibilityAssessment:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=f"mvp1_{roi.name.lower()}_visibility_")
        cropped_frame_path = Path(temp_file.name)
        temp_file.close()
        write_image(cropped_frame_path, current_crop)
        try:
            return self.visibility_client.analyze_visibility(
                cropped_media_path=cropped_frame_path,
                media_type="image",
                roi_name=roi.name,
                source_path=source_path,
            )
        finally:
            cropped_frame_path.unlink(missing_ok=True)

    def validate_image(
        self,
        current_image: np.ndarray,
        roi: ROI,
        source_path: Path | None = None,
    ) -> ValidationResult:
        current_crop = crop_roi(current_image, roi)
        visibility_assessment = self._assess_frame_visibility(current_crop, roi, source_path=source_path)
        brightness = mean_brightness(current_crop)
        visible_ratio = float(np.clip(1.0 - visibility_assessment.human_body_ratio, 0.0, 1.0))

        reject_reason = None
        if visible_ratio < self.visibility_threshold:
            reject_reason = "occluded"
        elif brightness < self.darkness_threshold:
            reject_reason = "too_dark"

        summary = (
            f"ROI {roi.name} image visibility={visible_ratio:.2f}, "
            f"human_body_ratio={visibility_assessment.human_body_ratio:.2f}, "
            f"brightness={brightness:.2f}; {visibility_assessment.summary}"
        )
        return ValidationResult(
            media_type="image",
            is_valid=reject_reason is None,
            visible_ratio=visible_ratio,
            occlusion_duration=0.0,
            brightness_mismatch_duration=0.0,
            occlusion_level=self.occlusion_level(visible_ratio),
            average_brightness=brightness,
            average_blur=0.0,
            sampled_frame_count=1,
            representative_frame=current_image,
            summary=summary,
            reject_reason=reject_reason,
        )

    def validate_video(
        self,
        video_path: Path,
        roi: ROI,
        sensor_brightness: float | None = None,
        enable_sensor_match: bool = True,
    ) -> ValidationResult:
        frames, frame_interval = sample_video_frames(video_path)
        if not frames:
            raise ValueError("video contains no readable frames")

        occlusion_ratios: list[float] = []
        brightness_values: list[float] = []
        visibility_summaries: list[str] = []
        representative_frame: np.ndarray | None = None
        representative_score = -1.0
        mismatch_duration = 0.0
        consecutive_mismatch = 0.0
        max_consecutive_mismatch = 0.0

        for frame in frames:
            current_crop = crop_roi(frame, roi)
            visibility_assessment = self._assess_frame_visibility(current_crop, roi, source_path=video_path)
            occlusion_ratio = visibility_assessment.human_body_ratio
            brightness = mean_brightness(current_crop)
            visibility = float(np.clip(1.0 - occlusion_ratio, 0.0, 1.0))

            occlusion_ratios.append(occlusion_ratio)
            brightness_values.append(brightness)
            visibility_summaries.append(visibility_assessment.summary)

            quality_score = visibility - max(0.0, self.darkness_threshold - brightness)
            if quality_score > representative_score:
                representative_score = quality_score
                representative_frame = frame.copy()

            if enable_sensor_match and sensor_brightness is not None:
                if abs(brightness - sensor_brightness) > self.brightness_mismatch_threshold:
                    consecutive_mismatch += frame_interval
                    mismatch_duration += frame_interval
                    max_consecutive_mismatch = max(max_consecutive_mismatch, consecutive_mismatch)
                else:
                    consecutive_mismatch = 0.0

        visible_ratio = float(np.mean([1.0 - ratio for ratio in occlusion_ratios]))
        occlusion_duration = float(sum(frame_interval for ratio in occlusion_ratios if ratio >= self.occlusion_threshold))
        average_brightness = float(np.mean(brightness_values))

        reject_reason = None
        if visible_ratio < self.visibility_threshold:
            reject_reason = "occluded"
        elif average_brightness < self.darkness_threshold:
            reject_reason = "too_dark"

        if enable_sensor_match and sensor_brightness is not None and max_consecutive_mismatch >= self.persistent_mismatch_seconds:
            emit_console_alert(
                "cctv_check_request",
                {
                    "roi_name": roi.name,
                    "sensor_brightness": sensor_brightness,
                    "average_video_brightness": round(average_brightness, 3),
                    "mismatch_duration_seconds": round(max_consecutive_mismatch, 2),
                    "message": "sensor/video brightness mismatch persisted; inspect CCTV",
                },
            )

        summary = (
            f"ROI {roi.name} video visibility={visible_ratio:.2f}, human_body_ratio={1.0 - visible_ratio:.2f}, "
            f"occlusion_seconds={occlusion_duration:.1f}, brightness={average_brightness:.2f}; "
            f"Gemini visibility: {visibility_summaries[0] if visibility_summaries else 'n/a'}"
        )

        return ValidationResult(
            media_type="video",
            is_valid=reject_reason is None,
            visible_ratio=visible_ratio,
            occlusion_duration=occlusion_duration,
            brightness_mismatch_duration=round(mismatch_duration, 2),
            occlusion_level=self.occlusion_level(visible_ratio),
            average_brightness=average_brightness,
            average_blur=0.0,
            sampled_frame_count=len(frames),
            representative_frame=representative_frame,
            summary=summary,
            reject_reason=reject_reason,
        )

    @staticmethod
    def occlusion_level(visible_ratio: float) -> str:
        if visible_ratio >= 0.8:
            return "low"
        if visible_ratio >= 0.6:
            return "medium"
        return "high"


@dataclass
class QualityResult:
    status: str
    confidence: float
    summary: str


POP_ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["Present", "Absent", "Unknown"],
            "description": "Present if the target poster is visible, Absent if the ROI is clear and the poster is not there, Unknown if visibility or confidence is insufficient.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Model confidence between 0 and 1.",
        },
        "summary": {
            "type": "string",
            "description": "Short Korean or English explanation grounded in visible evidence only.",
        },
    },
    "required": ["status", "confidence", "summary"],
    "propertyOrdering": ["status", "confidence", "summary"],
}


@dataclass
class GeminiPopAssessment:
    status: str
    confidence: float
    summary: str


class GeminiPopClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = GEMINI_MODEL,
        client: genai.Client | None = None,
    ) -> None:
        self.base_client = GeminiFileClient(api_key=api_key, model=model, client=client)
        self.model = self.base_client.model

    def analyze_pop(self, *, poster_template_path: Path, cropped_media_path: Path, media_type: str) -> GeminiPopAssessment:
        client = self.base_client._get_client()
        template_file: types.File | None = None
        media_file: types.File | None = None

        prompt = (
            "You are verifying whether a target promotional poster is present in a store ROI.\n"
            "The first file is the target poster template.\n"
            f"The second file is the cropped ROI {media_type} from the store.\n"
            "Return Present only if the same target poster is visibly present.\n"
            "Return Absent only if the ROI media is clear enough and the poster is not visible.\n"
            "Return Unknown if the ROI media is too dark, blurry, occluded, too brief, or uncertain.\n"
            "Use only visual evidence from the provided files."
        )

        try:
            template_file = self.base_client._upload_and_prepare_file(poster_template_path)
            media_file = self.base_client._upload_and_prepare_file(cropped_media_path)
            response = client.models.generate_content(
                model=self.model,
                contents=[prompt, template_file, media_file],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_json_schema=POP_ANALYSIS_SCHEMA,
                ),
            )

            payload = response.parsed
            if payload is None:
                payload = json.loads(response.text)
            elif hasattr(payload, "model_dump"):
                payload = payload.model_dump()

            status = str(payload["status"])
            confidence = float(payload["confidence"])
            summary = str(payload["summary"])
            if status not in {"Present", "Absent", "Unknown"}:
                raise RuntimeError(f"Gemini returned invalid POP status: {status}")
            return GeminiPopAssessment(
                status=status,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                summary=summary,
            )
        finally:
            self.base_client._delete_file_quietly(media_file)
            self.base_client._delete_file_quietly(template_file)


class QualityAnalyzer:
    def __init__(self, gemini_client: GeminiPopClient | None = None) -> None:
        self.unknown_confidence_threshold = UNKNOWN_CONFIDENCE_THRESHOLD
        self.gemini_client = gemini_client or GeminiPopClient()

    def analyze_pop(
        self,
        *,
        media_path: Path,
        cropped_media_path: Path,
        media_type: str,
        roi: ROI,
        poster_template_path: Path,
        validation_result: ValidationResult,
    ) -> QualityResult:
        if not validation_result.is_valid:
            return QualityResult(
                status="Unknown",
                confidence=0.0,
                summary=f"POP analysis skipped because the validator rejected ROI {roi.name}. {validation_result.summary}",
            )

        assessment = self.gemini_client.analyze_pop(
            poster_template_path=poster_template_path,
            cropped_media_path=cropped_media_path,
            media_type=media_type,
        )
        confidence = float(np.clip(assessment.confidence, 0.0, 0.99))
        status = assessment.status
        summary = assessment.summary

        if confidence < self.unknown_confidence_threshold and status != "Unknown":
            status = "Unknown"
            summary = f"Gemini confidence was below threshold. {assessment.summary}"

        return QualityResult(status=status, confidence=confidence, summary=summary)


class AnalysisService:
    def __init__(
        self,
        validator: VideoValidator | None = None,
        quality_analyzer: QualityAnalyzer | None = None,
    ) -> None:
        self.validator = validator or VideoValidator()
        self.quality_analyzer = quality_analyzer or QualityAnalyzer()

    def analyze_media(
        self,
        config: CCTVConfig,
        roi: ROI,
        media_path: Path,
        poster_template_path: Path,
        sensor_brightness: float | None = None,
        enable_sensor_match: bool = True,
        analyzed_at: datetime | None = None,
    ) -> dict[str, Any]:
        analyzed_at = analyzed_at or datetime.now()
        crop_preview_path: Path | None = None

        suffix = media_path.suffix.lower()
        media_type = "image" if suffix in {".png", ".jpg", ".jpeg", ".bmp"} else "video"
        if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
            current_image = read_image(media_path)
            validation = self.validator.validate_image(current_image, roi, source_path=media_path)
            crop_preview_path = save_analysis_crop(
                current_image,
                roi,
                f"{config.config_id}_{roi.name}_{analyzed_at.strftime('%Y%m%d%H%M%S')}",
            )
        else:
            validation = self.validator.validate_video(
                media_path,
                roi,
                sensor_brightness=sensor_brightness,
                enable_sensor_match=enable_sensor_match,
            )

        cropped_media_path: Path | None = None
        try:
            if media_type == "image":
                cropped_media_path = build_cropped_image_file(current_image, roi)
            else:
                cropped_media_path = build_cropped_video_file(media_path, roi)

            quality = self.quality_analyzer.analyze_pop(
                media_path=media_path,
                cropped_media_path=cropped_media_path,
                media_type=media_type,
                roi=roi,
                poster_template_path=poster_template_path,
                validation_result=validation,
            )
        finally:
            if cropped_media_path is not None and cropped_media_path.exists():
                cropped_media_path.unlink(missing_ok=True)

        if validation.reject_reason is not None:
            emit_console_alert(
                "reshoot_request",
                {
                    "roi_name": roi.name,
                    "reason": validation.reject_reason,
                    "summary": validation.summary,
                    "shooting_guide": "Hold the camera steady and keep the ROI fully visible for at least 10 seconds.",
                },
            )

        if quality.status == "Unknown":
            emit_console_alert(
                "reanalysis_request",
                {
                    "target_roi_name": roi.name,
                    "model_output_summary": quality.summary,
                    "shooting_guide": "Record the POP ROI from the front for at least 10 seconds without people or fixtures blocking it.",
                },
            )

        stored = {
            "analyzed_at": truncate_to_hour(analyzed_at),
            "store_name": config.store_name,
            "cctv_id": config.cctv_id,
            "cctv_nickname": config.cctv_nickname,
            "roi_name": roi.name,
            "item_type": "POP",
            "decision": quality.status,
            "confidence": round(quality.confidence, 3),
            "visible_ratio": round(validation.visible_ratio, 3),
            "occlusion_duration": round(validation.occlusion_duration, 2),
            "brightness_mismatch_duration": round(validation.brightness_mismatch_duration, 2),
            "summary": f"{validation.summary} | {quality.summary}",
            "source_path": str(media_path),
        }
        if crop_preview_path is not None:
            stored["analysis_crop_path"] = str(crop_preview_path)
            stored["analysis_crop_url"] = "/data/" + crop_preview_path.relative_to(ANALYSIS_CROP_DIR.parent).as_posix()
        insert_result(stored)
        stored["validator"] = validation
        stored["quality"] = quality
        return stored
