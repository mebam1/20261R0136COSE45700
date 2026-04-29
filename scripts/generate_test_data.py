from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import POSTER_TEMPLATE_DIR, TEST_DATA_DIR
from app.roi_store import ConfigStore
from app.schemas import Point, ROI


WIDTH = 640
HEIGHT = 360
FPS = 4


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def create_poster_template(path: Path) -> None:
    image = Image.new("RGB", (120, 170), "#f6e05e")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 112, 162), outline="#1f2937", width=4)
    draw.rectangle((18, 20, 102, 48), fill="#1d4ed8")
    draw.rectangle((18, 60, 102, 96), fill="#ef4444")
    draw.rectangle((18, 110, 102, 148), fill="#16a34a")
    image.save(path)


def roi_points_array(roi: ROI) -> np.ndarray:
    return np.array(roi.point_pairs(), dtype=np.float32)


def ordered_points(roi: ROI) -> np.ndarray:
    points = roi_points_array(roi)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)

    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def shrink_points(points: np.ndarray, factor: float) -> np.ndarray:
    centroid = points.mean(axis=0)
    return centroid + ((points - centroid) * factor)


def base_scene(bg_color: str, accent_color: str, roi: ROI) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), bg_color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 250, WIDTH, HEIGHT), fill="#cbd5e1")
    draw.rectangle((30, 40, 220, 180), fill="#f8fafc", outline="#94a3b8", width=3)
    draw.rectangle((260, 190, 610, 290), fill="#94a3b8")
    draw.polygon(roi.point_pairs(), fill=accent_color, outline="#475569")
    for x in range(40, 200, 30):
        draw.rectangle((x, 75, x + 16, 145), fill="#0f766e")
    draw.line(roi.point_pairs() + [roi.point_pairs()[0]], fill="#111827", width=2)
    return image


def paste_poster(scene: Image.Image, roi: ROI, poster_path: Path) -> Image.Image:
    base = pil_to_bgr(scene)
    poster = cv2.imdecode(np.fromfile(str(poster_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if poster is None:
        raise RuntimeError(f"cannot load poster template: {poster_path}")

    target = shrink_points(ordered_points(roi), 0.78).astype(np.float32)
    source = np.array(
        [
            [0, 0],
            [poster.shape[1] - 1, 0],
            [poster.shape[1] - 1, poster.shape[0] - 1],
            [0, poster.shape[0] - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, target)
    warped_poster = cv2.warpPerspective(poster, transform, (WIDTH, HEIGHT))
    mask = cv2.warpPerspective(
        np.full((poster.shape[0], poster.shape[1]), 255, dtype=np.uint8),
        transform,
        (WIDTH, HEIGHT),
    )
    base[mask > 0] = warped_poster[mask > 0]
    return bgr_to_pil(base)


def add_occluder(scene: Image.Image, center_x: int, roi: ROI) -> Image.Image:
    image = scene.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    bounds = roi.bounds
    body = (
        center_x - 34,
        bounds["y"] - 12,
        center_x + 34,
        bounds["y"] + bounds["height"] + 18,
    )
    draw.ellipse((center_x - 22, bounds["y"] - 30, center_x + 22, bounds["y"] + 12), fill=(66, 66, 66, 235))
    draw.rounded_rectangle(body, radius=18, fill=(31, 41, 55, 235))
    draw.rectangle(
        (center_x - 44, bounds["y"] + 30, center_x + 44, bounds["y"] + bounds["height"] - 12),
        fill=(15, 23, 42, 235),
    )
    return image


def darken(scene: Image.Image, factor: float) -> Image.Image:
    array = np.array(scene).astype(np.float32) * factor
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def write_video(path: Path, frames: list[Image.Image]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video writer for {path}")

    try:
        for frame in frames:
            writer.write(pil_to_bgr(frame))
    finally:
        writer.release()


def generate_test_data() -> dict[str, object]:
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    store = ConfigStore()

    poster_path = POSTER_TEMPLATE_DIR / "target_pop_template.png"
    create_poster_template(poster_path)

    configs = [
        {
            "store_name": "StoreAlpha",
            "cctv_nickname": "FrontCam",
            "roi": ROI(
                name="POP",
                points=[
                    Point(x=442, y=36),
                    Point(x=578, y=28),
                    Point(x=586, y=138),
                    Point(x=446, y=132),
                ],
            ),
            "bg": "#faf5ff",
            "accent": "#ddd6fe",
        },
        {
            "store_name": "StoreBeta",
            "cctv_nickname": "CounterCam",
            "roi": ROI(
                name="POP",
                points=[
                    Point(x=74, y=56),
                    Point(x=228, y=48),
                    Point(x=240, y=182),
                    Point(x=68, y=174),
                ],
            ),
            "bg": "#eff6ff",
            "accent": "#bfdbfe",
        },
    ]

    label_cases: list[dict[str, object]] = []

    for config in configs:
        roi = config["roi"]
        scene = base_scene(config["bg"], config["accent"], roi)
        reference_path = TEST_DATA_DIR / f"{config['store_name']}_{config['cctv_nickname']}_reference.png"
        scene.save(reference_path)

        saved_config = store.save_config(
            store_name=config["store_name"],
            cctv_nickname=config["cctv_nickname"],
            rois=[roi],
            reference_image_source=reference_path,
        )

        present_frame = paste_poster(scene, roi, poster_path)
        absent_frame = scene.copy()
        bounds = roi.bounds
        occluded_frame = add_occluder(present_frame, bounds["x"] + (bounds["width"] // 2), roi)
        dark_frame = darken(present_frame, 0.18)

        present_frame.save(TEST_DATA_DIR / f"{saved_config.config_id}_present_frame.png")
        absent_frame.save(TEST_DATA_DIR / f"{saved_config.config_id}_absent_frame.png")
        occluded_frame.save(TEST_DATA_DIR / f"{saved_config.config_id}_occluded_frame.png")
        dark_frame.save(TEST_DATA_DIR / f"{saved_config.config_id}_dark_frame.png")

        if saved_config.store_name == "StoreAlpha":
            present_video = [present_frame.copy() for _ in range(12)]
            absent_video = [absent_frame.copy() for _ in range(12)]
            write_video(TEST_DATA_DIR / "storealpha_present.avi", present_video)
            write_video(TEST_DATA_DIR / "storealpha_absent.avi", absent_video)

            label_cases.extend(
                [
                    {
                        "name": "storealpha_present",
                        "config_id": saved_config.config_id,
                        "roi_name": "POP",
                        "media_path": str(TEST_DATA_DIR / "storealpha_present.avi"),
                        "sensor_brightness": 0.82,
                        "enable_sensor_match": True,
                        "expected_decision": "Present",
                        "expected_valid": True,
                    },
                    {
                        "name": "storealpha_absent",
                        "config_id": saved_config.config_id,
                        "roi_name": "POP",
                        "media_path": str(TEST_DATA_DIR / "storealpha_absent.avi"),
                        "sensor_brightness": 0.82,
                        "enable_sensor_match": True,
                        "expected_decision": "Absent",
                        "expected_valid": True,
                    },
                ]
            )
        else:
            occluded_frames = []
            for offset in (24, 42, 60, 78, 96, 114, 132, 150, 150, 150, 150, 150):
                occluded_frames.append(add_occluder(present_frame, bounds["x"] + offset, roi))
            dark_frames = [dark_frame.copy() for _ in range(12)]
            write_video(TEST_DATA_DIR / "storebeta_occluded.avi", occluded_frames)
            write_video(TEST_DATA_DIR / "storebeta_dark.avi", dark_frames)

            label_cases.extend(
                [
                    {
                        "name": "storebeta_occluded",
                        "config_id": saved_config.config_id,
                        "roi_name": "POP",
                        "media_path": str(TEST_DATA_DIR / "storebeta_occluded.avi"),
                        "sensor_brightness": 0.58,
                        "enable_sensor_match": False,
                        "expected_decision": "Unknown",
                        "expected_valid": False,
                    },
                    {
                        "name": "storebeta_dark",
                        "config_id": saved_config.config_id,
                        "roi_name": "POP",
                        "media_path": str(TEST_DATA_DIR / "storebeta_dark.avi"),
                        "sensor_brightness": 0.68,
                        "enable_sensor_match": True,
                        "expected_decision": "Unknown",
                        "expected_valid": False,
                    },
                ]
            )

    labels = {
        "poster_template_path": str(poster_path),
        "cases": label_cases,
    }
    with (TEST_DATA_DIR / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(labels, handle, ensure_ascii=False, indent=2)
    return labels


if __name__ == "__main__":
    output = generate_test_data()
    print(json.dumps(output, ensure_ascii=False, indent=2))
