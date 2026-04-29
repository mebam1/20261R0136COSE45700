from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.config import REFERENCE_IMAGE_DIR, ROI_CONFIG_DIR, STORE_CATALOG_PATH
from app.schemas import CCTVConfig, ROI, safe_filename_part


class ConfigStore:
    def __init__(
        self,
        config_dir: Path = ROI_CONFIG_DIR,
        reference_dir: Path = REFERENCE_IMAGE_DIR,
        store_catalog_path: Path = STORE_CATALOG_PATH,
    ) -> None:
        self.config_dir = config_dir
        self.reference_dir = reference_dir
        self.store_catalog_path = store_catalog_path
        if not self.store_catalog_path.exists():
            self._write_store_catalog([])

    def _load_store_catalog(self) -> list[str]:
        if not self.store_catalog_path.exists():
            return []
        with self.store_catalog_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_stores = payload.get("stores", []) if isinstance(payload, dict) else payload
        return sorted({str(item).strip() for item in raw_stores if str(item).strip()})

    def _write_store_catalog(self, store_names: list[str]) -> None:
        normalized = sorted({name.strip() for name in store_names if name.strip()})
        with self.store_catalog_path.open("w", encoding="utf-8") as handle:
            json.dump({"stores": normalized}, handle, ensure_ascii=False, indent=2)

    def add_store_name(self, store_name: str) -> None:
        normalized = store_name.strip()
        if not normalized:
            raise ValueError("store name is required")
        store_names = self._load_store_catalog()
        if normalized not in store_names:
            store_names.append(normalized)
            self._write_store_catalog(store_names)

    def list_store_names(self) -> list[str]:
        store_names = set(self._load_store_catalog())
        for config in self.list_configs():
            if config.store_name.strip():
                store_names.add(config.store_name.strip())
        return sorted(store_names)

    def list_store_summaries(self) -> list[dict[str, int | str]]:
        cctv_counts = Counter(config.store_name for config in self.list_configs())
        return [
            {"name": store_name, "cctv_count": cctv_counts.get(store_name, 0)}
            for store_name in self.list_store_names()
        ]

    def list_configs(self) -> list[CCTVConfig]:
        configs: list[CCTVConfig] = []
        for path in sorted(self.config_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                configs.append(CCTVConfig.from_dict(json.load(handle)))
        return configs

    def list_configs_by_store(self, store_name: str) -> list[CCTVConfig]:
        configs = [config for config in self.list_configs() if config.store_name == store_name]
        return sorted(configs, key=lambda item: item.cctv_nickname)

    def get_config_path(self, store_name: str, cctv_nickname: str) -> Path:
        return self.config_dir / f"{safe_filename_part(store_name)}_{safe_filename_part(cctv_nickname)}.json"

    def save_config(
        self,
        store_name: str,
        cctv_nickname: str,
        rois: Iterable[ROI],
        reference_image_source: Path | None = None,
        existing_reference_path: str | None = None,
    ) -> CCTVConfig:
        normalized_store_name = store_name.strip()
        normalized_cctv_nickname = cctv_nickname.strip()
        if not normalized_store_name:
            raise ValueError("store name is required")
        if not normalized_cctv_nickname:
            raise ValueError("cctv nickname is required")

        config_path = self.get_config_path(normalized_store_name, normalized_cctv_nickname)
        config_id = config_path.stem
        rois_list = list(rois)
        roi_names = [roi.name.strip() for roi in rois_list]
        if len(roi_names) != len(set(roi_names)):
            raise ValueError("roi names must be unique within a cctv")

        if reference_image_source is None and not existing_reference_path:
            raise ValueError("reference image is required")

        if reference_image_source is not None:
            suffix = reference_image_source.suffix.lower() or ".png"
            saved_reference = self.reference_dir / f"{config_id}{suffix}"
            shutil.copy2(reference_image_source, saved_reference)
            reference_path = saved_reference.relative_to(self.reference_dir.parent).as_posix()
        else:
            reference_path = existing_reference_path or ""

        self.add_store_name(normalized_store_name)

        config = CCTVConfig(
            store_name=normalized_store_name,
            cctv_nickname=normalized_cctv_nickname,
            reference_image_path=reference_path,
            areas=rois_list,
        )
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
        return config

    def load(self, config_id: str) -> CCTVConfig:
        path = self.config_dir / f"{config_id}.json"
        if not path.exists():
            raise FileNotFoundError(config_id)
        with path.open("r", encoding="utf-8") as handle:
            return CCTVConfig.from_dict(json.load(handle))

    def get_roi(self, config_id: str, roi_name: str) -> ROI:
        config = self.load(config_id)
        for area in config.areas:
            if area.name == roi_name:
                return area
        raise KeyError(roi_name)
