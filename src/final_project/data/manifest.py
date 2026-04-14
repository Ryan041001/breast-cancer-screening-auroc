from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REQUIRED_VIEWS = frozenset({"CC", "MLO"})
ALLOWED_PATHOLOGIES = frozenset({"N", "B", "M"})


@dataclass(frozen=True, slots=True)
class BreastManifestRecord:
    breast_id: str
    cc_path: Path
    mlo_path: Path
    label: int | None


def build_train_manifest(train_csv: str | Path) -> list[BreastManifestRecord]:
    csv_path = Path(train_csv)
    dataset_root = csv_path.parent.resolve(strict=False)
    grouped_views: dict[str, dict[str, Path]] = {}
    pathologies: dict[str, set[str]] = {}

    for row in _read_rows(csv_path):
        breast_id = _require_value(row, "breast_id")
        view = _require_view(row, breast_id)
        views = grouped_views.setdefault(breast_id, {})
        if view in views:
            raise ValueError(f"Breast '{breast_id}' must have exactly one {view} view")
        views[view] = _resolve_path_within_root(
            dataset_root,
            _require_value(row, "image_path"),
            field_name="image_path",
        )
        pathologies.setdefault(breast_id, set()).add(_require_pathology(row, breast_id))

    return [
        BreastManifestRecord(
            breast_id=breast_id,
            cc_path=views["CC"],
            mlo_path=views["MLO"],
            label=1 if "M" in pathologies[breast_id] else 0,
        )
        for breast_id, views in grouped_views.items()
        if _validate_views(breast_id, views)
    ]


def build_test_manifest(
    submission_csv: str | Path,
    test_images_dir: str | Path,
) -> list[BreastManifestRecord]:
    submission_path = Path(submission_csv)
    images_root = _resolve_relative_path(submission_path.parent, Path(test_images_dir))
    manifest: list[BreastManifestRecord] = []
    seen_breast_ids: set[str] = set()

    for row in _read_rows(submission_path):
        breast_id = _require_breast_id(row)
        if breast_id in seen_breast_ids:
            raise ValueError(
                f"Duplicate breast_id '{breast_id}' found in submission order"
            )
        seen_breast_ids.add(breast_id)

        breast_dir = images_root / breast_id
        _validate_test_images(breast_id, breast_dir)
        cc_path = breast_dir / f"{breast_id}_CC.jpg"
        mlo_path = breast_dir / f"{breast_id}_MLO.jpg"

        manifest.append(
            BreastManifestRecord(
                breast_id=breast_id,
                cc_path=cc_path,
                mlo_path=mlo_path,
                label=None,
            )
        )

    return manifest


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_value(row: dict[str, str], key: str) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required column '{key}'")
    return value


def _require_view(row: dict[str, str], breast_id: str) -> str:
    view = _require_value(row, "cc_mlo")
    if view not in REQUIRED_VIEWS:
        raise ValueError(f"Breast '{breast_id}' has unsupported view '{view}'")
    return view


def _require_pathology(row: dict[str, str], breast_id: str) -> str:
    pathology = _require_value(row, "pathology")
    if pathology not in ALLOWED_PATHOLOGIES:
        raise ValueError(
            f"Breast '{breast_id}' has unsupported pathology '{pathology}'"
        )
    return pathology


def _require_breast_id(row: dict[str, str]) -> str:
    breast_id = _require_value(row, "breast_id")
    if any(separator in breast_id for separator in ("/", "\\")):
        raise ValueError(f"Unsafe breast_id '{breast_id}' in submission order")
    if breast_id in {".", ".."} or any(part == ".." for part in Path(breast_id).parts):
        raise ValueError(f"Unsafe breast_id '{breast_id}' in submission order")
    return breast_id


def _validate_views(breast_id: str, views: dict[str, Path]) -> bool:
    if len(views) != len(REQUIRED_VIEWS) or any(
        view not in REQUIRED_VIEWS for view in views
    ):
        raise ValueError(
            f"Breast '{breast_id}' must have exactly one CC and one MLO view"
        )
    return True


def _validate_test_images(breast_id: str, breast_dir: Path) -> None:
    if not breast_dir.is_dir():
        raise ValueError(
            f"Breast '{breast_id}' must have exactly one CC and one MLO image in test data"
        )

    actual_files = {entry.name for entry in breast_dir.iterdir() if entry.is_file()}
    expected_files = {f"{breast_id}_CC.jpg", f"{breast_id}_MLO.jpg"}
    if actual_files != expected_files:
        raise ValueError(
            f"Breast '{breast_id}' must have exactly one CC and one MLO image in test data"
        )


def _resolve_relative_path(base_dir: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _resolve_path_within_root(
    root_dir: Path,
    raw_path: str | Path,
    *,
    field_name: str,
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        raise ValueError(f"{field_name} must be relative to the dataset root")

    resolved_root = root_dir.resolve(strict=False)
    resolved_path = (resolved_root / candidate).resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"{field_name} must stay within the dataset root")
    return resolved_path
