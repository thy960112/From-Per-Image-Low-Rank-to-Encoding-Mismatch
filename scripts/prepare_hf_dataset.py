#!/usr/bin/env python3
"""Prepare ``Raw data/`` for the Hugging Face Dataset Hub.

The generated directory contains byte-identical copies of the source artifacts,
two viewer-friendly indexes, one Dataset configuration per source CSV, and a
Dataset Card. The command is deterministic for a fixed source tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET_ID = "Huiyuancs/Encoding_Mismatch_Analysis_Data"
DEFAULT_LICENSE = "apache-2.0"
PAPER_URL = "https://arxiv.org/abs/2511.15572"
PAPER_PAGE_URL = "https://huggingface.co/papers/2511.15572"
GITHUB_URL = (
    "https://github.com/thy960112/"
    "From-Per-Image-Low-Rank-to-Encoding-Mismatch"
)
MODEL_URL = "https://huggingface.co/Huiyuancs/Encoding_Mismatch"
RESERVED_CONFIG_NAMES = {"manifest", "npz_array_catalog"}


@dataclass(frozen=True)
class CsvConfig:
    name: str
    source_path: str
    data_path: str
    rows: int
    columns: int
    normalized_for_viewer: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Raw data"),
        help="Source directory (default: Raw data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hf_dataset_release"),
        help="Generated upload directory (default: hf_dataset_release).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    parser.add_argument(
        "--license",
        default=DEFAULT_LICENSE,
        help=(
            "Dataset Card license identifier "
            f"(default: {DEFAULT_LICENSE}, matching this repository's LICENSE). "
            "Pass an empty string to omit the field."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Dataset Hub repository ID (default: {DEFAULT_DATASET_ID}).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recommended_loader(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return "hf_hub_download + numpy.load(allow_pickle=False)"
    if suffix == ".csv":
        return "datasets.load_dataset or pandas.read_csv"
    return "hf_hub_download"


def yaml_string(value: str) -> str:
    # JSON double-quoted strings are valid YAML strings.
    return json.dumps(value, ensure_ascii=False)


def config_base(relative_path: Path) -> str:
    raw = relative_path.with_suffix("").as_posix().lower()
    name = re.sub(r"[^a-z0-9]+", "_", raw).strip("_") or "csv_table"
    return name[:80].rstrip("_") or "csv_table"


def stable_config_names(paths: list[Path]) -> dict[Path, str]:
    bases = {path: config_base(path) for path in paths}
    counts = Counter(bases.values())
    result: dict[Path, str] = {}
    used = set(RESERVED_CONFIG_NAMES)

    for path in sorted(paths, key=lambda item: item.as_posix()):
        base = bases[path]
        needs_hash = counts[base] > 1 or base in RESERVED_CONFIG_NAMES
        digest = hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()
        name = f"{base[:71].rstrip('_')}_{digest[:8]}" if needs_hash else base
        hash_length = 8
        while name in used:
            hash_length += 2
            name = f"{base[: 80 - hash_length - 1].rstrip('_')}_{digest[:hash_length]}"
        used.add(name)
        result[path] = name
    return result


def clean_error(exc: BaseException) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())


def json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:197] + "..."
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


def array_summary(array: np.ndarray, preview_size: int = 12) -> dict[str, Any]:
    preview = [json_safe_scalar(item) for item in array.reshape(-1)[:preview_size]]
    row: dict[str, Any] = {
        "dtype": str(array.dtype),
        "shape": json.dumps(list(array.shape), separators=(",", ":")),
        "ndim": int(array.ndim),
        "num_elements": int(array.size),
        "min": "",
        "max": "",
        "mean": "",
        "std": "",
        "preview": json.dumps(
            preview, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ),
    }

    is_numeric = np.issubdtype(array.dtype, np.number)
    is_boolean = np.issubdtype(array.dtype, np.bool_)
    if array.size and (is_numeric or is_boolean):
        values = np.abs(array) if np.iscomplexobj(array) else array
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = values[np.isfinite(values)]
        if finite.size:
            row.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )
    return row


def write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def inspect_npz_files(raw_output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_output.rglob("*.npz")):
        source_file = path.relative_to(raw_output.parent).as_posix()
        try:
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    row: dict[str, Any] = {
                        "source_file": source_file,
                        "array_key": key,
                        "dtype": "",
                        "shape": "",
                        "ndim": "",
                        "num_elements": "",
                        "min": "",
                        "max": "",
                        "mean": "",
                        "std": "",
                        "preview": "",
                        "inspection_error": "",
                    }
                    try:
                        row.update(array_summary(np.asarray(archive[key])))
                    except Exception as exc:
                        row["inspection_error"] = clean_error(exc)
                    rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "source_file": source_file,
                    "array_key": "",
                    "dtype": "",
                    "shape": "",
                    "ndim": "",
                    "num_elements": "",
                    "min": "",
                    "max": "",
                    "mean": "",
                    "std": "",
                    "preview": "",
                    "inspection_error": clean_error(exc),
                }
            )
    return rows


def load_csv_with_datasets(path: Path) -> tuple[int, list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "csv",
        data_files={"train": str(path.resolve())},
        split="train",
    )
    return len(dataset), list(dataset.column_names)


def normalize_csv_for_viewer(source: Path, destination: Path) -> tuple[int, int]:
    import pandas as pd
    from pandas.testing import assert_frame_equal

    original = pd.read_csv(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    original.to_csv(
        destination,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )
    normalized = pd.read_csv(destination)

    if original.shape != normalized.shape:
        raise ValueError(
            f"CSV normalization changed shape: {original.shape} -> {normalized.shape}"
        )
    if list(original.columns) != list(normalized.columns):
        raise ValueError("CSV normalization changed column names or order")
    assert_frame_equal(original, normalized, check_dtype=False, check_exact=True)
    loaded_rows, loaded_columns = load_csv_with_datasets(destination)
    if loaded_rows != len(original) or loaded_columns != list(original.columns):
        raise ValueError("Datasets validation changed CSV rows or columns")
    return len(original), len(original.columns)


def prepare_csv_configs(raw_output: Path, data_output: Path) -> list[CsvConfig]:
    source_paths = sorted(
        (path.relative_to(raw_output) for path in raw_output.rglob("*.csv")),
        key=lambda item: item.as_posix(),
    )
    names = stable_config_names(source_paths)
    configs: list[CsvConfig] = []

    for relative in source_paths:
        source = raw_output / relative
        source_path = source.relative_to(raw_output.parent).as_posix()
        name = names[relative]
        try:
            rows, columns = load_csv_with_datasets(source)
            configs.append(
                CsvConfig(name, source_path, source_path, rows, len(columns), False)
            )
        except Exception as direct_error:
            viewer_path = data_output / "viewer_csv" / f"{name}.csv"
            try:
                rows, columns = normalize_csv_for_viewer(source, viewer_path)
            except Exception as normalize_error:
                raise RuntimeError(
                    f"Could not create a lossless viewer representation for {source}: "
                    f"direct read failed ({clean_error(direct_error)}); "
                    f"normalization failed ({clean_error(normalize_error)})"
                ) from normalize_error
            configs.append(
                CsvConfig(
                    name,
                    source_path,
                    viewer_path.relative_to(raw_output.parent).as_posix(),
                    rows,
                    columns,
                    True,
                )
            )
    return configs


def build_readme(
    dataset_id: str,
    license_id: str,
    configs: list[CsvConfig],
    first_npz_path: str,
) -> str:
    yaml_lines = ["---"]
    if license_id:
        yaml_lines.append(f"license: {yaml_string(license_id)}")
    yaml_lines.extend(
        [
            "pretty_name: Encoding Mismatch Analysis Data",
            "task_categories:",
            "- image-classification",
            "tags:",
            "- computer-vision",
            "- vision-transformer",
            "- knowledge-distillation",
            "- representation-analysis",
            "- encoding-mismatch",
            "- spectral-energy-pattern",
            "- pca",
            "- svd",
            "- icml-2026",
            "- arxiv:2511.15572",
            "configs:",
            "- config_name: npz_array_catalog",
            "  data_files:",
            "  - split: train",
            "    path: data/npz_array_catalog.csv",
            "  default: true",
            "- config_name: manifest",
            "  data_files:",
            "  - split: train",
            "    path: data/manifest.csv",
        ]
    )
    for config in configs:
        yaml_lines.extend(
            [
                f"- config_name: {config.name}",
                "  data_files:",
                "  - split: train",
                f"    path: {yaml_string(config.data_path)}",
            ]
        )
    yaml_lines.append("---")

    first_csv = configs[0]
    config_rows = "\n".join(
        "| `{}` | `{}` | `{}` | {} | {} |".format(
            config.name,
            config.source_path,
            config.data_path,
            config.rows,
            "normalized viewer copy" if config.normalized_for_viewer else "original",
        )
        for config in configs
    )
    normalized_configs = [
        config for config in configs if config.normalized_for_viewer
    ]
    if normalized_configs:
        normalization_note = (
            "Some source CSV files required a normalized UTF-8/comma-delimited copy "
            "under `data/viewer_csv/`. The original files remain byte-identical under "
            "`raw/`; the script verifies the normalized representation has identical "
            "rows, columns, column order, and parsed values."
        )
    else:
        normalization_note = (
            "All released CSV files load directly with Hugging Face Datasets, so their "
            "configurations point to the byte-identical files under `raw/`; no "
            "viewer-normalized copies were needed."
        )
    if license_id:
        license_note = (
            f"The repository content is released under `{license_id}`. ImageNet images "
            "are not included; users remain responsible for the terms of ImageNet and "
            "all upstream software or model assets."
        )
    else:
        license_note = (
            "No license identifier was supplied when this release directory was "
            "generated. Consult the source repository before redistributing the "
            "artifacts. ImageNet images are not included."
        )

    body = f"""
# Encoding Mismatch Analysis Data

This repository publishes the prepared numerical analysis artifacts associated
with **From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature
Distillation in Vision Transformers**. It is analysis data, not an image or
model-training dataset, and it does not redistribute ImageNet.

## Links

- Paper: {PAPER_URL}
- Hugging Face paper page: {PAPER_PAGE_URL}
- Code and analysis scripts: {GITHUB_URL}
- Lift and WideLast checkpoints: {MODEL_URL}

## Load the default configuration

The default `npz_array_catalog` configuration has one row per safely inspected
array inside the original NPZ files. It records the source file, array key,
dtype, JSON-encoded shape, dimensionality, element count, finite numeric
summary statistics where applicable, a small JSON preview, and any safe
inspection error.

```python
from datasets import load_dataset

catalog = load_dataset(
    "{dataset_id}",
    split="train",
)
```

## Load the manifest

The manifest records the repository-relative path, file type, byte size,
SHA-256 digest, and recommended loader for every artifact copied from the
GitHub repository's `Raw data/` directory.

```python
from datasets import load_dataset

manifest = load_dataset(
    "{dataset_id}",
    "manifest",
    split="train",
)
```

## Load an original CSV table

Each original CSV has a separate configuration. For example:

```python
from datasets import load_dataset

table = load_dataset(
    "{dataset_id}",
    "{first_csv.name}",
    split="train",
)
```

## Download and read an original NPZ file

Use `hf_hub_download` for the original binary artifacts and keep NumPy's
pickle loading disabled:

```python
from huggingface_hub import hf_hub_download
import numpy as np

path = hf_hub_download(
    repo_id="{dataset_id}",
    repo_type="dataset",
    filename="{first_npz_path}",
)

with np.load(path, allow_pickle=False) as archive:
    print(archive.files)
```

The same download method can be used with any `relative_path` from the
`manifest` configuration.

## Repository structure

```text
README.md
data/
├── manifest.csv
├── npz_array_catalog.csv
└── viewer_csv/              # only created when a source CSV needs it
raw/                         # byte-identical copy of Raw data/
```

`data/npz_array_catalog.csv` is a compact inspection index, not a replacement
for the original arrays. `data/manifest.csv` supplies checksums for verifying
the originals. {normalization_note}

## CSV configurations

| Configuration | Original file | Config data file | Rows | Representation |
|---|---|---|---:|---|
{config_rows}

## Source and intended use

The files are derived from the paper's representation-analysis workflow,
including per-image SVD, dataset-level PCA, and Spectral Energy Pattern
summaries. They are provided for inspecting the reported analyses and for
regenerating tables or figures with the corresponding GitHub scripts. The
artifacts are not a substitute for ImageNet-1K or for rerunning feature
extraction.

## License and third-party data

{license_note}

## Citation

```bibtex
@inproceedings{{tian2026encodingmismatch,
  title     = {{From Per-Image Low-Rank to Encoding Mismatch:
               Rethinking Feature Distillation in Vision Transformers}},
  author    = {{Tian, Huiyuan and Xu, Bonan and Li, Shijian}},
  booktitle = {{Proceedings of the 43rd International Conference on Machine Learning}},
  year      = {{2026}}
}}
```
""".lstrip()
    return "\n".join(yaml_lines) + "\n\n" + body


def validate_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("source and output directories must not contain one another")
    if output == Path(output.anchor):
        raise ValueError("refusing to use a filesystem root as output")


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()

    try:
        validate_paths(source, output)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if output.exists():
        if not args.overwrite:
            print(
                f"ERROR: output already exists: {output}; use --overwrite to replace it",
                file=sys.stderr,
            )
            return 2
        if not output.is_dir():
            print(f"ERROR: output exists and is not a directory: {output}", file=sys.stderr)
            return 2
        shutil.rmtree(output)

    source_files = sorted(
        (path for path in source.rglob("*") if path.is_file()),
        key=lambda item: item.relative_to(source).as_posix(),
    )
    if not source_files:
        print(f"ERROR: no source files found under {source}", file=sys.stderr)
        return 2

    raw_output = output / "raw"
    data_output = output / "data"
    output.mkdir(parents=True)
    raw_output.mkdir()
    data_output.mkdir()

    manifest_rows: list[dict[str, Any]] = []
    for source_path in source_files:
        relative = source_path.relative_to(source)
        destination = raw_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        source_hash = sha256_file(source_path)
        copied_hash = sha256_file(destination)
        if copied_hash != source_hash:
            raise RuntimeError(f"checksum mismatch after copying {relative.as_posix()}")
        manifest_rows.append(
            {
                "relative_path": destination.relative_to(output).as_posix(),
                "file_type": source_path.suffix.lower().lstrip(".") or "unknown",
                "size_bytes": source_path.stat().st_size,
                "sha256": source_hash,
                "recommended_loader": recommended_loader(source_path),
            }
        )

    write_csv(
        data_output / "manifest.csv",
        manifest_rows,
        [
            "relative_path",
            "file_type",
            "size_bytes",
            "sha256",
            "recommended_loader",
        ],
    )

    npz_rows = inspect_npz_files(raw_output)
    write_csv(
        data_output / "npz_array_catalog.csv",
        npz_rows,
        [
            "source_file",
            "array_key",
            "dtype",
            "shape",
            "ndim",
            "num_elements",
            "min",
            "max",
            "mean",
            "std",
            "preview",
            "inspection_error",
        ],
    )

    csv_configs = prepare_csv_configs(raw_output, data_output)
    npz_files = sorted(raw_output.rglob("*.npz"))
    if not npz_files:
        print("ERROR: no NPZ files found", file=sys.stderr)
        return 2
    if not csv_configs:
        print("ERROR: no CSV files found", file=sys.stderr)
        return 2

    readme = build_readme(
        dataset_id=args.dataset_id,
        license_id=args.license.strip(),
        configs=csv_configs,
        first_npz_path=npz_files[0].relative_to(output).as_posix(),
    )
    (output / "README.md").write_text(readme, encoding="utf-8")

    print(f"Prepared upload directory: {output}")
    print(f"Source files copied: {len(source_files)}")
    print(f"NPZ files copied: {len(npz_files)}")
    print(f"CSV files copied: {len(csv_configs)}")
    print(f"Manifest rows: {len(manifest_rows)}")
    print(f"NPZ catalog rows: {len(npz_rows)}")
    print(f"Dataset configurations: {len(csv_configs) + 2}")
    print(
        "Viewer-normalized CSV copies: "
        f"{sum(config.normalized_for_viewer for config in csv_configs)}"
    )
    for config in csv_configs:
        print(f"  {config.name}: {config.data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
