# utils/io.py
from __future__ import annotations

import json
import csv
import shutil
from pathlib import Path
from typing import Any, Iterable, Optional, Union

try:
    import yaml  # optional
except Exception:  # pragma: no cover
    yaml = None

# -------- Path helpers --------
PathLike = Union[str, Path]


def to_path(p: PathLike) -> Path:
    """Convert string-like to pathlib.Path."""
    return p if isinstance(p, Path) else Path(p)


def ensure_dir(p: PathLike) -> Path:
    """Ensure a directory exists and return it."""
    d = to_path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_parent(path: PathLike) -> Path:
    """Ensure parent directory exists for a file path."""
    p = to_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def list_files(folder: PathLike, pattern: str = "*") -> list[Path]:
    """List files matching a glob pattern (non-recursive)."""
    return sorted(to_path(folder).glob(pattern))


# -------- Text / JSON / YAML / CSV --------
def read_text(path: PathLike, encoding: str = "utf-8") -> str:
    """Read a whole text file."""
    return to_path(path).read_text(encoding=encoding)


def write_text(path: PathLike, text: str, encoding: str = "utf-8") -> Path:
    """Write text atomically (via temp file then replace)."""
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(p)
    return p


def read_json(path: PathLike) -> Any:
    """Load JSON file with UTF-8."""
    with to_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any, indent: int = 2) -> Path:
    """Write JSON atomically, pretty-formatted."""
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    tmp.replace(p)
    return p


def read_yaml(path: PathLike) -> Any:
    """Load YAML if PyYAML is available."""
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    with to_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: PathLike, data: Any) -> Path:
    """Write YAML if PyYAML is available."""
    if yaml is None:
        raise RuntimeError("PyYAML not installed. `pip install pyyaml`")
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    tmp.replace(p)
    return p


def write_csv(path: PathLike, rows: Iterable[Iterable[Any]], header: Optional[Iterable[str]] = None) -> Path:
    """Write CSV with optional header."""
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))
    tmp.replace(p)
    return p


# -------- Generic loader --------
def load_any(path: PathLike) -> Any:
    """
    Load data by extension:
      - .json -> JSON
      - .yaml/.yml -> YAML (if available)
      - .txt/.md -> str
    """
    p = to_path(path)
    suf = p.suffix.lower()
    if suf == ".json":
        return read_json(p)
    if suf in (".yaml", ".yml"):
        return read_yaml(p)
    if suf in (".txt", ".md"):
        return read_text(p)
    raise ValueError(f"Unsupported extension: {suf} for {p}")


# -------- Binary helpers / copies --------
def copy_file(src: PathLike, dst: PathLike) -> Path:
    """Copy a file to destination (create parents)."""
    dst_p = ensure_parent(dst)
    shutil.copy2(to_path(src), dst_p)
    return dst_p


def save_bytes(path: PathLike, data: bytes) -> Path:
    """Save raw bytes atomically."""
    p = ensure_parent(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.replace(p)
    return p


# -------- Matplotlib integration --------
def save_fig(fig, path: PathLike, **kwargs) -> Path:
    """
    Save matplotlib figure with sensible defaults.
    Example kwargs: bbox_inches="tight", pad_inches=0.03, dpi=200
    """
    p = ensure_parent(path)
    fig.savefig(p, **({"bbox_inches": "tight", "pad_inches": 0.03, "dpi": 200} | kwargs))
    return p