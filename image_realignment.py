from __future__ import annotations

import gc
import json
import os
import platform as platform_lib
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import affine_transform as ndi_affine_transform
from scipy.ndimage import map_coordinates as ndi_map_coordinates


PIPELINE_VERSION = "image_realignment_v4"
SUPPORTED_OUTPUT_FORMATS = (".zarr", ".ome.zarr", ".ome.tiff")
SUPPORTED_BACKENDS = ("cpu", "gpu")


def detect_platform_tag():
    """Return the host platform using Pixi/conda-style names when possible."""
    system = platform_lib.system()
    machine = platform_lib.machine().lower()
    if system == "Linux":
        return "linux-64" if machine in {"x86_64", "amd64"} else f"linux-{machine}"
    if system == "Windows":
        return "win-64" if machine in {"x86_64", "amd64"} else f"win-{machine}"
    if system == "Darwin":
        return "osx-arm64" if machine in {"arm64", "aarch64"} else "osx-64"
    return f"{system.lower() or 'unknown'}-{machine or 'unknown'}"


def resolve_backend(backend="cpu", platform_tag=None):
    """Resolve public BACKEND='cpu'/'gpu' to the implementation for this OS."""
    requested = str(backend).strip().lower()
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend!r}. Use 'cpu' or 'gpu'.")

    platform_tag = str(platform_tag or detect_platform_tag())
    if requested == "cpu":
        effective = "cpu"
        implementation = "cpu"
    elif platform_tag.startswith("osx"):
        raise NotImplementedError(
            "BACKEND='gpu' on macOS would require a validated Metal/MPS affine transform backend. "
            "This workflow currently supports BACKEND='cpu' on macOS and BACKEND='gpu' via CuPy/CUDA "
            "on Linux or Windows."
        )
    elif platform_tag.startswith(("linux", "win")):
        effective = "cuda"
        implementation = "gpu"
    else:
        raise NotImplementedError(
            f"BACKEND='gpu' is not implemented for detected platform {platform_tag!r}. "
            "Use BACKEND='cpu' on this platform."
        )

    return {
        "requested_backend": requested,
        "implementation_backend": implementation,
        "effective_backend": effective,
        "host_platform": platform_tag,
    }


def now():
    return time.perf_counter()


def _get_zarr():
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    import zarr

    return zarr


def _get_bioimage():
    from bioio import BioImage

    return BioImage


def _get_cupy():
    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BACKEND='gpu' requires CuPy, but CuPy is not installed in this environment. "
            "Either set BACKEND='cpu' or install a CUDA-compatible CuPy build into the same Pixi environment."
        ) from exc

    return cp


def _get_cupy_affine_transform():
    try:
        from cupyx.scipy.ndimage import affine_transform as cupy_affine_transform
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BACKEND='gpu' requires cupyx.scipy.ndimage from a CUDA-compatible CuPy install. "
            "Either set BACKEND='cpu' or install a CuPy build that matches this machine's CUDA stack."
        ) from exc

    return cupy_affine_transform


def _free_cupy_blocks(cp=None):
    if cp is None:
        cp = _get_cupy()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _dtype_name(dtype):
    if dtype is None:
        return None
    try:
        return str(np.dtype(dtype))
    except TypeError:
        return str(dtype).replace("<class 'cupy.", "cupy.").replace("'>", "")


def _normalize_gpu_dtype(gpu_source_dtype, cp):
    if gpu_source_dtype is None:
        return None
    if isinstance(gpu_source_dtype, str):
        name = gpu_source_dtype.replace("cp.", "").replace("cupy.", "")
        return getattr(cp, name)
    return gpu_source_dtype


def _threadpool_context(inner_num_threads=None):
    if inner_num_threads is None:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=int(inner_num_threads))


def _memory_status():
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "ram_percent": float(vm.percent),
            "ram_available_gb": float(vm.available / (1024**3)),
            "ram_total_gb": float(vm.total / (1024**3)),
        }
    except Exception:
        return {
            "ram_percent": None,
            "ram_available_gb": None,
            "ram_total_gb": None,
        }


def _wait_for_ram(start_pct=85.0, resume_pct=75.0, poll_s=10.0, max_wait_s=None):
    status = _memory_status()
    ram_percent = status["ram_percent"]
    if ram_percent is None or ram_percent < float(start_pct):
        status["ram_wait_s"] = 0.0
        return status

    t0 = now()
    while True:
        elapsed = now() - t0
        if max_wait_s is not None and elapsed >= float(max_wait_s):
            break
        time.sleep(float(poll_s))
        status = _memory_status()
        ram_percent = status["ram_percent"]
        if ram_percent is None or ram_percent <= float(resume_pct):
            break

    status["ram_wait_s"] = float(now() - t0)
    return status


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.dtype):
        return str(value)
    return value


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(data), f, indent=2, sort_keys=True)


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _path_signature(path):
    path = Path(path)
    sig = {"path": str(path), "exists": bool(path.exists())}
    if path.exists():
        stat = path.stat()
        sig["size"] = int(stat.st_size)
        sig["mtime_ns"] = int(stat.st_mtime_ns)
    return sig


def _metadata_sidecar_path(output_path):
    return Path(str(output_path) + ".realignment.json")


def _metadata_matches_sidecar(output_path, expected_metadata):
    sidecar = _metadata_sidecar_path(output_path)
    if not Path(output_path).exists() or not sidecar.exists():
        return False
    try:
        return _read_json(sidecar).get("alignment_inputs") == _json_ready(expected_metadata)
    except Exception:
        return False


def _normalize_path_list(paths):
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def normalize_output_formats(output_formats):
    if isinstance(output_formats, (str, Path)):
        values = [str(output_formats)]
    else:
        values = [str(v) for v in output_formats]

    normalized = []
    for value in values:
        fmt = value.strip().lower()
        if not fmt.startswith("."):
            fmt = "." + fmt
        if fmt == ".ome.tif":
            fmt = ".ome.tiff"
        if fmt not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output format {value!r}. "
                f"Use one or more of {SUPPORTED_OUTPUT_FORMATS}."
            )
        if fmt not in normalized:
            normalized.append(fmt)
    if not normalized:
        raise ValueError("At least one output format is required.")
    return tuple(normalized)


def _strip_known_image_suffix(path):
    name = Path(path).name
    lower = name.lower()
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr", ".tiff", ".tif", ".zarr"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "image"


def output_paths_for_image(image_path, output_dir="outputs", output_formats=(".zarr",)):
    output_dir = Path(output_dir)
    image_id = _safe_name(_strip_known_image_suffix(image_path))
    formats = normalize_output_formats(output_formats)
    out = {}
    for fmt in formats:
        if fmt == ".zarr":
            out[fmt] = str(output_dir / f"{image_id}_aligned.zarr")
        elif fmt == ".ome.zarr":
            out[fmt] = str(output_dir / f"{image_id}_aligned.ome.zarr")
        elif fmt == ".ome.tiff":
            out[fmt] = str(output_dir / f"{image_id}_aligned.ome.tiff")
    return image_id, out


def _alignment_id_from_path(alignment_csv):
    alignment_csv = Path(alignment_csv)
    stem = _safe_name(_strip_known_image_suffix(alignment_csv))
    if stem.lower() in {"matrix", "alignment", "keypoints"}:
        return _safe_name(alignment_csv.parent.name)
    return stem


def output_paths_for_image_alignment_pair(
    image_path,
    alignment_csv,
    output_dir="outputs",
    output_formats=(".zarr",),
    force_alignment_suffix=False,
):
    image_id, out = output_paths_for_image(
        image_path,
        output_dir=output_dir,
        output_formats=output_formats,
    )
    if not force_alignment_suffix:
        return image_id, out

    pair_id = _safe_name(f"{image_id}_{_alignment_id_from_path(alignment_csv)}")
    output_dir = Path(output_dir)
    pair_out = {}
    for fmt in normalize_output_formats(output_formats):
        if fmt == ".zarr":
            pair_out[fmt] = str(output_dir / f"{pair_id}_aligned.zarr")
        elif fmt == ".ome.zarr":
            pair_out[fmt] = str(output_dir / f"{pair_id}_aligned.ome.zarr")
        elif fmt == ".ome.tiff":
            pair_out[fmt] = str(output_dir / f"{pair_id}_aligned.ome.tiff")
    return pair_id, pair_out


def _estimate_affine_matrix_from_points(source_xy, target_xy):
    source_xy = np.asarray(source_xy, dtype=np.float64)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    if source_xy.shape != target_xy.shape or source_xy.ndim != 2 or source_xy.shape[1] != 2:
        raise ValueError("source_xy and target_xy must both have shape n x 2.")
    if source_xy.shape[0] < 3:
        raise ValueError("At least 3 keypoints are required to estimate an affine matrix.")

    design = np.c_[source_xy, np.ones(source_xy.shape[0], dtype=np.float64)]
    params, _, _, _ = np.linalg.lstsq(design, target_xy, rcond=None)
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2, :] = params.T.astype(np.float32)
    return matrix


def read_alignment_matrix(alignment_csv):
    alignment_csv = Path(alignment_csv)

    raw = pd.read_csv(alignment_csv, header=None)
    matrix = raw.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if matrix.shape == (3, 3) and np.isfinite(matrix).all():
        return matrix
    if matrix.shape == (2, 3) and np.isfinite(matrix).all():
        return np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32)])

    keypoints = pd.read_csv(alignment_csv)
    required = {"fixedX", "fixedY", "alignmentX", "alignmentY"}
    if required.issubset(keypoints.columns):
        valid = keypoints[list(required)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        keypoints = keypoints.loc[valid].copy()
        source_xy = keypoints[["alignmentX", "alignmentY"]].to_numpy(float)
        target_xy = keypoints[["fixedX", "fixedY"]].to_numpy(float)
        return _estimate_affine_matrix_from_points(source_xy, target_xy)

    raise ValueError(
        f"Expected a 3x3/2x3 matrix CSV or Xenium Explorer keypoint CSV "
        f"with fixedX/fixedY/alignmentX/alignmentY columns: {alignment_csv}"
    )


def normalize_channel_indices(channel_indices, channel_count):
    channel_count = int(channel_count)
    if channel_count < 1:
        raise ValueError(f"Expected at least one image channel, got {channel_count}")
    if channel_indices is None:
        return tuple(range(channel_count))
    if isinstance(channel_indices, (int, np.integer)):
        values = [int(channel_indices)]
    else:
        values = [int(v) for v in channel_indices]
    if not values:
        raise ValueError("CHANNEL_INDICES cannot be empty. Use None to align all channels.")
    bad = [v for v in values if v < 0 or v >= channel_count]
    if bad:
        raise ValueError(f"Channel index/indices out of range for {channel_count} channels: {bad}")
    return tuple(values)


def normalize_z_indices(z_indices, z_count):
    z_count = int(z_count)
    if z_count < 1:
        raise ValueError(f"Expected at least one image Z plane, got {z_count}")
    if z_indices is None:
        return tuple(range(z_count))
    if isinstance(z_indices, (int, np.integer)):
        values = [int(z_indices)]
    else:
        values = [int(v) for v in z_indices]
    if not values:
        raise ValueError("Z_INDICES cannot be empty. Use None to align all Z planes.")
    bad = [v for v in values if v < 0 or v >= z_count]
    if bad:
        raise ValueError(f"Z index/indices out of range for {z_count} planes: {bad}")
    return tuple(values)


def _maybe_ome_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ome_pixel_metadata(ome_xml):
    if not ome_xml:
        return {}

    try:
        root = ET.fromstring(ome_xml)
    except ET.ParseError:
        return {}

    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0].strip("{")
        pixels = root.find(".//ome:Pixels", {"ome": namespace})
        channel_items = root.findall(".//ome:Channel", {"ome": namespace})
    else:
        pixels = root.find(".//Pixels")
        channel_items = root.findall(".//Channel")

    if pixels is None:
        return {}

    channel_names = [
        item.attrib.get("Name")
        for item in channel_items
        if item.attrib.get("Name")
    ]

    return {
        "source_pixel_size_x_um": _maybe_ome_float(pixels.attrib.get("PhysicalSizeX")),
        "source_pixel_size_y_um": _maybe_ome_float(pixels.attrib.get("PhysicalSizeY")),
        "channel_names": channel_names,
    }


def read_image_metadata_tifffile(image_path):
    """Read image dimensions and physical pixel size with tifffile."""
    image_path = Path(image_path)
    with tifffile.TiffFile(str(image_path)) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", None) or [series]
        level0 = levels[0]
        axes = str(getattr(level0, "axes", getattr(series, "axes", "")) or "")
        shape = tuple(int(v) for v in level0.shape)
        c, z, y, x = _infer_tifffile_czyx_shape(shape, axes=axes)
        ome_meta = _parse_ome_pixel_metadata(getattr(tif, "ome_metadata", None))

    px_y = ome_meta.get("source_pixel_size_y_um")
    px_x = ome_meta.get("source_pixel_size_x_um")
    channel_names = list(ome_meta.get("channel_names") or [])

    return {
        "image_path": str(image_path),
        "bioio_dims_order": axes,
        "bioio_shape": list(shape),
        "source_shape_yx": [int(y), int(x)],
        "channel_count": int(c),
        "z_count": int(z),
        "channel_names": channel_names,
        "source_pixel_size_y_um": None if px_y is None else float(px_y),
        "source_pixel_size_x_um": None if px_x is None else float(px_x),
        "pixel_size_y_um": None if px_y is None else float(px_y),
        "pixel_size_x_um": None if px_x is None else float(px_x),
        "metadata_type": "tifffile",
    }


def read_image_metadata(image_path):
    """Read image dimensions and physical pixel size.

    BioIO cannot open some Xenium OME-TIFF files without extra plugins, so
    fall back to tifffile metadata when BioIO rejects the image.
    """
    try:
        BioImage = _get_bioimage()
        img = BioImage(str(image_path))
    except Exception:
        return read_image_metadata_tifffile(image_path)

    dims = img.dims
    dims_order = str(getattr(dims, "order", "") or "")
    shape = tuple(int(v) for v in getattr(img, "shape", ()))
    y = int(getattr(dims, "Y"))
    x = int(getattr(dims, "X"))
    c = int(getattr(dims, "C", 1))
    z = int(getattr(dims, "Z", 1))

    pps = getattr(img, "physical_pixel_sizes", None)
    px_y = getattr(pps, "Y", None) if pps is not None else None
    px_x = getattr(pps, "X", None) if pps is not None else None
    channel_names = list(getattr(img, "channel_names", []) or [])

    try:
        metadata_type = type(getattr(img, "metadata", None)).__name__
    except Exception:
        metadata_type = None

    return {
        "image_path": str(image_path),
        "bioio_dims_order": dims_order,
        "bioio_shape": list(shape),
        "source_shape_yx": [y, x],
        "channel_count": c,
        "z_count": z,
        "channel_names": channel_names,
        "source_pixel_size_y_um": None if px_y is None else float(px_y),
        "source_pixel_size_x_um": None if px_x is None else float(px_x),
        "pixel_size_y_um": None if px_y is None else float(px_y),
        "pixel_size_x_um": None if px_x is None else float(px_x),
        "metadata_type": metadata_type,
    }


def _infer_tifffile_czyx_shape(shape, axes=None):
    shape = tuple(int(v) for v in shape)
    axes = "" if axes is None else str(axes).upper()
    if len(axes) == len(shape) and "Y" in axes and "X" in axes:
        if "C" in axes:
            c = int(shape[axes.index("C")])
        elif "S" in axes:
            c = int(shape[axes.index("S")])
        else:
            c = 1
        z = int(shape[axes.index("Z")]) if "Z" in axes else 1
        return c, z, int(shape[axes.index("Y")]), int(shape[axes.index("X")])
    if len(shape) == 2:
        return 1, 1, int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[0]), 1, int(shape[1]), int(shape[2])
    if len(shape) == 4:
        if shape[1] == 1:
            return int(shape[0]), 1, int(shape[2]), int(shape[3])
        if shape[0] == 1:
            return int(shape[1]), 1, int(shape[2]), int(shape[3])
    if len(shape) == 5:
        return int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4])
    raise ValueError(f"Unexpected TIFF shape for C/Z/Y/X extraction: {shape}")


def _tifffile_czyx_index(shape, axes=None, channel_index=0, z_index=0, y_slice=None, x_slice=None):
    shape = tuple(int(v) for v in shape)
    axes = "" if axes is None else str(axes).upper()
    y_slice = slice(None) if y_slice is None else y_slice
    x_slice = slice(None) if x_slice is None else x_slice
    channel_index = int(channel_index)
    z_index = int(z_index)

    if len(axes) == len(shape) and "Y" in axes and "X" in axes:
        index = []
        remaining = []
        for ax, size in zip(axes, shape):
            if ax == "Y":
                index.append(y_slice)
                remaining.append("Y")
            elif ax == "X":
                index.append(x_slice)
                remaining.append("X")
            elif ax == "C" or (ax == "S" and "C" not in axes):
                if channel_index < 0 or channel_index >= int(size):
                    raise ValueError(f"channel_index={channel_index} outside TIFF channel dimension of size {size}")
                index.append(channel_index)
            elif ax == "Z":
                if z_index < 0 or z_index >= int(size):
                    raise ValueError(f"z_index={z_index} outside TIFF Z dimension of size {size}")
                index.append(z_index)
            else:
                index.append(0)
        transpose = "".join(remaining) == "XY"
        return tuple(index), transpose

    if len(shape) == 2:
        if channel_index != 0:
            raise ValueError("Cannot read channel_index > 0 from a 2D TIFF series with no C axis")
        return (y_slice, x_slice), False
    if len(shape) == 3:
        if channel_index < 0 or channel_index >= int(shape[0]):
            raise ValueError(f"channel_index={channel_index} outside inferred C dimension of size {shape[0]}")
        return (channel_index, y_slice, x_slice), False
    if len(shape) == 4:
        if shape[1] == 1:
            if channel_index < 0 or channel_index >= int(shape[0]):
                raise ValueError(f"channel_index={channel_index} outside inferred C dimension of size {shape[0]}")
            return (channel_index, 0, y_slice, x_slice), False
        if shape[0] == 1:
            if channel_index < 0 or channel_index >= int(shape[1]):
                raise ValueError(f"channel_index={channel_index} outside inferred C dimension of size {shape[1]}")
            return (0, channel_index, y_slice, x_slice), False
    if len(shape) == 5:
        if channel_index < 0 or channel_index >= int(shape[1]):
            raise ValueError(f"channel_index={channel_index} outside inferred C dimension of size {shape[1]}")
        if z_index < 0 or z_index >= int(shape[2]):
            raise ValueError(f"z_index={z_index} outside inferred Z dimension of size {shape[2]}")
        return (0, channel_index, z_index, y_slice, x_slice), False
    raise ValueError(f"Unexpected TIFF shape for C/Z/Y/X extraction: {shape}")


def _as_zarr_array_from_tiff_series(store):
    zarr = _get_zarr()
    opened = zarr.open(store, mode="r")
    if hasattr(opened, "shape") and hasattr(opened, "__getitem__"):
        return opened
    for key in ("0", 0):
        try:
            candidate = opened[key]
            if hasattr(candidate, "shape") and hasattr(candidate, "__getitem__"):
                return candidate
        except Exception:
            pass
    raise ValueError("Could not open tifffile Zarr store as an array")


class TiledTiffImageReader:
    """Tiled channel/YX reader backed by tifffile's Zarr view of the source TIFF."""

    def __init__(self, image_path, dtype=None):
        self.image_path = str(image_path)
        self.dtype = dtype
        self.tif = tifffile.TiffFile(self.image_path)
        self.series = self.tif.series[0]
        self.axes = str(getattr(self.series, "axes", "") or "")
        self.store = self.series.aszarr()
        self.arr = _as_zarr_array_from_tiff_series(self.store)
        self.shape_czyx = _infer_tifffile_czyx_shape(
            self.arr.shape,
            axes=self.axes,
        )
        self.channel_count = int(self.shape_czyx[0])
        self.z_count = int(self.shape_czyx[1])
        self.shape_yx = (int(self.shape_czyx[2]), int(self.shape_czyx[3]))

    def read_window(self, channel_index, y0, y1, x0, x1, step=1, dtype=None, z_index=0):
        y_slice = slice(int(y0), int(y1), int(step))
        x_slice = slice(int(x0), int(x1), int(step))
        index, transpose = _tifffile_czyx_index(
            self.arr.shape,
            axes=self.axes,
            channel_index=channel_index,
            z_index=z_index,
            y_slice=y_slice,
            x_slice=x_slice,
        )
        out = np.asarray(self.arr[index])
        if transpose:
            out = out.T
        out_dtype = dtype if dtype is not None else self.dtype
        if out_dtype is not None:
            out = out.astype(out_dtype, copy=False)
        return out

    def close(self):
        for obj in (getattr(self, "store", None), getattr(self, "tif", None)):
            try:
                obj.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def generate_output_tiles(height, width, tile_y, tile_x):
    for y0 in range(0, int(height), int(tile_y)):
        y1 = min(int(y0) + int(tile_y), int(height))
        for x0 in range(0, int(width), int(tile_x)):
            x1 = min(int(x0) + int(tile_x), int(width))
            yield int(y0), int(y1), int(x0), int(x1)


def compute_canvas_from_image_corners(matrix_orig, source_shape_yx):
    height, width = [int(v) for v in source_shape_yx]
    pts = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(width), 0.0, 1.0],
            [0.0, float(height), 1.0],
            [float(width), float(height), 1.0],
        ],
        dtype=np.float32,
    )
    pts_t = (np.asarray(matrix_orig, dtype=np.float32) @ pts.T).T
    xs = pts_t[:, 0]
    ys = pts_t[:, 1]

    xmin = float(np.floor(xs.min()))
    xmax = float(np.ceil(xs.max()))
    ymin = float(np.floor(ys.min()))
    ymax = float(np.ceil(ys.max()))
    out_x = int(xmax - xmin)
    out_y = int(ymax - ymin)
    if out_x <= 0 or out_y <= 0:
        raise ValueError(f"Invalid output canvas shape: {(out_y, out_x)}")

    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "out_X": out_x,
        "out_Y": out_y,
        "Y": int(source_shape_yx[0]),
        "X": int(source_shape_yx[1]),
    }


def source_window_from_output_tile(A_inv, xmin, ymin, y0, y1, x0, x1, source_shape_yx, pad_px=2):
    a, b, tx = A_inv[0]
    c, d, ty = A_inv[1]
    corners_out = np.array(
        [
            [x0 + xmin, y0 + ymin, 1.0],
            [x1 + xmin, y0 + ymin, 1.0],
            [x0 + xmin, y1 + ymin, 1.0],
            [x1 + xmin, y1 + ymin, 1.0],
        ],
        dtype=np.float32,
    )

    xs = corners_out[:, 0]
    ys = corners_out[:, 1]
    x_src = a * xs + b * ys + tx
    y_src = c * xs + d * ys + ty

    sy0 = int(np.floor(y_src.min())) - int(pad_px)
    sy1 = int(np.ceil(y_src.max())) + int(pad_px)
    sx0 = int(np.floor(x_src.min())) - int(pad_px)
    sx1 = int(np.ceil(x_src.max())) + int(pad_px)

    h, w = [int(v) for v in source_shape_yx]
    sy0 = max(0, sy0)
    sx0 = max(0, sx0)
    sy1 = min(h, sy1)
    sx1 = min(w, sx1)
    if sy1 <= sy0 or sx1 <= sx0:
        return None
    return sy0, sy1, sx0, sx1


def align_tile_cpu_affine(src_tile, src_y0, src_x0, A_inv, xmin, ymin, y0, y1, x0, x1, out_dtype):
    tile_h = int(y1 - y0)
    tile_w = int(x1 - x0)
    a, b, tx = A_inv[0]
    c, d, ty = A_inv[1]
    matrix_yx = np.array([[d, c], [b, a]], dtype=np.float32)
    offset_yx = np.array(
        [
            c * np.float32(x0 + xmin) + d * np.float32(y0 + ymin) + ty - np.float32(src_y0),
            a * np.float32(x0 + xmin) + b * np.float32(y0 + ymin) + tx - np.float32(src_x0),
        ],
        dtype=np.float32,
    )
    aligned = ndi_affine_transform(
        src_tile,
        matrix=matrix_yx,
        offset=offset_yx,
        output_shape=(tile_h, tile_w),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    if np.issubdtype(np.dtype(out_dtype), np.integer):
        info = np.iinfo(out_dtype)
        aligned = np.rint(aligned).clip(info.min, info.max)
    return aligned.astype(out_dtype, copy=False)


def align_tile_cpu_mapcoords(src_tile, src_y0, src_x0, A_inv, xmin, ymin, y0, y1, x0, x1, out_dtype):
    tile_h = int(y1 - y0)
    tile_w = int(x1 - x0)
    a, b, tx = A_inv[0]
    c, d, ty = A_inv[1]

    yy, xx = np.mgrid[0:tile_h, 0:tile_w].astype(np.float32)
    x_out = xx + np.float32(x0 + xmin)
    y_out = yy + np.float32(y0 + ymin)
    x_src = a * x_out + b * y_out + tx - np.float32(src_x0)
    y_src = c * x_out + d * y_out + ty - np.float32(src_y0)

    aligned = ndi_map_coordinates(
        src_tile,
        [y_src, x_src],
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    if np.issubdtype(np.dtype(out_dtype), np.integer):
        info = np.iinfo(out_dtype)
        aligned = np.rint(aligned).clip(info.min, info.max)
    return aligned.astype(out_dtype, copy=False)


def align_tile_cpu(
    src_tile,
    src_y0,
    src_x0,
    A_inv,
    xmin,
    ymin,
    y0,
    y1,
    x0,
    x1,
    out_dtype,
    align_resampler="affine",
):
    if align_resampler == "affine":
        return align_tile_cpu_affine(src_tile, src_y0, src_x0, A_inv, xmin, ymin, y0, y1, x0, x1, out_dtype)
    if align_resampler == "mapcoords":
        return align_tile_cpu_mapcoords(src_tile, src_y0, src_x0, A_inv, xmin, ymin, y0, y1, x0, x1, out_dtype)
    raise ValueError(f"Unsupported CPU align_resampler: {align_resampler}")


def align_tile_gpu(src_tile_gpu, src_y0, src_x0, A_inv, xmin, ymin, y0, y1, x0, x1, out_dtype):
    cp = _get_cupy()
    cupy_affine_transform = _get_cupy_affine_transform()
    tile_h = int(y1 - y0)
    tile_w = int(x1 - x0)
    a, b, tx = A_inv[0]
    c, d, ty = A_inv[1]

    matrix_yx = cp.asarray([[d, c], [b, a]], dtype=cp.float32)
    offset_yx = cp.asarray(
        [
            c * np.float32(x0 + xmin) + d * np.float32(y0 + ymin) + ty - np.float32(src_y0),
            a * np.float32(x0 + xmin) + b * np.float32(y0 + ymin) + tx - np.float32(src_x0),
        ],
        dtype=cp.float32,
    )
    aligned_gpu = cupy_affine_transform(
        src_tile_gpu,
        matrix=matrix_yx,
        offset=offset_yx,
        output_shape=(tile_h, tile_w),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    if np.issubdtype(np.dtype(out_dtype), np.integer):
        info = np.iinfo(out_dtype)
        aligned_gpu = cp.rint(aligned_gpu).clip(info.min, info.max)
    return cp.asnumpy(aligned_gpu).astype(out_dtype, copy=False)


def _create_plain_zarr_array(out_path, shape, dtype=np.uint16, chunks=(1, 2048, 2048), overwrite=False):
    zarr = _get_zarr()
    out_path = Path(out_path)
    if overwrite and out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(out_path), mode="w")
    arr = root.create_array(
        "0",
        shape=tuple(int(v) for v in shape),
        chunks=tuple(int(v) for v in chunks),
        dtype=np.dtype(dtype),
    )
    return root, arr


def _open_plain_zarr_array(zarr_path, mode="r"):
    zarr = _get_zarr()
    root = zarr.open_group(str(zarr_path), mode=mode)
    return root, root["0"]


def _write_zarr_metadata(zarr_path, metadata):
    root, _ = _open_plain_zarr_array(zarr_path, mode="a")
    root.attrs["image_realignment_metadata"] = _json_ready(metadata)
    for key, value in _coordinate_metadata_attrs(metadata).items():
        root.attrs[key] = _json_ready(value)


def _read_zarr_metadata(zarr_path):
    try:
        root, _ = _open_plain_zarr_array(zarr_path, mode="r")
        return root.attrs.get("image_realignment_metadata", None)
    except Exception:
        return None


def _read_zarr_root_attrs(zarr_path):
    try:
        zarr = _get_zarr()
        root = zarr.open_group(str(zarr_path), mode="r")
        return dict(root.attrs)
    except Exception:
        return {}


def _finite_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _origin_px_xy(metadata):
    if not isinstance(metadata, dict):
        return None
    for key in ("origin_global_xy_px", "canvas_xy_origin_px"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x = _finite_float_or_none(value[0])
            y = _finite_float_or_none(value[1])
            if x is not None and y is not None:
                return [x, y]
    canvas = metadata.get("canvas") or {}
    if isinstance(canvas, dict):
        x = _finite_float_or_none(canvas.get("xmin"))
        y = _finite_float_or_none(canvas.get("ymin"))
        if x is not None and y is not None:
            return [x, y]
    return None


def _origin_um_xy(metadata):
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("origin_um_xy")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _finite_float_or_none(value[0])
        y = _finite_float_or_none(value[1])
        if x is not None and y is not None:
            return [x, y]

    origin_px = _origin_px_xy(metadata)
    px_x = _finite_float_or_none(metadata.get("pixel_size_x_um"))
    px_y = _finite_float_or_none(metadata.get("pixel_size_y_um"))
    if origin_px is None or px_x is None or px_y is None:
        return None
    return [float(origin_px[0] * px_x), float(origin_px[1] * px_y)]


def _coordinate_transformations_for_axes(metadata, axes_names, scale_override=None):
    origin_px = _origin_px_xy(metadata) or [0.0, 0.0]
    origin_um = _origin_um_xy(metadata)
    px_x = _finite_float_or_none(metadata.get("pixel_size_x_um"))
    px_y = _finite_float_or_none(metadata.get("pixel_size_y_um"))
    use_um = origin_um is not None and px_x is not None and px_y is not None

    if scale_override is not None and len(scale_override) == len(axes_names):
        scale = [float(v) for v in scale_override]
    else:
        scale = []
        for axis in axes_names:
            axis = str(axis).lower()
            if axis == "x":
                scale.append(float(px_x) if px_x is not None else 1.0)
            elif axis == "y":
                scale.append(float(px_y) if px_y is not None else 1.0)
            else:
                scale.append(1.0)

    translation = []
    for axis in axes_names:
        axis = str(axis).lower()
        if axis == "x":
            translation.append(float(origin_um[0]) if use_um else float(origin_px[0]))
        elif axis == "y":
            translation.append(float(origin_um[1]) if use_um else float(origin_px[1]))
        else:
            translation.append(0.0)

    return [
        {"type": "scale", "scale": scale},
        {"type": "translation", "translation": translation},
    ]


def _coordinate_metadata_attrs(metadata):
    axes_names = list(metadata.get("axes") or ["c", "y", "x"])
    attrs = {
        "coordinate_frame": metadata.get("coordinate_frame"),
        "origin_global_xy_px": _origin_px_xy(metadata),
        "origin_um_xy": _origin_um_xy(metadata),
        "pixel_size_x_um": metadata.get("pixel_size_x_um"),
        "pixel_size_y_um": metadata.get("pixel_size_y_um"),
        "pixel_size_um": metadata.get("pixel_size_um"),
        "canvas": metadata.get("canvas"),
        "axes": axes_names,
        "coordinateTransformations": _coordinate_transformations_for_axes(metadata, axes_names),
    }
    return {k: v for k, v in attrs.items() if v is not None}


def _default_channel_color(index):
    palette = (
        "FFFFFF",
        "00FFFF",
        "FF00FF",
        "FFFF00",
        "00FF00",
        "FF0000",
        "0000FF",
        "FF8000",
    )
    return palette[int(index) % len(palette)]


def _patch_ome_zarr_channel_metadata(root, metadata):
    channel_names = list(metadata.get("selected_channel_names") or [])
    if not channel_names:
        return False

    omero = root.attrs.get("omero", {})
    if not isinstance(omero, dict):
        omero = {}

    existing_channels = omero.get("channels", [])
    if not isinstance(existing_channels, list):
        existing_channels = []

    channels = []
    for index, name in enumerate(channel_names):
        channel = {}
        if index < len(existing_channels) and isinstance(existing_channels[index], dict):
            channel.update(existing_channels[index])
        channel["label"] = str(name)
        channel.setdefault("active", True)
        channel.setdefault("color", _default_channel_color(index))
        channels.append(channel)

    omero["channels"] = channels
    root.attrs["omero"] = _json_ready(omero)
    return True


def _ome_zarr_axes_metadata(axes_names, has_physical_units):
    axes = []
    for axis in axes_names:
        axis = str(axis).lower()
        if axis == "c":
            axes.append({"name": "c", "type": "channel"})
        elif axis in {"z", "y", "x"}:
            item = {"name": axis, "type": "space"}
            if has_physical_units and axis in {"y", "x"}:
                item["unit"] = "micrometer"
            axes.append(item)
        else:
            axes.append({"name": axis})
    return axes


def _write_output_sidecar(output_path, metadata):
    alignment_inputs = metadata.get("alignment_inputs", metadata) if isinstance(metadata, dict) else metadata
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": pd.Timestamp.utcnow().isoformat().replace("+00:00", "Z"),
        "output_path": str(output_path),
        "alignment_inputs": _json_ready(alignment_inputs),
    }
    if isinstance(metadata, dict) and "alignment_inputs" in metadata:
        payload["image_realignment_metadata"] = _json_ready(metadata)
    _write_json(
        _metadata_sidecar_path(output_path),
        payload,
    )


def build_alignment_input_metadata(row, config):
    matrix = np.asarray(row["alignment_matrix"], dtype=np.float32)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "image_id": str(row["image_id"]),
        "image_path": _path_signature(row["image_path"]),
        "alignment_csv": _path_signature(row["alignment_csv"]),
        "alignment_matrix": matrix.round(7).tolist(),
        "source_shape_yx": [int(v) for v in row["source_shape_yx"]],
        "channel_count": int(row["channel_count"]),
        "z_count": int(row.get("z_count", 1)),
        "channel_indices": [int(v) for v in row["channel_indices"]],
        "z_indices": [int(v) for v in row.get("z_indices", (0,))],
        "channel_names": list(row.get("channel_names") or []),
        "selected_channel_names": list(row.get("selected_channel_names") or []),
        "source_pixel_size_x_um": row.get("source_pixel_size_x_um"),
        "source_pixel_size_y_um": row.get("source_pixel_size_y_um"),
        "pixel_size_x_um": row.get("pixel_size_x_um"),
        "pixel_size_y_um": row.get("pixel_size_y_um"),
        "output_pixel_size_source": row.get("output_pixel_size_source"),
        "output_formats": list(normalize_output_formats(config["output_formats"])),
        "backend": str(config["backend"]),
        "requested_backend": str(config.get("requested_backend", config["backend"])),
        "implementation_backend": str(config.get("implementation_backend", config["backend"])),
        "effective_backend": str(config.get("effective_backend", config["backend"])),
        "host_platform": str(config.get("host_platform", detect_platform_tag())),
        "align_dtype": _dtype_name(config["align_dtype"]),
        "cpu_read_dtype": _dtype_name(config["cpu_read_dtype"]),
        "gpu_source_dtype": (
            _dtype_name(config["gpu_source_dtype"])
            if config.get("implementation_backend", config["backend"]) == "gpu"
            else None
        ),
        "align_tile_y": int(config["align_tile_y"]),
        "align_tile_x": int(config["align_tile_x"]),
        "write_chunks": [int(config["write_chunks"][0]), int(config["write_chunks"][1])],
        "align_resampler": str(config["align_resampler"]),
        "source_window_pad_px": int(config["source_window_pad_px"]),
        "ome_zarr_format": int(config["ome_zarr_format"]),
        "ome_tiff_compression": config["ome_tiff_compression"],
        "ome_zarr_compression": config["ome_zarr_compression"],
        "ome_tile_size": int(config["ome_tile_size"]),
        "pyramid_scale": int(config["pyramid_scale"]),
    }


def build_realignment_plan(
    image_paths,
    alignment_csv_paths,
    output_dir="outputs",
    output_formats=(".zarr",),
    overwrite=False,
    validate_metadata=True,
    backend="cpu",
    align_dtype=np.uint16,
    cpu_read_dtype=np.uint16,
    gpu_source_dtype="float32",
    align_tile_y=2048,
    align_tile_x=2048,
    write_chunks=(2048, 2048),
    align_resampler="affine",
    channel_indices=None,
    z_indices=None,
    source_window_pad_px=2,
    ome_zarr_format=3,
    ome_tiff_compression="zlib",
    ome_zarr_compression="zlib",
    ome_tile_size=1024,
    pyramid_scale=2,
    target_pixel_size_x_um=None,
    target_pixel_size_y_um=None,
):
    images = _normalize_path_list(image_paths)
    alignments = _normalize_path_list(alignment_csv_paths)
    repeated_single_image = len(images) == 1 and len(alignments) > 1
    if repeated_single_image:
        images = images * len(alignments)

    if len(images) != len(alignments):
        raise ValueError(
            f"Image/alignment count mismatch: {len(images)} image paths and "
            f"{len(alignments)} alignment CSV paths. Pairing is sequential. "
            "If one OME-TIFF contains multiple slides, pass that one image path "
            "with multiple alignment CSVs; it will be reused for each alignment."
        )

    formats = normalize_output_formats(output_formats)
    backend_info = resolve_backend(backend)
    config = {
        "output_formats": formats,
        "backend": backend_info["requested_backend"],
        "requested_backend": backend_info["requested_backend"],
        "implementation_backend": backend_info["implementation_backend"],
        "effective_backend": backend_info["effective_backend"],
        "host_platform": backend_info["host_platform"],
        "align_dtype": np.dtype(align_dtype),
        "cpu_read_dtype": np.dtype(cpu_read_dtype),
        "gpu_source_dtype": gpu_source_dtype,
        "align_tile_y": int(align_tile_y),
        "align_tile_x": int(align_tile_x),
        "write_chunks": tuple(int(v) for v in write_chunks),
        "align_resampler": str(align_resampler),
        "source_window_pad_px": int(source_window_pad_px),
        "ome_zarr_format": int(ome_zarr_format),
        "ome_tiff_compression": ome_tiff_compression,
        "ome_zarr_compression": ome_zarr_compression,
        "ome_tile_size": int(ome_tile_size),
        "pyramid_scale": int(pyramid_scale),
        "target_pixel_size_x_um": None if target_pixel_size_x_um is None else float(target_pixel_size_x_um),
        "target_pixel_size_y_um": None if target_pixel_size_y_um is None else float(target_pixel_size_y_um),
    }

    rows = []
    lines = []
    image_counts = pd.Series([str(path) for path in images]).value_counts()
    duplicate_image_paths = set(image_counts[image_counts > 1].index)

    for index, (image_path, alignment_csv) in enumerate(zip(images, alignments), start=1):
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        if not alignment_csv.exists():
            raise FileNotFoundError(f"Alignment CSV path does not exist: {alignment_csv}")

        image_id, out_paths = output_paths_for_image_alignment_pair(
            image_path,
            alignment_csv,
            output_dir=output_dir,
            output_formats=formats,
            force_alignment_suffix=str(image_path) in duplicate_image_paths,
        )
        image_meta = read_image_metadata(image_path)
        source_px_x = image_meta["source_pixel_size_x_um"]
        source_px_y = image_meta["source_pixel_size_y_um"]
        output_px_x = config["target_pixel_size_x_um"]
        output_px_y = config["target_pixel_size_y_um"]
        if output_px_x is None and output_px_y is not None:
            output_px_x = output_px_y
        if output_px_y is None and output_px_x is not None:
            output_px_y = output_px_x
        if output_px_x is None:
            output_px_x = source_px_x
        if output_px_y is None:
            output_px_y = source_px_y
        output_pixel_size_source = (
            "target_pixel_size_config"
            if config["target_pixel_size_x_um"] is not None or config["target_pixel_size_y_um"] is not None
            else "source_image_bioio"
        )
        selected_channels = normalize_channel_indices(channel_indices, image_meta["channel_count"])
        selected_z = normalize_z_indices(z_indices, image_meta["z_count"])
        channel_names = list(image_meta.get("channel_names") or [])
        selected_channel_names = [
            channel_names[channel] if channel < len(channel_names) else f"Channel {channel}"
            for channel in selected_channels
        ]
        matrix = read_alignment_matrix(alignment_csv)

        row = {
            "pair_index": int(index),
            "image_id": image_id,
            "image_path": str(image_path),
            "alignment_csv": str(alignment_csv),
            "alignment_matrix": matrix,
            "backend": config["requested_backend"],
            "implementation_backend": config["implementation_backend"],
            "effective_backend": config["effective_backend"],
            "host_platform": config["host_platform"],
            "source_shape_yx": image_meta["source_shape_yx"],
            "source_pixel_size_x_um": source_px_x,
            "source_pixel_size_y_um": source_px_y,
            "pixel_size_x_um": output_px_x,
            "pixel_size_y_um": output_px_y,
            "output_pixel_size_source": output_pixel_size_source,
            "bioio_dims_order": image_meta["bioio_dims_order"],
            "bioio_shape": image_meta["bioio_shape"],
            "channel_count": image_meta["channel_count"],
            "z_count": image_meta["z_count"],
            "channel_indices": selected_channels,
            "z_indices": selected_z,
            "channel_names": channel_names,
            "selected_channel_names": selected_channel_names,
            "output_formats": formats,
            "output_paths": out_paths,
            "out_zarr": out_paths.get(".zarr"),
            "out_ome_zarr": out_paths.get(".ome.zarr"),
            "out_ome_tiff": out_paths.get(".ome.tiff"),
        }
        expected = build_alignment_input_metadata(row, config)
        row["alignment_expected_inputs"] = expected

        if overwrite:
            row["needs_run"] = True
            row["plan_reason"] = "OVERWRITE=True"
        else:
            missing = [fmt for fmt, path in out_paths.items() if not Path(path).exists()]
            metadata_mismatch = []
            if validate_metadata:
                metadata_mismatch = [
                    fmt for fmt, path in out_paths.items() if not _metadata_matches_sidecar(path, expected)
                ]
            if missing:
                row["needs_run"] = True
                row["plan_reason"] = "missing outputs: " + ", ".join(missing)
            elif metadata_mismatch:
                row["needs_run"] = True
                row["plan_reason"] = "metadata mismatch: " + ", ".join(metadata_mismatch)
            else:
                row["needs_run"] = False
                row["plan_reason"] = "outputs already exist"

        rows.append(row)
        lines.append(f"{image_id}: {row['plan_reason']}")

    return pd.DataFrame(rows), lines


def _pixel_size_for_metadata(row_dict):
    px_x = row_dict.get("pixel_size_x_um")
    px_y = row_dict.get("pixel_size_y_um")
    if px_x is None and px_y is None:
        return None, None, None
    if px_x is None:
        px_x = px_y
    if px_y is None:
        px_y = px_x
    px = float(np.sqrt(float(px_x) * float(px_y)))
    return float(px_x), float(px_y), px


def align_image_to_plain_zarr(
    row_dict,
    out_zarr_path,
    backend="cpu",
    overwrite=True,
    align_dtype=np.uint16,
    cpu_read_dtype=np.uint16,
    gpu_source_dtype="float32",
    align_tile_y=2048,
    align_tile_x=2048,
    write_chunks=(2048, 2048),
    align_resampler="affine",
    source_window_pad_px=2,
):
    backend_info = resolve_backend(backend)
    requested_backend = backend_info["requested_backend"]
    implementation_backend = backend_info["implementation_backend"]
    effective_backend = backend_info["effective_backend"]
    host_platform = backend_info["host_platform"]
    backend = implementation_backend

    if backend == "gpu" and align_resampler != "affine":
        raise ValueError("CuPy/CUDA GPU backend currently supports align_resampler='affine' only.")

    cp = None
    image_path = Path(row_dict["image_path"])
    source_shape_yx = tuple(int(v) for v in row_dict["source_shape_yx"])
    channel_indices = tuple(int(v) for v in row_dict.get("channel_indices", (0,)))
    z_indices = tuple(int(v) for v in row_dict.get("z_indices", (0,)))
    selected_channel_names = list(row_dict.get("selected_channel_names") or [f"Channel {c}" for c in channel_indices])
    matrix_orig = np.asarray(row_dict["alignment_matrix"], dtype=np.float32)
    align_dtype = np.dtype(align_dtype)
    cpu_read_dtype = np.dtype(cpu_read_dtype)

    timing = {
        "t_inverse_s": 0.0,
        "t_zarr_init_s": 0.0,
        "t_read_s": 0.0,
        "t_compute_s": 0.0,
        "t_write_s": 0.0,
    }
    n_tiles = 0
    n_tiles_skipped = 0

    if backend == "gpu":
        cp = _get_cupy()
        gpu_source_dtype = _normalize_gpu_dtype(gpu_source_dtype, cp)

    canvas = compute_canvas_from_image_corners(
        matrix_orig=matrix_orig,
        source_shape_yx=source_shape_yx,
    )
    xmin = np.float32(canvas["xmin"])
    ymin = np.float32(canvas["ymin"])
    out_shape_yx = (int(canvas["out_Y"]), int(canvas["out_X"]))
    has_z_axis = len(z_indices) > 1
    out_axes = ["c", "z", "y", "x"] if has_z_axis else ["c", "y", "x"]
    out_shape = (
        (len(channel_indices), len(z_indices), int(canvas["out_Y"]), int(canvas["out_X"]))
        if has_z_axis
        else (len(channel_indices), int(canvas["out_Y"]), int(canvas["out_X"]))
    )
    out_chunks = (
        (1, 1, int(write_chunks[0]), int(write_chunks[1]))
        if has_z_axis
        else (1, int(write_chunks[0]), int(write_chunks[1]))
    )

    t0 = now()
    A_inv = np.linalg.inv(matrix_orig).astype(np.float32)
    timing["t_inverse_s"] = now() - t0

    t0 = now()
    root, arr = _create_plain_zarr_array(
        out_path=out_zarr_path,
        shape=out_shape,
        dtype=align_dtype,
        chunks=out_chunks,
        overwrite=overwrite,
    )
    timing["t_zarr_init_s"] = now() - t0

    with TiledTiffImageReader(image_path, dtype=cpu_read_dtype) as reader:
        for out_c, source_c in enumerate(channel_indices):
            for out_z, source_z in enumerate(z_indices):
                for y0, y1, x0, x1 in generate_output_tiles(
                    out_shape_yx[0],
                    out_shape_yx[1],
                    align_tile_y,
                    align_tile_x,
                ):
                    src_window = source_window_from_output_tile(
                        A_inv=A_inv,
                        xmin=xmin,
                        ymin=ymin,
                        y0=y0,
                        y1=y1,
                        x0=x0,
                        x1=x1,
                        source_shape_yx=source_shape_yx,
                        pad_px=source_window_pad_px,
                    )
                    if src_window is None:
                        n_tiles_skipped += 1
                        continue
                    sy0, sy1, sx0, sx1 = src_window

                    t0 = now()
                    src_tile_cpu = reader.read_window(
                        source_c,
                        sy0,
                        sy1,
                        sx0,
                        sx1,
                        dtype=cpu_read_dtype,
                        z_index=source_z,
                    )
                    timing["t_read_s"] += now() - t0
                    if src_tile_cpu.size == 0:
                        n_tiles_skipped += 1
                        continue

                    t0 = now()
                    if backend == "gpu":
                        src_tile_gpu = cp.asarray(src_tile_cpu)
                        if gpu_source_dtype is not None and src_tile_gpu.dtype != gpu_source_dtype:
                            src_tile_gpu = src_tile_gpu.astype(gpu_source_dtype, copy=False)
                        aligned_tile = align_tile_gpu(
                            src_tile_gpu=src_tile_gpu,
                            src_y0=sy0,
                            src_x0=sx0,
                            A_inv=A_inv,
                            xmin=xmin,
                            ymin=ymin,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                            out_dtype=align_dtype,
                        )
                        src_tile_gpu = None
                        _free_cupy_blocks(cp)
                    else:
                        aligned_tile = align_tile_cpu(
                            src_tile=src_tile_cpu.astype(np.float32, copy=False),
                            src_y0=sy0,
                            src_x0=sx0,
                            A_inv=A_inv,
                            xmin=xmin,
                            ymin=ymin,
                            y0=y0,
                            y1=y1,
                            x0=x0,
                            x1=x1,
                            out_dtype=align_dtype,
                            align_resampler=align_resampler,
                        )
                    timing["t_compute_s"] += now() - t0

                    t0 = now()
                    if has_z_axis:
                        arr[out_c, out_z, y0:y1, x0:x1] = aligned_tile
                    else:
                        arr[out_c, y0:y1, x0:x1] = aligned_tile
                    timing["t_write_s"] += now() - t0
                    n_tiles += 1

                    del src_tile_cpu, aligned_tile
                    if n_tiles % 16 == 0:
                        gc.collect()

    px_x, px_y, px = _pixel_size_for_metadata(row_dict)
    origin_global_xy_px = [float(canvas["xmin"]), float(canvas["ymin"])]
    origin_um_xy = None
    if px_x is not None and px_y is not None:
        origin_um_xy = [
            float(origin_global_xy_px[0] * float(px_x)),
            float(origin_global_xy_px[1] * float(px_y)),
        ]

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "image_id": row_dict["image_id"],
        "image_path": row_dict["image_path"],
        "alignment_csv": row_dict["alignment_csv"],
        "source_shape_yx": [int(v) for v in source_shape_yx],
        "aligned_shape_yx": [int(out_shape_yx[0]), int(out_shape_yx[1])],
        "aligned_shape": [int(v) for v in out_shape],
        "aligned_shape_cyx": [int(v) for v in out_shape],
        "channel_count": int(row_dict.get("channel_count", len(channel_indices))),
        "z_count": int(row_dict.get("z_count", len(z_indices))),
        "channel_indices": [int(v) for v in channel_indices],
        "z_indices": [int(v) for v in z_indices],
        "channel_names": list(row_dict.get("channel_names") or []),
        "selected_channel_names": selected_channel_names,
        "canvas": canvas,
        "canvas_xy_origin_px": origin_global_xy_px,
        "origin_global_xy_px": origin_global_xy_px,
        "origin_um_xy": origin_um_xy,
        "source_pixel_size_x_um": row_dict.get("source_pixel_size_x_um"),
        "source_pixel_size_y_um": row_dict.get("source_pixel_size_y_um"),
        "pixel_size_x_um": px_x,
        "pixel_size_y_um": px_y,
        "pixel_size_um": px,
        "output_pixel_size_source": row_dict.get("output_pixel_size_source", "source_image_bioio"),
        "dtype": str(align_dtype),
        "axes": out_axes,
        "coordinate_frame": "xenium_aligned_px",
        "coordinate_frame_note": (
            "Alignment CSV is treated as a native-image-pixel to Xenium-pixel transform. "
            "The aligned array is cropped to the transformed image canvas; origin_global_xy_px "
            "records where local output pixel (0, 0) sits in the Xenium pixel frame."
        ),
        "backend": requested_backend,
        "implementation_backend": implementation_backend,
        "effective_backend": effective_backend,
        "host_platform": host_platform,
        "align_resampler": align_resampler,
        "alignment_matrix": matrix_orig.tolist(),
        "alignment_inputs": row_dict.get("alignment_expected_inputs"),
        "tiles": {
            "align_tile_yx": [int(align_tile_y), int(align_tile_x)],
            "write_chunks": [int(write_chunks[0]), int(write_chunks[1])],
            "n_tiles": int(n_tiles),
            "n_tiles_skipped": int(n_tiles_skipped),
        },
    }
    _write_zarr_metadata(out_zarr_path, metadata)

    del root, arr
    gc.collect()
    if backend == "gpu" and cp is not None:
        _free_cupy_blocks(cp)

    return {
        "plain_zarr_path": str(out_zarr_path),
        "metadata": metadata,
        "aligned_shape_yx": out_shape_yx,
        "aligned_shape_cyx": out_shape,
        "aligned_shape": out_shape,
        "aligned_axes": out_axes,
        "n_tiles": int(n_tiles),
        "n_tiles_skipped": int(n_tiles_skipped),
        **timing,
    }


def _tile_iterator_from_zarr_array(arr, tile_shape):
    tile_y, tile_x = [int(v) for v in tile_shape]
    if arr.ndim == 2:
        height, width = [int(v) for v in arr.shape]
        for y0, y1, x0, x1 in generate_output_tiles(height, width, tile_y, tile_x):
            yield np.asarray(arr[y0:y1, x0:x1])
        return
    if arr.ndim == 3:
        channels, height, width = [int(v) for v in arr.shape]
        for channel in range(channels):
            for y0, y1, x0, x1 in generate_output_tiles(height, width, tile_y, tile_x):
                yield np.asarray(arr[channel, y0:y1, x0:x1])
        return
    if arr.ndim == 4:
        channels, z_planes, height, width = [int(v) for v in arr.shape]
        for channel in range(channels):
            for z_index in range(z_planes):
                for y0, y1, x0, x1 in generate_output_tiles(height, width, tile_y, tile_x):
                    yield np.asarray(arr[channel, z_index, y0:y1, x0:x1])
        return
    raise ValueError(f"Expected 2D YX, 3D CYX, or 4D CZYX Zarr array, got shape {arr.shape}")


def _normalize_tiff_compression(compression):
    if compression is None:
        return "zlib"
    value = str(compression).strip().lower()
    if value in {"", "none", "uncompressed"}:
        return None
    if value in {"zlib", "deflate", "adobe_deflate"}:
        return "zlib"
    if value in {"jpeg2000", "jpeg_2000", "jp2k", "j2k"}:
        return "jpeg2000"
    return compression


def _zarr_compressor_kwargs(compression, zarr_format=None):
    if compression is None:
        return {}
    value = str(compression).strip().lower()
    if value in {"", "none", "uncompressed"}:
        return {}
    if value in {"zlib", "deflate", "adobe_deflate"}:
        if int(zarr_format or 3) == 3:
            try:
                from zarr.codecs import GzipCodec
            except Exception:
                return {}
            try:
                return {"compressors": [GzipCodec(level=6)]}
            except TypeError:
                return {"compressors": [GzipCodec()]}
        try:
            from numcodecs import Zlib
        except Exception:
            return {}
        return {"compressor": Zlib(level=6)}
    return {}


def _create_zarr_array_compat(root, name, shape, chunks, dtype, compression=None, zarr_format=None):
    kwargs = {
        "shape": tuple(int(v) for v in shape),
        "chunks": tuple(int(v) for v in chunks),
        "dtype": np.dtype(dtype),
    }
    codec_kwargs = _zarr_compressor_kwargs(compression, zarr_format=zarr_format)
    if codec_kwargs:
        try:
            return root.create_array(str(name), **kwargs, **codec_kwargs)
        except Exception:
            if int(zarr_format or 3) != 3 and "compressor" not in codec_kwargs:
                try:
                    return root.create_array(str(name), **kwargs, compressor=codec_kwargs.get("compressors", [None])[0])
                except Exception:
                    pass
    return root.create_array(str(name), **kwargs)


def _open_zarr_group_compat(path, mode="w", zarr_format=None):
    zarr = _get_zarr()
    if zarr_format is not None:
        try:
            return zarr.open_group(str(path), mode=mode, zarr_format=int(zarr_format))
        except TypeError:
            pass
    return zarr.open_group(str(path), mode=mode)


def _pyramid_shapes(base_shape, scale=2, tile_size=1024):
    scale = int(scale)
    if scale < 2:
        raise ValueError(f"pyramid_scale must be >= 2, got {scale}")
    shapes = [tuple(int(v) for v in base_shape)]
    current = shapes[0]
    while max(current[-2:]) > int(tile_size) and min(current[-2:]) > 1:
        next_y = max(1, int(np.ceil(current[-2] / scale)))
        next_x = max(1, int(np.ceil(current[-1] / scale)))
        current = tuple(current[:-2]) + (next_y, next_x)
        shapes.append(current)
    return shapes


def _chunks_for_shape(shape, tile_size=1024):
    shape = tuple(int(v) for v in shape)
    tile_size = int(tile_size)
    if len(shape) == 2:
        return (min(tile_size, shape[-2]), min(tile_size, shape[-1]))
    leading = tuple(1 for _ in shape[:-2])
    return leading + (min(tile_size, shape[-2]), min(tile_size, shape[-1]))


def _yx_tile_slices(shape, tile_size):
    height, width = [int(v) for v in shape[-2:]]
    yield from generate_output_tiles(height, width, int(tile_size), int(tile_size))


def _downsample_yx_mean(tile, factor, out_dtype):
    factor = int(factor)
    tile = np.asarray(tile)
    target_y = int(np.ceil(tile.shape[-2] / factor))
    target_x = int(np.ceil(tile.shape[-1] / factor))
    pad_y = target_y * factor - tile.shape[-2]
    pad_x = target_x * factor - tile.shape[-1]
    if pad_y or pad_x:
        pad_width = [(0, 0)] * tile.ndim
        pad_width[-2] = (0, pad_y)
        pad_width[-1] = (0, pad_x)
        mode = "edge" if tile.shape[-2] > 0 and tile.shape[-1] > 0 else "constant"
        tile = np.pad(tile, pad_width, mode=mode)

    reshaped = tile.reshape(tile.shape[:-2] + (target_y, factor, target_x, factor))
    out = reshaped.mean(axis=(-3, -1))
    out_dtype = np.dtype(out_dtype)
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        out = np.rint(out).clip(info.min, info.max)
    return out.astype(out_dtype, copy=False)


def _write_downsampled_level(src, dst, factor=2, tile_size=1024):
    factor = int(factor)
    read_tile = int(tile_size) * factor
    for y0, y1, x0, x1 in _yx_tile_slices(dst.shape, tile_size):
        src_y0 = y0 * factor
        src_x0 = x0 * factor
        src_y1 = min(int(src.shape[-2]), y1 * factor)
        src_x1 = min(int(src.shape[-1]), x1 * factor)
        index = (slice(None),) * (src.ndim - 2) + (slice(src_y0, src_y1), slice(src_x0, src_x1))
        tile = np.asarray(src[index])
        down = _downsample_yx_mean(tile, factor=factor, out_dtype=dst.dtype)
        dst_index = (slice(None),) * (dst.ndim - 2) + (slice(y0, y1), slice(x0, x1))
        dst[dst_index] = down[..., : y1 - y0, : x1 - x0]


def _copy_zarr_level(src, dst, tile_size=1024):
    for y0, y1, x0, x1 in _yx_tile_slices(src.shape, tile_size):
        index = (slice(None),) * (src.ndim - 2) + (slice(y0, y1), slice(x0, x1))
        dst[index] = np.asarray(src[index])


def _build_pyramid_levels(base_arr, root, scale=2, tile_size=1024, compression=None, include_level0=True, zarr_format=None):
    shapes = _pyramid_shapes(base_arr.shape, scale=scale, tile_size=tile_size)
    levels = []
    previous = base_arr
    for level_index, shape in enumerate(shapes):
        if level_index == 0:
            if include_level0:
                arr = _create_zarr_array_compat(
                    root,
                    "0",
                    shape=shape,
                    chunks=_chunks_for_shape(shape, tile_size=tile_size),
                    dtype=base_arr.dtype,
                    compression=compression,
                    zarr_format=zarr_format,
                )
                _copy_zarr_level(base_arr, arr, tile_size=tile_size)
                previous = arr
            else:
                arr = base_arr
            levels.append(arr)
            continue

        arr = _create_zarr_array_compat(
            root,
            str(level_index),
            shape=shape,
            chunks=_chunks_for_shape(shape, tile_size=tile_size),
            dtype=base_arr.dtype,
            compression=compression,
            zarr_format=zarr_format,
        )
        _write_downsampled_level(previous, arr, factor=scale, tile_size=tile_size)
        levels.append(arr)
        previous = arr
    return levels


def export_plain_zarr_to_ome_tiff(
    plain_zarr_path,
    out_path,
    metadata,
    tile_shape=(1024, 1024),
    overwrite=True,
    compression=None,
    pyramid_scale=2,
):
    out_path = Path(out_path)
    if overwrite and out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _, arr = _open_plain_zarr_array(plain_zarr_path, mode="r")
    px_x = metadata.get("pixel_size_x_um")
    px_y = metadata.get("pixel_size_y_um")
    if arr.ndim == 4:
        axes = "CZYX"
    elif arr.ndim == 3:
        axes = "CYX"
    elif arr.ndim == 2:
        axes = "YX"
    else:
        raise ValueError(f"Expected 2D YX, 3D CYX, or 4D CZYX Zarr array, got shape {arr.shape}")
    ome_metadata = {"axes": axes}
    channel_names = list(metadata.get("selected_channel_names") or [])
    if channel_names and "C" in axes:
        ome_metadata["Channel"] = {"Name": channel_names}
    if px_x is not None:
        ome_metadata["PhysicalSizeX"] = float(px_x)
        ome_metadata["PhysicalSizeXUnit"] = "\u00b5m"
    if px_y is not None:
        ome_metadata["PhysicalSizeY"] = float(px_y)
        ome_metadata["PhysicalSizeYUnit"] = "\u00b5m"
    origin_um = _origin_um_xy(metadata)
    if origin_um is not None:
        if axes == "CZYX":
            plane_count = int(arr.shape[0]) * int(arr.shape[1])
        elif axes == "CYX":
            plane_count = int(arr.shape[0])
        else:
            plane_count = 1
        ome_metadata["Plane"] = {
            "PositionX": [float(origin_um[0])] * plane_count,
            "PositionXUnit": ["\u00b5m"] * plane_count,
            "PositionY": [float(origin_um[1])] * plane_count,
            "PositionYUnit": ["\u00b5m"] * plane_count,
        }

    tile_size = int(tile_shape[0])
    temp_pyramid = out_path.parent / f".{out_path.name}.pyramid.tmp.zarr"
    if temp_pyramid.exists():
        shutil.rmtree(temp_pyramid)
    temp_root = _open_zarr_group_compat(temp_pyramid, mode="w")
    levels = _build_pyramid_levels(
        arr,
        temp_root,
        scale=pyramid_scale,
        tile_size=tile_size,
        compression=None,
        include_level0=False,
    )
    compression = _normalize_tiff_compression(compression)

    try:
        with tifffile.TiffWriter(str(out_path), bigtiff=True, ome=True) as tif:
            tif.write(
                data=_tile_iterator_from_zarr_array(levels[0], tile_shape),
                shape=tuple(int(v) for v in levels[0].shape),
                dtype=np.dtype(levels[0].dtype),
                photometric="minisblack",
                tile=tuple(int(v) for v in tile_shape),
                compression=compression,
                metadata=ome_metadata,
                subifds=max(0, len(levels) - 1),
            )
            for level in levels[1:]:
                tif.write(
                    data=_tile_iterator_from_zarr_array(level, tile_shape),
                    shape=tuple(int(v) for v in level.shape),
                    dtype=np.dtype(level.dtype),
                    photometric="minisblack",
                    tile=tuple(int(v) for v in tile_shape),
                    compression=compression,
                    metadata=None,
                    subfiletype=1,
                )
    finally:
        if temp_pyramid.exists():
            shutil.rmtree(temp_pyramid)
    _write_output_sidecar(out_path, metadata)
    return str(out_path)


def _iter_zarr_groups(group, max_depth=3):
    yield group
    if max_depth <= 0:
        return
    group_keys = getattr(group, "group_keys", None)
    if group_keys is None:
        return
    try:
        keys = list(group_keys())
    except Exception:
        return
    for key in keys:
        try:
            child = group[key]
        except Exception:
            continue
        if hasattr(child, "attrs") and not hasattr(child, "shape"):
            yield from _iter_zarr_groups(child, max_depth=max_depth - 1)


def _zarr_group_array_keys(group):
    array_keys = getattr(group, "array_keys", None)
    if array_keys is None:
        return []
    try:
        return [str(key) for key in array_keys()]
    except Exception:
        return []


def _axes_names_from_multiscale(multiscale, fallback):
    axes_meta = multiscale.get("axes") or []
    axes_names = []
    for axis in axes_meta:
        if isinstance(axis, dict):
            name = axis.get("name")
            if name:
                axes_names.append(str(name))
        elif axis:
            axes_names.append(str(axis))
    return axes_names or list(fallback)


def _patch_multiscales_value(multiscales, metadata):
    multiscales = json.loads(json.dumps(_json_ready(multiscales)))
    patched = False
    fallback_axes = list(metadata.get("axes") or ["c", "y", "x"])
    for multiscale in multiscales:
        axes_names = _axes_names_from_multiscale(multiscale, fallback_axes)
        for dataset in multiscale.get("datasets", []) or []:
            existing_scale = None
            for transform in dataset.get("coordinateTransformations", []) or []:
                if isinstance(transform, dict) and transform.get("type") == "scale":
                    scale = transform.get("scale")
                    if isinstance(scale, (list, tuple)) and len(scale) == len(axes_names):
                        existing_scale = scale
                        break
            dataset["coordinateTransformations"] = _coordinate_transformations_for_axes(
                metadata,
                axes_names,
                scale_override=existing_scale,
            )
            patched = True
    return multiscales, patched


def _patch_ome_zarr_coordinate_transformations(out_path, metadata):
    zarr = _get_zarr()
    root = zarr.open_group(str(out_path), mode="a")
    patched = False

    for group in _iter_zarr_groups(root):
        multiscales = group.attrs.get("multiscales", None)
        if multiscales:
            multiscales, group_patched = _patch_multiscales_value(multiscales, metadata)
            group.attrs["multiscales"] = multiscales
            patched = bool(patched or group_patched)
            _patch_ome_zarr_channel_metadata(group, metadata)

        group.attrs["image_realignment_metadata"] = _json_ready(metadata)
        for key, value in _coordinate_metadata_attrs(metadata).items():
            group.attrs[key] = _json_ready(value)

    if not patched:
        array_keys = _zarr_group_array_keys(root)
        if array_keys:
            axes_names = list(metadata.get("axes") or ["c", "y", "x"])
            has_physical_units = (
                _finite_float_or_none(metadata.get("pixel_size_x_um")) is not None
                and _finite_float_or_none(metadata.get("pixel_size_y_um")) is not None
            )
            root.attrs["multiscales"] = [
                {
                    "version": "0.5",
                    "name": str(metadata.get("image_id", "aligned_image")),
                    "axes": _ome_zarr_axes_metadata(axes_names, has_physical_units),
                    "datasets": [
                        {
                            "path": array_keys[0],
                            "coordinateTransformations": _coordinate_transformations_for_axes(
                                metadata,
                                axes_names,
                            ),
                        }
                    ],
                }
            ]
            _patch_ome_zarr_channel_metadata(root, metadata)
            patched = True

    return patched


def export_plain_zarr_to_ome_zarr(
    plain_zarr_path,
    out_path,
    metadata,
    chunks=(2048, 2048),
    overwrite=True,
    zarr_format=3,
    pyramid_scale=2,
    tile_size=1024,
    compression="zlib",
):
    out_path = Path(out_path)
    if overwrite and out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _, arr = _open_plain_zarr_array(plain_zarr_path, mode="r")
    px_x = metadata.get("pixel_size_x_um")
    px_y = metadata.get("pixel_size_y_um")
    has_physical_units = px_x is not None and px_y is not None

    if arr.ndim == 4:
        axes_names = ["c", "z", "y", "x"]
    elif arr.ndim == 3:
        axes_names = ["c", "y", "x"]
    elif arr.ndim == 2:
        axes_names = ["y", "x"]
    else:
        raise ValueError(f"Expected 2D YX, 3D CYX, or 4D CZYX Zarr array, got shape {arr.shape}")

    root = _open_zarr_group_compat(out_path, mode="w", zarr_format=zarr_format)
    levels = _build_pyramid_levels(
        arr,
        root,
        scale=pyramid_scale,
        tile_size=tile_size,
        compression=compression,
        include_level0=True,
        zarr_format=zarr_format,
    )

    datasets = []
    for level_index, level in enumerate(levels):
        factor = int(pyramid_scale) ** int(level_index)
        scale_override = []
        for axis in axes_names:
            axis = str(axis).lower()
            if axis == "x":
                scale_override.append((float(px_x) if px_x is not None else 1.0) * factor)
            elif axis == "y":
                scale_override.append((float(px_y) if px_y is not None else 1.0) * factor)
            else:
                scale_override.append(1.0)
        datasets.append(
            {
                "path": str(level_index),
                "coordinateTransformations": _coordinate_transformations_for_axes(
                    metadata,
                    axes_names,
                    scale_override=scale_override,
                ),
            }
        )

    root.attrs["multiscales"] = [
        {
            "version": "0.5" if int(zarr_format) == 3 else "0.4",
            "name": str(metadata.get("image_id", "aligned_image")),
            "axes": _ome_zarr_axes_metadata(axes_names, has_physical_units),
            "datasets": datasets,
        }
    ]
    root.attrs["image_realignment_metadata"] = _json_ready(metadata)
    root.attrs["pyramid"] = _json_ready(
        {
            "scale": int(pyramid_scale),
            "tile_size": int(tile_size),
            "n_levels": int(len(levels)),
            "compression": compression,
        }
    )
    for key, value in _coordinate_metadata_attrs({**metadata, "axes": axes_names}).items():
        root.attrs[key] = _json_ready(value)
    _patch_ome_zarr_channel_metadata(root, metadata)
    _write_output_sidecar(out_path, metadata)
    return str(out_path)


def realign_one_image_worker(
    row_dict,
    backend="cpu",
    requested_backend=None,
    effective_backend=None,
    host_platform=None,
    overwrite=True,
    align_dtype="uint16",
    cpu_read_dtype="uint16",
    gpu_source_dtype="float32",
    align_tile_y=2048,
    align_tile_x=2048,
    write_chunks=(2048, 2048),
    align_resampler="affine",
    source_window_pad_px=2,
    output_formats=(".zarr",),
    output_dir="outputs",
    ome_zarr_format=3,
    ome_tiff_compression="zlib",
    ome_zarr_compression="zlib",
    ome_tile_size=1024,
    pyramid_scale=2,
    device_id=None,
    inner_num_threads=None,
):
    t0_total = now()
    formats = normalize_output_formats(output_formats)
    cp = None
    temp_stage = None
    image_id = row_dict["image_id"]

    try:
        backend_info = resolve_backend(backend, platform_tag=host_platform)
        backend = backend_info["implementation_backend"]
        requested_backend = requested_backend or backend_info["requested_backend"]
        effective_backend = effective_backend or backend_info["effective_backend"]
        host_platform = host_platform or backend_info["host_platform"]

        if backend == "gpu":
            cp = _get_cupy()
            if device_id is not None:
                cp.cuda.Device(int(device_id)).use()

        out_paths = dict(row_dict.get("output_paths") or {})
        if not out_paths:
            _, out_paths = output_paths_for_image(
                row_dict["image_path"],
                output_dir=output_dir,
                output_formats=formats,
            )

        if ".zarr" in formats:
            stage_zarr = Path(out_paths[".zarr"])
            stage_is_final = True
        else:
            stage_zarr = Path(output_dir) / "_realignment_staging" / f"{image_id}_aligned.stage.zarr"
            temp_stage = stage_zarr
            stage_is_final = False

        expected_inputs = row_dict.get("alignment_expected_inputs")
        reuse_stage = (
            stage_is_final
            and not overwrite
            and stage_zarr.exists()
            and expected_inputs is not None
            and _metadata_matches_sidecar(stage_zarr, expected_inputs)
        )
        if reuse_stage:
            metadata = _read_zarr_metadata(stage_zarr) or {"alignment_inputs": expected_inputs}
            _, stage_arr = _open_plain_zarr_array(stage_zarr, mode="r")
            stage_shape = tuple(int(v) for v in stage_arr.shape)
            stage_shape_yx = stage_shape[-2:] if len(stage_shape) >= 2 else stage_shape
            stage_axes = ["c", "z", "y", "x"] if len(stage_shape) == 4 else ["c", "y", "x"] if len(stage_shape) == 3 else ["y", "x"]
            aligned = {
                "metadata": metadata,
                "aligned_shape_yx": stage_shape_yx,
                "aligned_shape_cyx": stage_shape,
                "aligned_shape": stage_shape,
                "aligned_axes": stage_axes,
                "n_tiles": 0,
                "n_tiles_skipped": 0,
                "t_inverse_s": 0.0,
                "t_zarr_init_s": 0.0,
                "t_read_s": 0.0,
                "t_compute_s": 0.0,
                "t_write_s": 0.0,
            }
        else:
            with _threadpool_context(inner_num_threads):
                aligned = align_image_to_plain_zarr(
                    row_dict=row_dict,
                    out_zarr_path=stage_zarr,
                    backend=requested_backend,
                    overwrite=True,
                    align_dtype=np.dtype(align_dtype),
                    cpu_read_dtype=np.dtype(cpu_read_dtype),
                    gpu_source_dtype=gpu_source_dtype,
                    align_tile_y=align_tile_y,
                    align_tile_x=align_tile_x,
                    write_chunks=write_chunks,
                    align_resampler=align_resampler,
                    source_window_pad_px=source_window_pad_px,
                )
            metadata = aligned["metadata"]

        exported = {}
        if ".zarr" in formats:
            _write_output_sidecar(stage_zarr, metadata)
            exported[".zarr"] = str(stage_zarr)
        if ".ome.zarr" in formats:
            out_path = out_paths[".ome.zarr"]
            if not overwrite and _metadata_matches_sidecar(out_path, metadata.get("alignment_inputs", metadata)):
                exported[".ome.zarr"] = str(out_path)
            else:
                exported[".ome.zarr"] = export_plain_zarr_to_ome_zarr(
                    plain_zarr_path=stage_zarr,
                    out_path=out_path,
                    metadata=metadata,
                    chunks=(int(ome_tile_size), int(ome_tile_size)),
                    overwrite=True,
                    zarr_format=ome_zarr_format,
                    pyramid_scale=pyramid_scale,
                    tile_size=ome_tile_size,
                    compression=ome_zarr_compression,
                )
        if ".ome.tiff" in formats:
            out_path = out_paths[".ome.tiff"]
            if not overwrite and _metadata_matches_sidecar(out_path, metadata.get("alignment_inputs", metadata)):
                exported[".ome.tiff"] = str(out_path)
            else:
                exported[".ome.tiff"] = export_plain_zarr_to_ome_tiff(
                    plain_zarr_path=stage_zarr,
                    out_path=out_path,
                    metadata=metadata,
                    tile_shape=(int(ome_tile_size), int(ome_tile_size)),
                    overwrite=True,
                    compression=ome_tiff_compression,
                    pyramid_scale=pyramid_scale,
                )

        if temp_stage is not None and temp_stage.exists() and not stage_is_final:
            shutil.rmtree(temp_stage)

        return {
            "ok": True,
            "image_id": image_id,
            "pair_index": int(row_dict.get("pair_index", 0)),
            "backend": requested_backend,
            "implementation_backend": backend,
            "effective_backend": effective_backend,
            "host_platform": host_platform,
            "device_id": None if device_id is None else int(device_id),
            "output_paths": exported,
            "aligned_shape_yx": aligned["aligned_shape_yx"],
            "aligned_shape_cyx": aligned["aligned_shape_cyx"],
            "aligned_shape": aligned.get("aligned_shape", aligned["aligned_shape_cyx"]),
            "aligned_axes": aligned.get("aligned_axes"),
            "n_tiles": int(aligned["n_tiles"]),
            "n_tiles_skipped": int(aligned["n_tiles_skipped"]),
            "reused_plain_zarr": bool(reuse_stage),
            "t_total_s": float(now() - t0_total),
            **{k: float(v) for k, v in aligned.items() if k.startswith("t_")},
            "error": None,
        }

    except Exception as exc:
        return {
            "ok": False,
            "image_id": image_id,
            "pair_index": int(row_dict.get("pair_index", 0)),
            "backend": requested_backend or backend,
            "implementation_backend": backend,
            "effective_backend": effective_backend,
            "host_platform": host_platform,
            "device_id": None if device_id is None else int(device_id),
            "t_total_s": float(now() - t0_total),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        gc.collect()
        if backend == "gpu" and cp is not None:
            try:
                _free_cupy_blocks(cp)
            except Exception:
                pass


def _jobs_from_plan(plan_df, run_only_needed=True):
    work = plan_df.copy()
    if run_only_needed and "needs_run" in work.columns:
        work = work.loc[work["needs_run"].astype(bool)].copy()
    return [row.to_dict() for _, row in work.iterrows()]


def _notify(progress_callback, n_total, n_done, n_failed, current_image_id=None, result=None, stage_name="realignment"):
    if progress_callback is None:
        return
    progress_callback(
        {
            "n_total": int(n_total),
            "n_done": int(n_done),
            "n_failed": int(n_failed),
            "current_image_id": current_image_id,
            "result": result,
            "stage_name": stage_name,
            **_memory_status(),
        }
    )


def run_realignment_batch(
    plan_df,
    backend="cpu",
    execution_mode="parallel",
    workers_cpu=4,
    workers_gpu=1,
    gpu_device_ids=(0,),
    gpu_jobs_per_device=1,
    run_only_needed=True,
    progress_callback=None,
    mp_context_name="spawn",
    ram_start_pct=85.0,
    ram_resume_pct=75.0,
    ram_poll_s=10.0,
    worker_stagger_s=0.0,
    **worker_kwargs,
):
    backend_info = resolve_backend(backend)
    requested_backend = backend_info["requested_backend"]
    backend = backend_info["implementation_backend"]
    effective_backend = backend_info["effective_backend"]
    host_platform = backend_info["host_platform"]

    if execution_mode not in {"serial", "parallel"}:
        raise ValueError(f"Unsupported execution_mode: {execution_mode}")
    if backend == "gpu":
        _get_cupy()
        _get_cupy_affine_transform()

    jobs = _jobs_from_plan(plan_df, run_only_needed=run_only_needed)
    n_total = len(jobs)
    n_done = 0
    n_failed = 0
    results = []
    _notify(progress_callback, n_total, n_done, n_failed)

    if n_total == 0:
        return {"results": results, "profile_df": pd.DataFrame(), "n_total": 0, "n_done": 0, "n_failed": 0}

    common_kwargs = dict(worker_kwargs)
    common_kwargs["backend"] = backend
    common_kwargs["requested_backend"] = requested_backend
    common_kwargs["effective_backend"] = effective_backend
    common_kwargs["host_platform"] = host_platform

    if execution_mode == "serial":
        device_id = list(gpu_device_ids)[0] if backend == "gpu" and gpu_device_ids else None
        for job in jobs:
            ram_status = _wait_for_ram(ram_start_pct, ram_resume_pct, ram_poll_s)
            _notify(progress_callback, n_total, n_done, n_failed, current_image_id=job.get("image_id"))
            result = realign_one_image_worker(job, device_id=device_id, **common_kwargs)
            result.update(
                {
                    "ram_before_start_pct": ram_status.get("ram_percent"),
                    "ram_wait_before_start_s": ram_status.get("ram_wait_s"),
                }
            )
            results.append(result)
            if result.get("ok", False):
                n_done += 1
            else:
                n_failed += 1
            _notify(progress_callback, n_total, n_done, n_failed, result=result)
        run_result = {
            "results": results,
            "profile_df": pd.DataFrame(results),
            "n_total": n_total,
            "n_done": n_done,
            "n_failed": n_failed,
        }
        return run_result

    import multiprocessing as mp

    ctx = mp.get_context(mp_context_name)
    if backend == "gpu":
        device_ids = list(gpu_device_ids)
        if not device_ids:
            raise ValueError("GPU backend selected but GPU_DEVICE_IDS is empty.")
        max_workers = min(
            n_total,
            max(1, int(workers_gpu)) * max(1, int(gpu_jobs_per_device)),
            len(device_ids) * max(1, int(gpu_jobs_per_device)),
        )
    else:
        device_ids = [None]
        max_workers = min(n_total, max(1, int(workers_cpu)))

    pending = list(jobs)
    future_to_info = {}

    def submit_next(executor):
        if not pending:
            return False
        job = pending.pop(0)
        active_count = len(future_to_info)
        device_id = None
        if backend == "gpu":
            device_id = device_ids[active_count % len(device_ids)]
        ram_status = _wait_for_ram(ram_start_pct, ram_resume_pct, ram_poll_s)
        kwargs = dict(common_kwargs)
        future = executor.submit(realign_one_image_worker, job, device_id=device_id, **kwargs)
        future_to_info[future] = {
            "image_id": job.get("image_id"),
            "ram_status": ram_status,
            "device_id": device_id,
        }
        _notify(progress_callback, n_total, n_done, n_failed, current_image_id=job.get("image_id"))
        return True

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        while pending and len(future_to_info) < max_workers:
            submit_next(executor)
            if pending and worker_stagger_s and worker_stagger_s > 0:
                time.sleep(float(worker_stagger_s))

        while future_to_info:
            done_futures, _ = wait(future_to_info, return_when=FIRST_COMPLETED)
            for future in done_futures:
                info = future_to_info.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "image_id": info.get("image_id"),
                        "backend": requested_backend,
                        "implementation_backend": backend,
                        "effective_backend": effective_backend,
                        "host_platform": host_platform,
                        "device_id": info.get("device_id"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                result.update(
                    {
                        "ram_before_start_pct": info["ram_status"].get("ram_percent"),
                        "ram_wait_before_start_s": info["ram_status"].get("ram_wait_s"),
                    }
                )
                results.append(result)
                if result.get("ok", False):
                    n_done += 1
                else:
                    n_failed += 1
                _notify(progress_callback, n_total, n_done, n_failed, result=result)

                if pending:
                    submit_next(executor)
                    if pending and worker_stagger_s and worker_stagger_s > 0:
                        time.sleep(float(worker_stagger_s))

    run_result = {
        "results": results,
        "profile_df": pd.DataFrame(results),
        "n_total": n_total,
        "n_done": n_done,
        "n_failed": n_failed,
    }
    return run_result


def summarize_realignment_results(profile_df):
    if profile_df is None or len(profile_df) == 0:
        return pd.DataFrame()
    cols = [
        "ok",
        "image_id",
        "backend",
        "implementation_backend",
        "effective_backend",
        "host_platform",
        "device_id",
        "aligned_axes",
        "aligned_shape",
        "aligned_shape_cyx",
        "aligned_shape_yx",
        "n_tiles",
        "n_tiles_skipped",
        "t_total_s",
        "error",
    ]
    keep = [col for col in cols if col in profile_df.columns]
    return profile_df[keep].copy()


def _preview_step_for_shape(shape_yx, max_size_px):
    height, width = [int(v) for v in shape_yx]
    return max(1, int(np.ceil(max(height, width) / float(max_size_px))))


def _first_channel_from_row(row, default=0):
    value = row.get("channel_indices", None) if hasattr(row, "get") else None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and value:
        return int(value[0])
    if isinstance(value, (int, np.integer)):
        return int(value)
    return int(default)


def read_source_preview(row_or_path, channel_index=None, max_size_px=1024, dtype=None):
    """Read a downsampled source-channel preview using tifffile-backed tiled reads."""
    if isinstance(row_or_path, (str, Path)):
        image_path = row_or_path
        channel = 0 if channel_index is None else int(channel_index)
    else:
        row = row_or_path.to_dict() if hasattr(row_or_path, "to_dict") else dict(row_or_path)
        image_path = row["image_path"]
        channel = _first_channel_from_row(row) if channel_index is None else int(channel_index)

    with TiledTiffImageReader(image_path, dtype=dtype) as reader:
        step = _preview_step_for_shape(reader.shape_yx, max_size_px=max_size_px)
        return reader.read_window(
            channel,
            0,
            reader.shape_yx[0],
            0,
            reader.shape_yx[1],
            step=step,
            dtype=dtype,
        )


def _find_first_zarr_array(node):
    if hasattr(node, "shape") and hasattr(node, "__getitem__"):
        return node

    for key in ("0", 0):
        try:
            return _find_first_zarr_array(node[key])
        except Exception:
            pass

    for method_name in ("array_keys", "group_keys"):
        method = getattr(node, method_name, None)
        if method is None:
            continue
        try:
            keys = list(method())
        except Exception:
            keys = []
        for key in keys:
            try:
                return _find_first_zarr_array(node[key])
            except Exception:
                pass

    raise ValueError("Could not find an image array inside the Zarr store")


def read_aligned_preview(output_path, channel_index=0, z_index=0, max_size_px=1024):
    """Read a downsampled aligned-output preview from .zarr, .ome.zarr, or .ome.tiff."""
    output_path = Path(output_path)
    lower = str(output_path).lower()
    channel = int(channel_index)

    if lower.endswith((".ome.tiff", ".ome.tif", ".tiff", ".tif")):
        with TiledTiffImageReader(output_path) as reader:
            step = _preview_step_for_shape(reader.shape_yx, max_size_px=max_size_px)
            return reader.read_window(
                min(channel, reader.channel_count - 1),
                0,
                reader.shape_yx[0],
                0,
                reader.shape_yx[1],
                step=step,
            )

    zarr = _get_zarr()
    root = zarr.open(str(output_path), mode="r")
    arr = _find_first_zarr_array(root)
    shape = tuple(int(v) for v in arr.shape)
    if len(shape) == 2:
        step = _preview_step_for_shape(shape, max_size_px=max_size_px)
        return np.asarray(arr[::step, ::step])
    if len(shape) == 3:
        step = _preview_step_for_shape(shape[-2:], max_size_px=max_size_px)
        channel = min(max(channel, 0), shape[0] - 1)
        return np.asarray(arr[channel, ::step, ::step])
    if len(shape) == 4:
        step = _preview_step_for_shape(shape[-2:], max_size_px=max_size_px)
        channel = min(max(channel, 0), shape[0] - 1)
        z_index = min(max(int(z_index), 0), shape[1] - 1)
        return np.asarray(arr[channel, z_index, ::step, ::step])
    raise ValueError(f"Expected aligned output to be YX, CYX, or CZYX, got shape {shape}")


def _normalize_preview_for_display(image, p_low=1.0, p_high=99.0):
    image = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(image)
    if not valid.any():
        return np.zeros(image.shape, dtype=np.float32)
    lo, hi = np.percentile(image[valid], [float(p_low), float(p_high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(image[valid]))
        hi = float(np.nanmax(image[valid]))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _output_path_from_row_or_result(row, result=None, preference=(".zarr", ".ome.zarr", ".ome.tiff")):
    result = result or {}
    output_paths = result.get("output_paths") if isinstance(result, dict) else None
    if isinstance(output_paths, dict):
        for fmt in preference:
            path = output_paths.get(fmt)
            if path and Path(path).exists():
                return str(path)

    row_keys = {
        ".zarr": "out_zarr",
        ".ome.zarr": "out_ome_zarr",
        ".ome.tiff": "out_ome_tiff",
    }
    for fmt in preference:
        key = row_keys.get(fmt)
        path = row.get(key) if hasattr(row, "get") else None
        if isinstance(path, str) and path and Path(path).exists():
            return str(path)
    return None


def plot_realignment_qc(
    plan_df,
    results_df=None,
    max_images=6,
    max_size_px=1024,
    source_channel=None,
    aligned_channel=0,
    output_format_preference=(".zarr", ".ome.zarr", ".ome.tiff"),
    cmap="gray",
):
    """Display downsampled before/after panels for completed alignments."""
    import matplotlib.pyplot as plt

    if plan_df is None or len(plan_df) == 0:
        print("No input pairs configured for QC.")
        return None

    result_by_image_id = {}
    if results_df is not None and len(results_df) and "image_id" in results_df.columns:
        for _, result_row in results_df.iterrows():
            result_by_image_id[str(result_row["image_id"])] = result_row.to_dict()

    rows = []
    for _, plan_row in plan_df.iterrows():
        row = plan_row.to_dict()
        result = result_by_image_id.get(str(row.get("image_id")), {})
        if result and result.get("ok") is False:
            continue
        output_path = _output_path_from_row_or_result(
            row,
            result=result,
            preference=output_format_preference,
        )
        if output_path is not None:
            rows.append((row, result, output_path))
        if len(rows) >= int(max_images):
            break

    if not rows:
        print("No aligned output files were found for QC.")
        return None

    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 4 * len(rows)), squeeze=False)
    for i, (row, _result, output_path) in enumerate(rows):
        image_id = row.get("image_id", f"image_{i + 1}")
        channel = _first_channel_from_row(row) if source_channel is None else int(source_channel)

        try:
            before = read_source_preview(row, channel_index=channel, max_size_px=max_size_px)
            axes[i, 0].imshow(_normalize_preview_for_display(before), cmap=cmap, interpolation="nearest")
            axes[i, 0].set_title(f"{image_id}: before, channel {channel}")
        except Exception as exc:
            axes[i, 0].text(0.5, 0.5, f"Before preview failed\n{type(exc).__name__}: {exc}", ha="center", va="center")
        axes[i, 0].axis("off")

        try:
            after = read_aligned_preview(
                output_path,
                channel_index=aligned_channel,
                max_size_px=max_size_px,
            )
            axes[i, 1].imshow(_normalize_preview_for_display(after), cmap=cmap, interpolation="nearest")
            axes[i, 1].set_title(f"{image_id}: after, aligned channel {int(aligned_channel)}")
        except Exception as exc:
            axes[i, 1].text(0.5, 0.5, f"After preview failed\n{type(exc).__name__}: {exc}", ha="center", va="center")
        axes[i, 1].axis("off")

        zarr_meta = _read_zarr_metadata(output_path) or _read_zarr_root_attrs(output_path)
        origin_px = _origin_px_xy(zarr_meta) if str(output_path).lower().endswith(".zarr") else None
        print(f"{image_id}: QC output={output_path}")
        if origin_px is not None:
            print(f"{image_id}: origin_global_xy_px={origin_px}")

    plt.tight_layout()
    return fig


_COMPLETION_MESSAGE = "I hope you have a wonderful day, Valisha! \U0001F60A"


def _completion_message():
    return _COMPLETION_MESSAGE
