"""Minimal IF registration, pixel calibration, and plaque-shape detection.

Coordinate convention
---------------------
Points are stored as (x, y), while NumPy images are indexed as [y, x].
The alignment matrix is assumed to map full-resolution IF pixels to Xenium
pixels. Xenium spatial coordinates are assumed to be in micrometres.

The core IF-scale calculation does not require ``alphashape`` or ``shapely``.
The non-round plaque option does; install them with:
    pip install alphashape shapely
"""

from pathlib import Path

import numpy as np
import tifffile as tf
from scipy import ndimage as ndi
from scipy.spatial import Delaunay
from skimage import feature, filters, measure, morphology


XENIUM_PIXEL_SIZE_UM = 0.2125


def read_ome_level(path, level=1, channel=0):
    """Read one channel from one OME-TIFF pyramid level."""
    with tf.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", None) or [series])
        if not 0 <= level < len(levels):
            raise ValueError(f"level must be 0..{len(levels) - 1}; got {level}")

        full_shape = levels[0].shape[-2:]
        level_shape = levels[level].shape[-2:]
        image = np.asarray(levels[level].asarray())

    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[channel]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D channel image; got {image.shape}")

    full_y, full_x = full_shape
    level_y, level_x = level_shape
    downsample_xy = np.array([full_x / level_x, full_y / level_y], float)
    return image.astype(np.float32), downsample_xy


def apply_affine_xy(points_xy, matrix):
    """Apply a 3 x 3 homogeneous transform to (x, y) points."""
    points_xy = np.asarray(points_xy, float)
    homogeneous = np.column_stack((points_xy, np.ones(len(points_xy))))
    transformed = homogeneous @ np.asarray(matrix, float).T
    return transformed[:, :2] / transformed[:, 2, None]


def xenium_um_to_if_level_px(spatial_xy_um, if_to_xenium, downsample_xy):
    """Map Xenium coordinates in micrometres onto the selected IF level."""
    xenium_xy_px = np.asarray(spatial_xy_um, float) / XENIUM_PIXEL_SIZE_UM
    if_full_xy_px = apply_affine_xy(xenium_xy_px, np.linalg.inv(if_to_xenium))
    return if_full_xy_px / np.asarray(downsample_xy, float)


def if_level_pixel_scale(if_to_xenium, downsample_xy):
    """Return x/y resolution, true pixel area, and equivalent pixel size.

    The determinant gives the correct area even if the affine has shear.
    ``equivalent_um`` is sqrt(area), not the arithmetic mean of x/y lengths.
    """
    matrix = np.asarray(if_to_xenium, float)
    if not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("Pixel scale is position-dependent for a projective transform")

    ds_x, ds_y = np.asarray(downsample_xy, float)
    linear = matrix[:2, :2]
    x_um = np.linalg.norm(linear[:, 0]) * ds_x * XENIUM_PIXEL_SIZE_UM
    y_um = np.linalg.norm(linear[:, 1]) * ds_y * XENIUM_PIXEL_SIZE_UM
    area_um2 = abs(np.linalg.det(linear)) * ds_x * ds_y * XENIUM_PIXEL_SIZE_UM**2
    return {
        "x_um": x_um,
        "y_um": y_um,
        "area_um2": area_um2,
        "equivalent_um": np.sqrt(area_um2),
    }


def plaque_signal(image, tissue_mask, pixel_scale, spot_sigma_um=1.0,
                  background_sigma_um=25.0):
    """Suppress broad background and retain locally bright plaque signal."""
    sx, sy = pixel_scale["x_um"], pixel_scale["y_um"]
    image = np.where(tissue_mask, image, 0).astype(np.float32)
    small = ndi.gaussian_filter(image, (spot_sigma_um / sy, spot_sigma_um / sx))
    background = ndi.gaussian_filter(
        image, (background_sigma_um / sy, background_sigma_um / sx)
    )
    signal = np.clip(small - background, 0, None)
    signal[~tissue_mask] = 0
    high = np.percentile(signal[tissue_mask], 99.9)
    return np.clip(signal / high, 0, 1) if high > 0 else signal


def segment_plaque_candidates(signal, tissue_mask, pixel_scale,
                              threshold=None, min_area_um2=10.0):
    """Threshold the intensity image before either shape model is applied."""
    values = signal[tissue_mask]
    threshold = filters.threshold_otsu(values) if threshold is None else threshold
    binary = (signal >= threshold) & tissue_mask
    min_pixels = max(1, round(min_area_um2 / pixel_scale["area_um2"]))
    binary = morphology.remove_small_objects(binary, min_size=min_pixels)
    return morphology.binary_closing(binary, morphology.disk(1))


def detect_round_plaques(signal, tissue_mask, pixel_scale,
                         min_diameter_um=3.0, max_diameter_um=400.0,
                         threshold=0.05, num_sigma=12):
    """Original round-plaque option: Laplacian-of-Gaussian circles."""
    pixel_um = pixel_scale["equivalent_um"]
    blobs = feature.blob_log(
        np.where(tissue_mask, signal, 0),
        min_sigma=min_diameter_um / (2 * np.sqrt(2) * pixel_um),
        max_sigma=max_diameter_um / (2 * np.sqrt(2) * pixel_um),
        num_sigma=num_sigma,
        threshold=threshold,
    )
    # blob_log returns (y, x, sigma); radius = sqrt(2) * sigma.
    return [
        {"x_px": x, "y_px": y, "radius_px": np.sqrt(2) * sigma}
        for y, x, sigma in blobs
    ]


def _sample_component_xy_um(component_mask, pixel_scale, spacing_um=2.0):
    """Sample component interiors plus boundaries for stable triangulation."""
    sx, sy = pixel_scale["x_um"], pixel_scale["y_um"]
    step_x = max(1, round(spacing_um / sx))
    step_y = max(1, round(spacing_um / sy))
    y, x = np.nonzero(component_mask)
    keep = (x % step_x == 0) & (y % step_y == 0)
    interior = np.column_stack((x[keep] * sx, y[keep] * sy))

    contours = measure.find_contours(component_mask, 0.5)
    boundary = [np.column_stack((c[:, 1] * sx, c[:, 0] * sy)) for c in contours]
    points = np.vstack([interior, *boundary]) if boundary else interior
    return np.unique(np.round(points, 6), axis=0)


def _alpha_shape_from_delaunay(points_xy_um, alpha_per_um):
    """Build an alpha shape explicitly from SciPy Delaunay triangles."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    triangles = Delaunay(points_xy_um).simplices
    xyz = points_xy_um[triangles]
    a = np.linalg.norm(xyz[:, 1] - xyz[:, 0], axis=1)
    b = np.linalg.norm(xyz[:, 2] - xyz[:, 1], axis=1)
    c = np.linalg.norm(xyz[:, 0] - xyz[:, 2], axis=1)
    semiperimeter = (a + b + c) / 2
    triangle_area = np.sqrt(
        np.maximum(semiperimeter * (semiperimeter - a)
                   * (semiperimeter - b) * (semiperimeter - c), 0)
    )
    circumradius = a * b * c / np.maximum(4 * triangle_area, np.finfo(float).eps)
    kept = xyz[circumradius <= 1.0 / alpha_per_um]
    if len(kept) == 0:
        return None
    return unary_union([Polygon(triangle) for triangle in kept]).buffer(0)


def detect_nonround_plaques(binary_mask, pixel_scale, alpha_per_um=0.15,
                            min_area_um2=10.0, backend="alphashape"):
    """Return plaque polygons without assuming plaques are circular.

    ``backend='alphashape'`` calls the alphashape library.  The
    ``backend='delaunay'`` option exposes the equivalent SciPy triangulation,
    which is useful for inspecting/tuning the retained triangles. Coordinates
    and all geometry measurements are in micrometres.
    """
    from shapely.geometry import MultiPolygon

    if alpha_per_um <= 0:
        raise ValueError("alpha_per_um must be positive")

    labels = measure.label(binary_mask)
    plaques = []
    for region in measure.regionprops(labels):
        if region.area * pixel_scale["area_um2"] < min_area_um2:
            continue

        component = labels == region.label
        points = _sample_component_xy_um(component, pixel_scale)
        if len(points) < 4:
            continue

        if backend == "alphashape":
            import alphashape
            geometry = alphashape.alphashape(points, alpha_per_um)
        elif backend == "delaunay":
            geometry = _alpha_shape_from_delaunay(points, alpha_per_um)
        else:
            raise ValueError("backend must be 'alphashape' or 'delaunay'")

        if geometry is None or geometry.is_empty:
            continue
        parts = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
        for polygon in parts:
            if polygon.area >= min_area_um2:
                plaques.append({
                    "geometry": polygon,
                    "area_um2": polygon.area,
                    "perimeter_um": polygon.length,
                    "centroid_x_um": polygon.centroid.x,
                    "centroid_y_um": polygon.centroid.y,
                    # Summary only; the stored boundary remains non-circular.
                    "equivalent_diameter_um": 2 * np.sqrt(polygon.area / np.pi),
                })
    return plaques


if __name__ == "__main__":
    # Keep the chosen pyramid level in one place. Both scale and coordinates
    # are then guaranteed to refer to the same image array.
    IF_PATH = Path("/path/to/APPPS1_infectedIF.ome.tif")
    MATRIX_PATH = Path("/path/to/IF_to_Xenium_matrix.csv")
    PLAQUE_LEVEL = 1
    PLAQUE_CHANNEL = 0

    if_image, downsample_xy = read_ome_level(
        IF_PATH, level=PLAQUE_LEVEL, channel=PLAQUE_CHANNEL
    )
    if_to_xenium = np.loadtxt(MATRIX_PATH, delimiter=",")
    scale = if_level_pixel_scale(if_to_xenium, downsample_xy)
    print(f"level {PLAQUE_LEVEL} downsample (x, y): {downsample_xy}")
    print(f"IF pixel: {scale['x_um']:.4f} x {scale['y_um']:.4f} µm")
    print(f"IF pixel area: {scale['area_um2']:.4f} µm²")

