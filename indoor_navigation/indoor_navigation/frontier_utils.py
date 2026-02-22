from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class Frontier:
    # pixels (col=x, row=y) in map grid coordinates
    cells: np.ndarray  # shape (N, 2) with [x, y]
    centroid: Tuple[float, float]  # (x, y) in world meters
    size: int


def occupancygrid_to_numpy(data, width: int, height: int) -> np.ndarray:
    """
    Converts OccupancyGrid.data (flat list) to a (H, W) numpy array of int8.
    Values: -1 unknown, 0 free, 100 occupied (or similar).
    """
    arr = np.asarray(data, dtype=np.int16).reshape((height, width))
    return arr


def find_frontier_mask(grid: np.ndarray) -> np.ndarray:
    """
    Frontier definition:
      - cell is FREE (== 0)
      - at least one 8-neighbor is UNKNOWN (== -1)
    Returns a boolean mask (H, W) where True indicates frontier cell.
    """
    free = (grid == 0)
    unknown = (grid == -1)

    # Build "unknown in neighborhood" mask by shifting unknown around
    unk_neigh = np.zeros_like(unknown, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(unknown, dtype=bool)

            # compute slice ranges safely
            y_src0 = max(0, -dy)
            y_src1 = grid.shape[0] - max(0, dy)
            x_src0 = max(0, -dx)
            x_src1 = grid.shape[1] - max(0, dx)

            y_dst0 = max(0, dy)
            y_dst1 = grid.shape[0] - max(0, -dy)
            x_dst0 = max(0, dx)
            x_dst1 = grid.shape[1] - max(0, -dx)

            shifted[y_dst0:y_dst1, x_dst0:x_dst1] = unknown[y_src0:y_src1, x_src0:x_src1]
            unk_neigh |= shifted

    return free & unk_neigh


def _connected_components_8(mask: np.ndarray) -> List[np.ndarray]:
    """
    Simple 8-connected components labeling for boolean mask.
    Returns list of components, each component is array of [x, y] integer cells.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[np.ndarray] = []

    # neighbor offsets (8-connect)
    nbrs = [(-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)]

    ys, xs = np.where(mask)
    for y0, x0 in zip(ys, xs):
        if visited[y0, x0]:
            continue
        # BFS/DFS
        stack = [(x0, y0)]
        visited[y0, x0] = True
        cells = []

        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx, dy in nbrs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((nx, ny))

        components.append(np.asarray(cells, dtype=np.int32))

    return components


def grid_to_world(x: float, y: float, origin_xy: Tuple[float, float], resolution: float) -> Tuple[float, float]:
    """
    Converts grid cell coords (x, y) to world meters.
    Here x,y can be cell indices; we map to cell center.
    """
    ox, oy = origin_xy
    wx = ox + (x + 0.5) * resolution
    wy = oy + (y + 0.5) * resolution
    return wx, wy


def extract_frontiers(
    grid: np.ndarray,
    origin_xy: Tuple[float, float],
    resolution: float,
    min_cluster_size: int = 10,
) -> List[Frontier]:
    """
    Returns clustered frontiers with centroids in world coordinates.
    """
    frontier_mask = find_frontier_mask(grid)
    comps = _connected_components_8(frontier_mask)

    frontiers: List[Frontier] = []
    for comp in comps:
        if comp.shape[0] < min_cluster_size:
            continue
        # centroid in grid coords
        cx = float(np.mean(comp[:, 0]))
        cy = float(np.mean(comp[:, 1]))
        wx, wy = grid_to_world(cx, cy, origin_xy, resolution)
        frontiers.append(Frontier(cells=comp, centroid=(wx, wy), size=int(comp.shape[0])))

    # sort: largest first (useful later)
    frontiers.sort(key=lambda f: f.size, reverse=True)
    return frontiers