from __future__ import annotations

import csv
import json
import math
from pathlib import Path


# =========================
# 你通常只改这里
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent

NODES_FILE = SCRIPT_DIR / "chengdu_nodes.txt"   # 点文件
EDGES_FILE = SCRIPT_DIR / "chengdu_edges.txt"   # 边文件
OUTPUT_DIR = SCRIPT_DIR / "out"                 # 输出目录

CELL_SIZE_M = 50.0          # 网格边长（米）
HAS_HEADER = False          # 你的 txt 没有表头，所以这里必须是 False
PRINT_PROGRESS_EVERY = 200  # 每处理多少条原始边打印一次进度

# =========================
# 一般不用改
# =========================
EARTH_RADIUS_M = 6378137.0
EPS = 1e-12
SNAP_TOL_M = 1e-6


def sniff_delimiter(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8-sig", errors="ignore")
    lines = [x for x in text.splitlines()[:10] if x.strip()]
    if not lines:
        return r"\s+"
    sample = "\n".join(lines)
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    if ";" in sample:
        return ";"
    if "|" in sample:
        return "|"
    return r"\s+"


def read_first_three_columns(file_path: Path, has_header: bool):
    sep = sniff_delimiter(file_path)
    rows = []
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        if sep == r"\s+":
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if has_header and i == 0:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"文件 {file_path.name} 第 {i + 1} 行少于 3 列")
                rows.append(parts[:3])
        else:
            reader = csv.reader(f, delimiter=sep)
            for i, row in enumerate(reader):
                if not row:
                    continue
                if has_header and i == 0:
                    continue
                if len(row) < 3:
                    raise ValueError(f"文件 {file_path.name} 第 {i + 1} 行少于 3 列")
                rows.append(row[:3])
    return rows


def read_nodes(file_path: Path, has_header: bool):
    rows = read_first_three_columns(file_path, has_header)
    nodes = {}
    for row in rows:
        node_id = str(row[0]).strip()
        lon = float(row[1])
        lat = float(row[2])
        if node_id in nodes:
            raise ValueError(f"点文件存在重复点id: {node_id}")
        nodes[node_id] = (lon, lat)
    return nodes


def read_edges(file_path: Path, has_header: bool):
    rows = read_first_three_columns(file_path, has_header)
    edges = []
    edge_ids = set()
    for row in rows:
        edge_id = str(row[0]).strip()
        start_id = str(row[1]).strip()
        end_id = str(row[2]).strip()
        if edge_id in edge_ids:
            raise ValueError(f"边文件存在重复边id: {edge_id}")
        edge_ids.add(edge_id)
        edges.append((edge_id, start_id, end_id))
    return edges


class Projector:
    def __init__(self, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        self.cos_lat0 = max(math.cos(math.radians(lat0)), 1e-12)

    def lonlat_to_xy(self, lon: float, lat: float):
        x = EARTH_RADIUS_M * math.radians(lon - self.lon0) * self.cos_lat0
        y = EARTH_RADIUS_M * math.radians(lat - self.lat0)
        return x, y

    def xy_to_lonlat(self, x: float, y: float):
        lon = self.lon0 + math.degrees(x / (EARTH_RADIUS_M * self.cos_lat0))
        lat = self.lat0 + math.degrees(y / EARTH_RADIUS_M)
        return lon, lat


class NodeRegistry:
    """所有新边端点都从这里拿点id；同坐标自动去重，不会重复。"""

    def __init__(self, projector: Projector, snap_tol_m: float = SNAP_TOL_M):
        self.projector = projector
        self.snap_tol_m = snap_tol_m
        self.coord_key_to_id = {}
        self.rows = []
        self.next_id = 1

    def _key(self, x: float, y: float):
        return (
            int(round(x / self.snap_tol_m)),
            int(round(y / self.snap_tol_m)),
        )

    def get_or_create(self, x: float, y: float, source_kind: str, source_original_node_id=None):
        key = self._key(x, y)
        if key in self.coord_key_to_id:
            return self.coord_key_to_id[key]

        node_id = self.next_id
        self.next_id += 1
        lon, lat = self.projector.xy_to_lonlat(x, y)
        self.coord_key_to_id[key] = node_id
        self.rows.append(
            {
                "node_id": node_id,
                "lon": lon,
                "lat": lat,
                "source_kind": source_kind,
                "source_original_node_id": source_original_node_id,
            }
        )
        return node_id

    def get_row(self, node_id: int):
        return self.rows[node_id - 1]


def segment_length(x0, y0, x1, y1):
    return math.hypot(x1 - x0, y1 - y0)


def unique_sorted(values, tol=1e-10):
    out = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def build_grid(points_xy, cell_size_m):
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    origin_x = math.floor(min_x / cell_size_m) * cell_size_m
    origin_y = math.floor(min_y / cell_size_m) * cell_size_m
    max_aligned_x = math.ceil(max_x / cell_size_m) * cell_size_m
    max_aligned_y = math.ceil(max_y / cell_size_m) * cell_size_m

    nx = max(1, int(round((max_aligned_x - origin_x) / cell_size_m)))
    ny = max(1, int(round((max_aligned_y - origin_y) / cell_size_m)))

    return {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "cell_size_m": cell_size_m,
        "nx": nx,
        "ny": ny,
    }


def point_to_index(value, origin, cell_size, max_index):
    rel = (value - origin) / cell_size
    nearest = round(rel)
    if abs(rel - nearest) < 1e-10:
        rel -= 1e-10
    idx = math.floor(rel)
    if idx < 0:
        return 0
    if idx > max_index:
        return max_index
    return idx


def split_segment_by_grid(x0, y0, x1, y1, grid):
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
        raise ValueError("检测到零长度边")

    ts = [0.0, 1.0]
    c = grid["cell_size_m"]

    if abs(dx) > 1e-8:
        min_x = min(x0, x1)
        max_x = max(x0, x1)
        k_start = math.floor((min_x - grid["origin_x"]) / c)
        k_end = math.ceil((max_x - grid["origin_x"]) / c)
        for k in range(k_start, k_end + 1):
            xg = grid["origin_x"] + k * c
            if abs(x0 - xg) <= 1e-8 and abs(x1 - xg) <= 1e-8:
                continue
            t = (xg - x0) / dx
            if EPS < t < 1.0 - EPS:
                ts.append(t)

    if abs(dy) > 1e-8:
        min_y = min(y0, y1)
        max_y = max(y0, y1)
        k_start = math.floor((min_y - grid["origin_y"]) / c)
        k_end = math.ceil((max_y - grid["origin_y"]) / c)
        for k in range(k_start, k_end + 1):
            yg = grid["origin_y"] + k * c
            if abs(y0 - yg) <= 1e-8 and abs(y1 - yg) <= 1e-8:
                continue
            t = (yg - y0) / dy
            if EPS < t < 1.0 - EPS:
                ts.append(t)

    ts = unique_sorted(ts, tol=1e-12)
    pts = [(x0 + t * dx, y0 + t * dy) for t in ts]

    result = []
    for a, b in zip(pts[:-1], pts[1:]):
        if segment_length(a[0], a[1], b[0], b[1]) > 1e-8:
            result.append((a, b))
    return result


def write_csv(path: Path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    print("开始读取文件...")
    if not NODES_FILE.exists():
        raise FileNotFoundError(f"找不到点文件: {NODES_FILE}")
    if not EDGES_FILE.exists():
        raise FileNotFoundError(f"找不到边文件: {EDGES_FILE}")

    nodes = read_nodes(NODES_FILE, HAS_HEADER)
    edges = read_edges(EDGES_FILE, HAS_HEADER)

    for edge_id, s, t in edges:
        if s not in nodes:
            raise ValueError(f"边 {edge_id} 的起点 {s} 在点文件中不存在")
        if t not in nodes:
            raise ValueError(f"边 {edge_id} 的终点 {t} 在点文件中不存在")

    print(f"原始点数: {len(nodes)}")
    print(f"原始边数: {len(edges)}")

    lons = [v[0] for v in nodes.values()]
    lats = [v[1] for v in nodes.values()]
    projector = Projector(sum(lons) / len(lons), sum(lats) / len(lats))

    node_xy = {}
    for node_id, (lon, lat) in nodes.items():
        node_xy[node_id] = projector.lonlat_to_xy(lon, lat)

    grid = build_grid(list(node_xy.values()), CELL_SIZE_M)
    print(f"网格: {grid['nx']} 列 x {grid['ny']} 行 = {grid['nx'] * grid['ny']} 个格子")
    print(f"网格边长: {CELL_SIZE_M} 米")

    registry = NodeRegistry(projector)
    original_to_new_node_id = {}
    for old_node_id, (x, y) in node_xy.items():
        new_node_id = registry.get_or_create(x, y, "original_node", old_node_id)
        original_to_new_node_id[old_node_id] = new_node_id

    new_edges_rows = []
    # 每个 old_edge_id 对应一个列表；列表元素依次是：
    # [new_edge_id, subedge_start_node_id, subedge_end_node_id, length_ratio_of_old_edge]
    grouped_subedges = {}

    next_new_edge_id = 1
    total_edges = len(edges)

    print("开始切分边...")
    for i, (old_edge_id, old_start_id, old_end_id) in enumerate(edges, start=1):
        if i == 1 or i % PRINT_PROGRESS_EVERY == 0 or i == total_edges:
            pct = i * 100.0 / total_edges
            print(f"进度: {i}/{total_edges} ({pct:.2f}%)")

        x0, y0 = node_xy[old_start_id]
        x1, y1 = node_xy[old_end_id]
        total_len = segment_length(x0, y0, x1, y1)

        if total_len <= 1e-8:
            raise ValueError(f"检测到零长度原始边: {old_edge_id}")

        subsegments = split_segment_by_grid(x0, y0, x1, y1, grid)
        ratio_sum = 0.0
        subedge_list = []

        for part_index, ((ax, ay), (bx, by)) in enumerate(subsegments, start=1):
            part_len = segment_length(ax, ay, bx, by)
            ratio = part_len / total_len
            ratio_sum += ratio

            if part_index == 1 and abs(ax - x0) <= SNAP_TOL_M and abs(ay - y0) <= SNAP_TOL_M:
                start_new_node_id = original_to_new_node_id[old_start_id]
            else:
                start_new_node_id = registry.get_or_create(ax, ay, "grid_split_point")

            if part_index == len(subsegments) and abs(bx - x1) <= SNAP_TOL_M and abs(by - y1) <= SNAP_TOL_M:
                end_new_node_id = original_to_new_node_id[old_end_id]
            else:
                end_new_node_id = registry.get_or_create(bx, by, "grid_split_point")

            if start_new_node_id == end_new_node_id:
                continue

            new_edge_id = next_new_edge_id
            next_new_edge_id += 1

            new_edges_rows.append([
                new_edge_id,
                start_new_node_id,
                end_new_node_id,
            ])

            subedge_list.append([
                new_edge_id,
                start_new_node_id,
                end_new_node_id,
                round(ratio, 12),
            ])

        if abs(ratio_sum - 1.0) > 1e-8:
            raise ValueError(f"原始边 {old_edge_id} 切分后比例和不是 1，而是 {ratio_sum}")

        grouped_subedges[old_edge_id] = subedge_list

    # 所有新边的顶点都来自 registry；registry 自动去重，所以 new_nodes.csv 不会有重复点
    used_node_ids = set()
    for row in new_edges_rows:
        used_node_ids.add(row[1])
        used_node_ids.add(row[2])

    new_nodes_rows = []
    for row in registry.rows:
        node_id = row["node_id"]
        if node_id in used_node_ids:
            new_nodes_rows.append([
                node_id,
                round(row["lon"], 12),
                round(row["lat"], 12),
            ])

    new_nodes_rows.sort(key=lambda x: x[0])
    new_edges_rows.sort(key=lambda x: x[0])

    seen = set()
    for row in new_nodes_rows:
        if row[0] in seen:
            raise ValueError(f"新点文件出现重复点id: {row[0]}")
        seen.add(row[0])

    node_id_set = {row[0] for row in new_nodes_rows}
    for row in new_edges_rows:
        if row[1] not in node_id_set or row[2] not in node_id_set:
            raise ValueError("存在新边端点没有出现在 new_nodes.csv 中")
        if row[1] == row[2]:
            raise ValueError(f"存在自环新边: {row}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        OUTPUT_DIR / "new_nodes.csv",
        ["node_id", "lon", "lat"],
        new_nodes_rows,
    )

    write_csv(
        OUTPUT_DIR / "new_edges.csv",
        ["edge_id", "start_node_id", "end_node_id"],
        new_edges_rows,
    )

    # 每个 old_edge_id 只占一行，第二列存 JSON 列表字符串
    grouped_rows = []
    for old_edge_id, subedge_list in grouped_subedges.items():
        grouped_rows.append([
            old_edge_id,
            json.dumps(subedge_list, ensure_ascii=False),
        ])

    write_csv(
        OUTPUT_DIR / "subedges_grouped.csv",
        ["old_edge_id", "subedges"],
        grouped_rows,
    )

    summary = {
        "input_nodes": len(nodes),
        "input_edges": len(edges),
        "output_nodes": len(new_nodes_rows),
        "output_edges": len(new_edges_rows),
        "grouped_subedge_rows": len(grouped_rows),
        "grid_cols": grid["nx"],
        "grid_rows": grid["ny"],
        "grid_total_cells": grid["nx"] * grid["ny"],
        "cell_size_m": CELL_SIZE_M,
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\\n处理完成。")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"new_nodes.csv       : {len(new_nodes_rows)} 行")
    print(f"new_edges.csv       : {len(new_edges_rows)} 行")
    print(f"subedges_grouped.csv: {len(grouped_rows)} 行")
    print("说明：")
    print("1. new_nodes.csv 里的点就是所有新边的顶点，而且已去重，不会重复。")
    print("2. subedges_grouped.csv 每个 old_edge_id 只占一行。")
    print("3. subedges 列是 JSON 列表，列表元素依次是 [new_edge_id, start_node_id, end_node_id, length_ratio_of_old_edge]。")
    print("4. 原始老边不会出现在 new_edges.csv 中，只保留切分后的新边。")


if __name__ == "__main__":
    main()
