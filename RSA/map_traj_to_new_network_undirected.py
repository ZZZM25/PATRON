from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


# ============================================================
# 你通常只改这里
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# 原始边文件：无向边文件。每一行 edge_id start_node_id end_node_id
# 注意：这里认为 start-end 和 end-start 是同一条边。
ORIGINAL_EDGES_FILE = SCRIPT_DIR / "chengdu_edges.txt"

# 原始轨迹 JSON 文件
TRAJ_JSON_FILE = SCRIPT_DIR / "traj-10-1.json"

# 网格切分后的输出目录
OUTPUT_DIR = SCRIPT_DIR / "out"

# 网格切分脚本生成的分组子边文件
# 格式: old_edge_id, subedges
# subedges 每个元素为 [new_edge_id, start_node_id, end_node_id, ratio]
SUBEDGES_GROUPED_FILE = OUTPUT_DIR / "subedges_grouped.csv"

# 新路网文件
NEW_EDGES_FILE = OUTPUT_DIR / "new_edges.csv"
NEW_NODES_FILE = OUTPUT_DIR / "new_nodes.csv"

# 输出：最终映射后的轨迹文件
MAPPED_TRAJ_FILE = OUTPUT_DIR / "mapped_trajectories_grouped_undirected.csv"

# 输出：真正失败的轨迹边
FAILED_FILE = OUTPUT_DIR / "mapped_trajectories_failed.csv"

# 输出：反向使用记录，仅用于排查；可以删除
REVERSE_USED_FILE = OUTPUT_DIR / "reverse_mapping_used.csv"

# 原始边文件是否有表头
HAS_HEADER_ORIGINAL_EDGES = False

# 输出轨迹中每个元素是否包含 edge_id
# True  -> [edge_id, start_node_id, start_timestamp, end_node_id, end_timestamp]
# False -> [start_node_id, start_timestamp, end_node_id, end_timestamp]
INCLUDE_EDGE_ID = True

# 每处理多少条轨迹打印一次进度
PRINT_PROGRESS_EVERY = 100


# ============================================================
# 基础读取函数
# ============================================================

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


def read_original_edges(path: Path, has_header: bool):
    if not path.exists():
        raise FileNotFoundError(f"找不到原始边文件: {path}")

    rows = read_first_three_columns(path, has_header)
    edges = []
    for row in rows:
        old_edge_id = str(row[0]).strip()
        start_node_id = str(row[1]).strip()
        end_node_id = str(row[2]).strip()
        edges.append((old_edge_id, start_node_id, end_node_id))
    return edges


def canonical_pair(a, b):
    """
    无向边 key。
    816-817 和 817-816 会得到同一个 key。
    """
    a = str(a)
    b = str(b)

    # 尽量按数字排序；如果不是数字，按字符串排序
    try:
        ia = int(a)
        ib = int(b)
        return (str(min(ia, ib)), str(max(ia, ib)))
    except ValueError:
        return tuple(sorted([a, b]))


def build_undirected_pair_to_old_edges(original_edges):
    """
    建立无向端点对 -> old_edge_id 列表。
    """
    pair_to_edges = {}
    old_edge_direction = {}

    for old_edge_id, s, t in original_edges:
        key = canonical_pair(s, t)
        pair_to_edges.setdefault(key, []).append(str(old_edge_id))

        # 记录这个 old_edge_id 在原始边文件里的保存方向
        old_edge_direction[str(old_edge_id)] = (str(s), str(t))

    return pair_to_edges, old_edge_direction


def read_grouped_subedges(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}\n请先运行网格切分脚本，生成 out/subedges_grouped.csv"
        )

    result = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"old_edge_id", "subedges"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("subedges_grouped.csv 必须包含列: old_edge_id, subedges")

        for row in reader:
            old_edge_id = str(row["old_edge_id"]).strip()
            subedges = json.loads(row["subedges"])

            cleaned = []
            ratio_sum = 0.0
            for item in subedges:
                if len(item) < 4:
                    raise ValueError(f"old_edge_id={old_edge_id} 的 subedges 元素长度不足 4: {item}")

                new_edge_id = int(item[0])
                start_node_id = int(item[1])
                end_node_id = int(item[2])
                ratio = float(item[3])

                ratio_sum += ratio
                cleaned.append([new_edge_id, start_node_id, end_node_id, ratio])

            if cleaned and abs(ratio_sum - 1.0) > 1e-6:
                raise ValueError(f"old_edge_id={old_edge_id} 的子边比例和不是 1，而是 {ratio_sum}")

            result[old_edge_id] = cleaned

    return result


def read_new_edges(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到新边文件: {path}")

    result = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"edge_id", "start_node_id", "end_node_id"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"new_edges.csv 必须包含列: {required}")

        for row in reader:
            edge_id = int(row["edge_id"])
            s = int(row["start_node_id"])
            t = int(row["end_node_id"])
            result[edge_id] = (s, t)

    return result


def read_new_nodes(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到新点文件: {path}")

    node_ids = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "node_id" not in (reader.fieldnames or []):
            raise ValueError("new_nodes.csv 必须包含列: node_id")

        for row in reader:
            node_ids.add(int(row["node_id"]))

    return node_ids


def load_traj_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到轨迹 JSON 文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 轨迹映射逻辑
# ============================================================

def choose_old_edge_undirected(pair_to_edges, old_edge_to_subedges, traj_start, traj_end):
    """
    按无向边查 old_edge_id。

    返回:
    old_edge_id, issue

    如果一个无向端点对对应多个 old_edge_id，说明原始边文件仍然有无向重复边。
    这种情况下默认取第一个能在 subedges_grouped.csv 找到的 old_edge_id。
    """
    key = canonical_pair(traj_start, traj_end)
    candidates = pair_to_edges.get(key)

    if not candidates:
        return None, "old_edge_not_found_by_undirected_pair"

    candidates_with_subedges = [x for x in candidates if x in old_edge_to_subedges]

    if not candidates_with_subedges:
        return None, f"old_edge_candidates_have_no_subedges:{candidates}"

    if len(candidates_with_subedges) > 1:
        return candidates_with_subedges[0], f"duplicate_undirected_pair_use_first:{candidates_with_subedges}"

    return candidates_with_subedges[0], None


def need_reverse_subedges(old_edge_id, old_edge_direction, traj_start, traj_end):
    """
    判断轨迹方向是否与 old_edge_id 在原始边文件中的方向相反。

    old_edge_direction[old_edge_id] = (old_start, old_end)
    如果 traj_start == old_start 且 traj_end == old_end，则正向；
    如果 traj_start == old_end 且 traj_end == old_start，则反向。
    """
    old_start, old_end = old_edge_direction[old_edge_id]
    traj_start = str(traj_start)
    traj_end = str(traj_end)

    if traj_start == old_start and traj_end == old_end:
        return False

    if traj_start == old_end and traj_end == old_start:
        return True

    # 理论上不会到这里，因为前面已经通过无向 pair 找到了
    raise ValueError(
        f"方向判断异常: old_edge_id={old_edge_id}, "
        f"old=({old_start},{old_end}), traj=({traj_start},{traj_end})"
    )


def reverse_subedges(subedges):
    """
    无向边反向行驶时：
    - 不创建新的 edge_id
    - 使用同一个 new_edge_id
    - 子边顺序倒过来
    - 每条子边起终点交换
    - ratio 不变
    """
    out = []
    for new_edge_id, s, t, ratio in reversed(subedges):
        out.append([int(new_edge_id), int(t), int(s), float(ratio)])
    return out


def get_edge_start_time(points_on_edge):
    # 轨迹点格式: [timestamp, lat, lon]
    return int(points_on_edge[0][0])


def get_edge_end_time(edge_point_groups, edge_index):
    # 非最后一条边：结束时间取下一条边第一个点 timestamp
    if edge_index < len(edge_point_groups) - 1:
        return int(edge_point_groups[edge_index + 1][0][0])

    # 最后一条边：结束时间取当前边最后一个点 timestamp
    return int(edge_point_groups[edge_index][-1][0])


def make_time_boundaries(start_ts: int, end_ts: int, ratios):
    if end_ts < start_ts:
        raise ValueError(f"时间戳倒退: start_ts={start_ts}, end_ts={end_ts}")

    duration = end_ts - start_ts
    boundaries = [start_ts]

    cum = 0.0
    for ratio in ratios[:-1]:
        cum += float(ratio)
        boundaries.append(int(round(start_ts + duration * cum)))

    boundaries.append(end_ts)

    # 避免 round 造成时间戳回退
    for i in range(1, len(boundaries)):
        if boundaries[i] < boundaries[i - 1]:
            boundaries[i] = boundaries[i - 1]

    return boundaries


def validate_new_edge_undirected(new_edge_id, output_start_node_id, output_end_node_id, new_edges, new_nodes):
    """
    因为新路网边也按无向边理解，所以校验时只要求端点集合一致。
    new_edges.csv 里可能存的是 A->B；
    轨迹输出可能是 B->A；
    两者都合法。
    """
    if new_edge_id not in new_edges:
        raise ValueError(f"new_edge_id={new_edge_id} 不存在于 new_edges.csv")

    real_s, real_t = new_edges[new_edge_id]
    if {int(real_s), int(real_t)} != {int(output_start_node_id), int(output_end_node_id)}:
        raise ValueError(
            f"new_edge_id={new_edge_id} 端点不匹配: "
            f"输出=({output_start_node_id},{output_end_node_id}), "
            f"new_edges.csv=({real_s},{real_t})"
        )

    if int(output_start_node_id) not in new_nodes:
        raise ValueError(f"start_node_id={output_start_node_id} 不存在于 new_nodes.csv")

    if int(output_end_node_id) not in new_nodes:
        raise ValueError(f"end_node_id={output_end_node_id} 不存在于 new_nodes.csv")


def write_csv(path: Path, header, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def reason_prefix(reason):
    reason = str(reason)
    if ":" in reason:
        return reason.split(":", 1)[0]
    return reason


def main():
    print("开始读取文件...")

    original_edges = read_original_edges(ORIGINAL_EDGES_FILE, HAS_HEADER_ORIGINAL_EDGES)
    pair_to_edges, old_edge_direction = build_undirected_pair_to_old_edges(original_edges)

    old_edge_to_subedges = read_grouped_subedges(SUBEDGES_GROUPED_FILE)
    new_edges = read_new_edges(NEW_EDGES_FILE)
    new_nodes = read_new_nodes(NEW_NODES_FILE)
    traj_data = load_traj_json(TRAJ_JSON_FILE)

    print(f"原始边数: {len(original_edges)}")
    print(f"无向端点 pair 数: {len(pair_to_edges)}")
    print(f"可映射 old_edge 数: {len(old_edge_to_subedges)}")
    print(f"new_edges 数: {len(new_edges)}")
    print(f"new_nodes 数: {len(new_nodes)}")
    print(f"轨迹数: {len(traj_data)}")
    print("说明：本脚本按无向边处理，A-B 和 B-A 使用同一个 edge_id。")

    output_rows = []
    failed_rows = []
    reverse_used_rows = []

    direction_counter = Counter()
    issue_counter = Counter()

    print("开始映射轨迹...")
    total_traj = len(traj_data)

    for trajectory_id, traj_item in enumerate(traj_data):
        if not isinstance(traj_item, list) or len(traj_item) < 3:
            failed_rows.append([trajectory_id, "", "", "", "bad_trajectory_format"])
            output_rows.append([trajectory_id, json.dumps([], ensure_ascii=False)])
            continue

        edge_pairs = traj_item[1]
        edge_point_groups = traj_item[2]

        if len(edge_pairs) != len(edge_point_groups):
            failed_rows.append([
                trajectory_id,
                "",
                "",
                "",
                f"edge_pairs_count_{len(edge_pairs)}_not_equal_point_groups_count_{len(edge_point_groups)}",
            ])
            output_rows.append([trajectory_id, json.dumps([], ensure_ascii=False)])
            continue

        mapped_list = []

        for edge_index, pair in enumerate(edge_pairs):
            if not isinstance(pair, list) or len(pair) < 2:
                failed_rows.append([trajectory_id, edge_index, "", "", "bad_edge_pair_format"])
                continue

            traj_start = str(pair[0])
            traj_end = str(pair[1])
            points_on_edge = edge_point_groups[edge_index]

            if not points_on_edge:
                failed_rows.append([
                    trajectory_id,
                    edge_index,
                    traj_start,
                    traj_end,
                    "empty_points_on_edge",
                ])
                continue

            try:
                start_ts = get_edge_start_time(points_on_edge)
                end_ts = get_edge_end_time(edge_point_groups, edge_index)
            except Exception as e:
                failed_rows.append([
                    trajectory_id,
                    edge_index,
                    traj_start,
                    traj_end,
                    f"bad_timestamp:{e}",
                ])
                continue

            old_edge_id, issue = choose_old_edge_undirected(
                pair_to_edges,
                old_edge_to_subedges,
                traj_start,
                traj_end,
            )

            if old_edge_id is None:
                failed_rows.append([
                    trajectory_id,
                    edge_index,
                    traj_start,
                    traj_end,
                    issue,
                ])
                continue

            if issue is not None:
                issue_counter[reason_prefix(issue)] += 1

            try:
                reverse_needed = need_reverse_subedges(
                    old_edge_id,
                    old_edge_direction,
                    traj_start,
                    traj_end,
                )
            except Exception as e:
                failed_rows.append([
                    trajectory_id,
                    edge_index,
                    traj_start,
                    traj_end,
                    f"direction_error:{e}",
                ])
                continue

            subedges = old_edge_to_subedges[old_edge_id]

            if reverse_needed:
                direction_counter["reverse"] += 1
                subedges_to_use = reverse_subedges(subedges)
                reverse_used_rows.append([
                    trajectory_id,
                    edge_index,
                    traj_start,
                    traj_end,
                    old_edge_id,
                ])
            else:
                direction_counter["forward"] += 1
                subedges_to_use = subedges

            ratios = [x[3] for x in subedges_to_use]
            boundaries = make_time_boundaries(start_ts, end_ts, ratios)

            for k, sub in enumerate(subedges_to_use):
                new_edge_id = int(sub[0])
                new_start_node_id = int(sub[1])
                new_end_node_id = int(sub[2])

                sub_start_ts = int(boundaries[k])
                sub_end_ts = int(boundaries[k + 1])

                validate_new_edge_undirected(
                    new_edge_id,
                    new_start_node_id,
                    new_end_node_id,
                    new_edges,
                    new_nodes,
                )

                if INCLUDE_EDGE_ID:
                    mapped_list.append([
                        new_edge_id,
                        new_start_node_id,
                        sub_start_ts,
                        new_end_node_id,
                        sub_end_ts,
                    ])
                else:
                    mapped_list.append([
                        new_start_node_id,
                        sub_start_ts,
                        new_end_node_id,
                        sub_end_ts,
                    ])

        output_rows.append([
            trajectory_id,
            json.dumps(mapped_list, ensure_ascii=False),
        ])

        if (trajectory_id + 1) % PRINT_PROGRESS_EVERY == 0 or (trajectory_id + 1) == total_traj:
            pct = (trajectory_id + 1) * 100.0 / total_traj if total_traj else 100.0
            print(f"进度: {trajectory_id + 1}/{total_traj} ({pct:.2f}%)")

    write_csv(
        MAPPED_TRAJ_FILE,
        ["trajectory_id", "trajectory_data"],
        output_rows,
    )

    if failed_rows:
        write_csv(
            FAILED_FILE,
            ["trajectory_id", "edge_index", "old_start_node_id", "old_end_node_id", "reason"],
            failed_rows,
        )
    elif FAILED_FILE.exists():
        FAILED_FILE.unlink()

    if reverse_used_rows:
        write_csv(
            REVERSE_USED_FILE,
            ["trajectory_id", "edge_index", "traj_start_node_id", "traj_end_node_id", "used_old_edge_id"],
            reverse_used_rows,
        )
    elif REVERSE_USED_FILE.exists():
        REVERSE_USED_FILE.unlink()

    print("\n处理完成。")
    print(f"输出轨迹文件: {MAPPED_TRAJ_FILE}")
    print(f"输出轨迹数: {len(output_rows)}")

    if INCLUDE_EDGE_ID:
        print("trajectory_data 中每个元素格式:")
        print("[edge_id, start_node_id, start_timestamp, end_node_id, end_timestamp]")
    else:
        print("trajectory_data 中每个元素格式:")
        print("[start_node_id, start_timestamp, end_node_id, end_timestamp]")

    print("\n方向统计:")
    for k, v in direction_counter.items():
        print(f"  {k}: {v}")

    if issue_counter:
        print("\n额外情况统计:")
        for k, v in issue_counter.most_common(20):
            print(f"  {k}: {v}")

    print(f"\n反向使用次数: {len(reverse_used_rows)}")
    if reverse_used_rows:
        print(f"反向记录文件: {REVERSE_USED_FILE}")

    print(f"\n真正失败 failed 数: {len(failed_rows)}")
    if failed_rows:
        print(f"失败文件: {FAILED_FILE}")
        c = Counter(reason_prefix(r[-1]) for r in failed_rows)
        print("失败原因统计:")
        for k, v in c.most_common(20):
            print(f"  {k}: {v}")
    else:
        print("没有真正失败的轨迹边。")

    print("\n重要说明：")
    print("1. 本脚本没有创建反向补边文件。")
    print("2. 反向行驶时仍使用同一个 edge_id，只是输出里的 start/end 会按轨迹方向反过来。")
    print("3. 因为你的路网默认是无向边，所以这是正确逻辑。")


if __name__ == "__main__":
    main()
