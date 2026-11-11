from __future__ import annotations

import math


class RoadNetwork:
    __slots__ = (
        "node_lon", "node_lat", "node_present",
        "edge_u", "edge_v", "edge_present", "pair_to_eids",
        "max_nid", "max_eid",
    )

    def __init__(self) -> None:
        self.node_lon: list[float] = []
        self.node_lat: list[float] = []
        self.node_present: list[bool] = []
        self.edge_u: list[int] = []
        self.edge_v: list[int] = []
        self.edge_present: list[bool] = []
        self.pair_to_eids: dict[tuple[int, int], list[int]] = {}
        self.max_nid = -1
        self.max_eid = -1

    @staticmethod
    def _canonical_pair(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    def load(self, node_file: str, edge_file: str) -> None:
        nodes: list[tuple[int, float, float]] = []
        with open(node_file, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                p = line.split()
                if len(p) < 3:
                    raise ValueError(f"node file line {line_no}: {line}")
                nid, lon, lat = int(p[0]), float(p[1]), float(p[2])
                nodes.append((nid, lon, lat))
        if not nodes:
            raise ValueError("empty node file")
        self.max_nid = max(n[0] for n in nodes)
        self.node_lon = [math.nan] * (self.max_nid + 1)
        self.node_lat = [math.nan] * (self.max_nid + 1)
        self.node_present = [False] * (self.max_nid + 1)
        for nid, lon, lat in nodes:
            if self.node_present[nid]:
                raise ValueError(f"duplicate node id {nid}")
            self.node_present[nid] = True
            self.node_lon[nid] = lon
            self.node_lat[nid] = lat

        edges: list[tuple[int, int, int]] = []
        with open(edge_file, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                p = line.split()
                if len(p) < 3:
                    raise ValueError(f"edge file line {line_no}: {line}")
                eid, u, v = int(p[0]), int(p[1]), int(p[2])
                if not self.node_exists(u) or not self.node_exists(v):
                    raise ValueError(f"edge {eid} references missing node {u},{v}")
                edges.append((eid, u, v))
        if not edges:
            raise ValueError("empty edge file")
        self.max_eid = max(e[0] for e in edges)
        self.edge_u = [-1] * (self.max_eid + 1)
        self.edge_v = [-1] * (self.max_eid + 1)
        self.edge_present = [False] * (self.max_eid + 1)
        self.pair_to_eids = {}
        for eid, u, v in edges:
            if self.edge_present[eid]:
                raise ValueError(f"duplicate eid {eid}")
            key = self._canonical_pair(u, v)
            self.edge_present[eid] = True
            self.edge_u[eid] = u
            self.edge_v[eid] = v
            self.pair_to_eids.setdefault(key, []).append(eid)

    def node_exists(self, nid: int) -> bool:
        return 0 <= nid < len(self.node_present) and self.node_present[nid]

    def edge_exists(self, eid: int) -> bool:
        return 0 <= eid < len(self.edge_present) and self.edge_present[eid]

    def get_edge_nodes(self, eid: int) -> tuple[int, int]:
        if not self.edge_exists(eid):
            raise KeyError(eid)
        return self.edge_u[eid], self.edge_v[eid]

    def resolve_edge(self, u: int, v: int, parallel_policy: str = "min") -> int:
        candidates = self.pair_to_eids.get(self._canonical_pair(u, v), [])
        if not candidates:
            raise KeyError(f"road network has no edge {u}->{v}")
        if len(candidates) == 1:
            return candidates[0]
        if parallel_policy == "min":
            return min(candidates)
        raise ValueError(f"ambiguous parallel edges for {u},{v}: {candidates}")

    def valid_node_ids(self) -> list[int]:
        return [i for i, ok in enumerate(self.node_present) if ok]

    def valid_edge_ids(self) -> list[int]:
        return [i for i, ok in enumerate(self.edge_present) if ok]
