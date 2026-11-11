from __future__ import annotations

from dataclasses import dataclass

NODE_LEAF = 1
NODE_INTERNAL = 2


@dataclass(slots=True)
class BinarySpatialTree:
    node_type: list[int]
    min_lon: list[float]
    min_lat: list[float]
    max_lon: list[float]
    max_lat: list[float]
    left_child: list[int]
    right_child: list[int]
    node_edges: list[tuple[int, ...]]  # leaf edges or internal cross_edges
    node_hash: list[bytes]
    edge_group_root: list[bytes]
    root_index: int
    theta: int


class BinarySpatialTreeBuilder:
    """Binary version of the spatial partition.

    Each edge is weighted by its number of distinct trajectories.  A vertex's
    split weight is the sum of the weights of its incident edges in the current
    recursive subspace.

    Two recursive children only. Edges crossing the split are retained as a plain
    cross_edges list at the current node. No MBR is stored for that list.
    """

    def __init__(self, road, theta: int = 64, edge_trajectory_counts=None):
        if theta < 2:
            raise ValueError("theta must be >=2")
        self.road = road
        self.theta = theta
        if edge_trajectory_counts is None:
            raise ValueError(
                "edge_trajectory_counts is required for trajectory-weighted "
                "spatial partitioning"
            )
        if len(edge_trajectory_counts) <= road.max_eid:
            raise ValueError("edge_trajectory_counts does not cover every edge ID")
        self.edge_trajectory_counts = edge_trajectory_counts
        self.node_type=[]; self.min_lon=[]; self.min_lat=[]; self.max_lon=[]; self.max_lat=[]
        self.left_child=[]; self.right_child=[]; self.node_edges=[]

    def _new_node(self, ntype, mbr, left=-1, right=-1, edges=()):
        idx=len(self.node_type)
        self.node_type.append(ntype)
        self.min_lon.append(mbr[0]); self.min_lat.append(mbr[1]); self.max_lon.append(mbr[2]); self.max_lat.append(mbr[3])
        self.left_child.append(left); self.right_child.append(right); self.node_edges.append(tuple(sorted(edges)))
        return idx

    def _mbr(self, vertices: list[int]):
        xs=[self.road.node_lon[v] for v in vertices]
        ys=[self.road.node_lat[v] for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)

    def _split_vertices(self, vertices: list[int], edges: list[int], mbr):
        width=mbr[2]-mbr[0]; height=mbr[3]-mbr[1]
        use_lon = width >= height
        weights={v:0 for v in vertices}
        for eid in edges:
            u,v=self.road.get_edge_nodes(eid)
            trajectory_count = int(self.edge_trajectory_counts[eid])
            if u in weights: weights[u] += trajectory_count
            if v in weights: weights[v] += trajectory_count
        ordered=sorted(vertices, key=lambda v: ((self.road.node_lon[v] if use_lon else self.road.node_lat[v]), v))
        if len(ordered)<2:
            return None
        total=sum(weights[v] for v in ordered)
        # Regions with no trajectories fall back to equal vertex count.
        if total == 0:
            cut=len(ordered)//2
        else:
            prefix=0; best=None
            for cut_candidate in range(1,len(ordered)):
                prefix += weights[ordered[cut_candidate-1]]
                diff=abs(total-2*prefix)
                candidate=(diff,cut_candidate)
                if best is None or candidate < best:
                    best=candidate
            cut=best[1]
        return ordered[:cut], ordered[cut:]

    def _build(self, vertices: list[int], edges: list[int]) -> int:
        mbr=self._mbr(vertices)
        if len(vertices) <= self.theta or len(vertices)<2:
            return self._new_node(NODE_LEAF,mbr,edges=edges)
        split=self._split_vertices(vertices,edges,mbr)
        if split is None:
            return self._new_node(NODE_LEAF,mbr,edges=edges)
        left_v,right_v=split
        left_set=set(left_v)
        left_e=[]; right_e=[]; cross=[]
        for eid in edges:
            u,v=self.road.get_edge_nodes(eid)
            ul=u in left_set; vl=v in left_set
            if ul and vl: left_e.append(eid)
            elif (not ul) and (not vl): right_e.append(eid)
            else: cross.append(eid)
        left_idx=self._build(left_v,left_e)
        right_idx=self._build(right_v,right_e)
        return self._new_node(NODE_INTERNAL,mbr,left_idx,right_idx,cross)

    def build(self) -> BinarySpatialTree:
        vertices=self.road.valid_node_ids()
        edges=self.road.valid_edge_ids()
        root=self._build(vertices,edges)
        n=len(self.node_type)
        return BinarySpatialTree(
            self.node_type,self.min_lon,self.min_lat,self.max_lon,self.max_lat,
            self.left_child,self.right_child,self.node_edges,
            [b"" for _ in range(n)], [b"" for _ in range(n)], root,self.theta,
        )
