from __future__ import annotations


def normalize_rectangle(min_lon, min_lat, max_lon, max_lat):
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


def rectangle_overlap(a_min_lon, a_min_lat, a_max_lon, a_max_lat,
                      b_min_lon, b_min_lat, b_max_lon, b_max_lat) -> bool:
    return not (
        a_max_lon < b_min_lon or b_max_lon < a_min_lon or
        a_max_lat < b_min_lat or b_max_lat < a_min_lat
    )


def segment_intersects_rectangle(x1, y1, x2, y2, min_lon, min_lat, max_lon, max_lat) -> bool:
    min_lon, min_lat, max_lon, max_lat = normalize_rectangle(min_lon, min_lat, max_lon, max_lat)
    dx, dy = x2 - x1, y2 - y1
    p = (-dx, dx, -dy, dy)
    q = (x1 - min_lon, max_lon - x1, y1 - min_lat, max_lat - y1)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return False
            continue
        t = qi / pi
        if pi < 0.0:
            if t > u2:
                return False
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return False
            if t < u2:
                u2 = t
    return u1 <= u2


def edge_intersects_query(eid: int, road, query_min_lon, query_min_lat, query_max_lon, query_max_lat) -> bool:
    if not road.edge_exists(eid):
        return False
    query_min_lon, query_min_lat, query_max_lon, query_max_lat = normalize_rectangle(
        query_min_lon, query_min_lat, query_max_lon, query_max_lat
    )
    u, v = road.get_edge_nodes(eid)
    x1, y1 = road.node_lon[u], road.node_lat[u]
    x2, y2 = road.node_lon[v], road.node_lat[v]
    if not rectangle_overlap(
        min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2),
        query_min_lon, query_min_lat, query_max_lon, query_max_lat,
    ):
        return False
    return segment_intersects_rectangle(
        x1, y1, x2, y2,
        query_min_lon, query_min_lat, query_max_lon, query_max_lat,
    )
