from __future__ import annotations

import csv,math
from pathlib import Path
SPATIAL_SIDES=[0.5, 1, 1.5, 2]
TEMPORAL_RANGES = [
    ("1h", 60 * 60),
    ("2h", 2 * 60 * 60),
    ("4h", 4 * 60 * 60),
    ("8h", 8 * 60 * 60),
    ("12h", 12 * 60 * 60),
]


def km_to_lat_degree(km): return km/111.32

def km_to_lon_degree(km,lat): return km/(111.32*math.cos(math.radians(lat)))

def make_square(lon,lat,side):
    h=side/2
    return lon-km_to_lon_degree(h,lat),lat-km_to_lat_degree(h),lon+km_to_lon_degree(h,lat),lat+km_to_lat_degree(h)


def load_setup(path,default_start=None):
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        row=next(csv.DictReader(f))
    lon=float(row["center_lon"]); lat=float(row["center_lat"])
    start=int(row.get("base_query_start") or default_start) if (row.get("base_query_start") or default_start) is not None else None
    return lon,lat,start


def write_matrix(path,rows,value_field):
    labels=[x[0] for x in TEMPORAL_RANGES]
    lookup={(r["spatial_side_km"],r["temporal_label"]):r[value_field] for r in rows}
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f); w.writerow(["spatial_side_km",*labels])
        for side in SPATIAL_SIDES: w.writerow([side,*[lookup[(side,l)] for l in labels]])
