"""Untrusted third-party query server (edge + time overlap, NO fine filtering)."""
import argparse
from pathlib import Path
from time import perf_counter

from baseline.config import load_config, resolve_path
from baseline.cell_query import (
    CELL_COVER_QUERY_MODE,
    cell_cover_policy_from_config,
)
from baseline.persistence import load_index
from baseline.smt_mode import CellCoverSmtBaselineServerQuery, is_per_edge_smt_index
from baseline.vo_io import save_vo
from baseline.response_io import save_response_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    for name in ('min_lon', 'min_lat', 'max_lon', 'max_lat'):
        ap.add_argument('--' + name.replace('_','-'), dest=name, required=True, type=float)
    ap.add_argument('--start', required=True, type=int)
    ap.add_argument('--end', required=True, type=int)
    ap.add_argument('--out', default='smt_query_output')
    a = ap.parse_args()
    cfg = load_config(a.config)
    idx = load_index(resolve_path(cfg, cfg['index_file']))
    if not is_per_edge_smt_index(idx):
        raise TypeError('not an Edge SMT baseline index; run build_edge_smt_index.py')
    cell_policy = cell_cover_policy_from_config(cfg, idx.road)
    query_args = (a.min_lon,a.min_lat,a.max_lon,a.max_lat,a.start,a.end)
    cover = cell_policy.cover_rectangle(*query_args[:4])
    t0 = perf_counter()
    resp = CellCoverSmtBaselineServerQuery(
        idx, cell_policy, *query_args
    ).build()
    wall_elapsed = perf_counter() - t0
    edge_load = resp.stats.edge_data_load_time_s
    smt_load = resp.stats.smt_materialization_time_s
    elapsed = max(0.0, wall_elapsed - edge_load - smt_load)
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    vo_file = out / 'edge_smt_vo.bin'
    response_file = out / 'response.json'
    vo_size = save_vo(resp.vo, vo_file)
    q = {
        'min_lon':a.min_lon,'min_lat':a.min_lat,
        'max_lon':a.max_lon,'max_lat':a.max_lat,
        'start':a.start,'end':a.end,
        'spatial_query_mode':CELL_COVER_QUERY_MODE,
        'cover_cells':{
            'min_x':cover.min_cell_x,'min_y':cover.min_cell_y,
            'max_x':cover.max_cell_x,'max_y':cover.max_cell_y,
            'count':cover.cell_count,
        },
        'effective_bounds':list(cover.effective_bounds),
    }
    # ONLY real returned result IDs, no original IDs of helper/non-result entries.
    save_response_json(response_file, vo_file, q, resp.candidate_ids, resp.stats, elapsed)
    print(f'Server Query + Proof Generation Time: {elapsed:.9f} s')
    print(f'Query Wall Time: {wall_elapsed:.9f} s')
    print(f'Edge Data Load Time (excluded): {edge_load:.9f} s')
    print(f'SMT Materialization Time (excluded): {smt_load:.9f} s')
    print('spatial query mode:', CELL_COVER_QUERY_MODE)
    print('covering cells:', cover.cell_count)
    print('result trajectories:',len(resp.candidate_ids))
    print('independent per-edge SMT proofs:',len(resp.vo.smt_proofs))
    print('proven edges:',len({p.eid for p in resp.vo.smt_proofs}))
    print('SMT sibling hashes:',sum(len(p.siblings) for p in resp.vo.smt_proofs))
    print('helper/verification entries:',len(resp.vo.verification_set))
    print('VO size bytes:',vo_size)
    print('response:',response_file)

if __name__ == '__main__':
    main()
