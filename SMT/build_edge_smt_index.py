"""Trusted data owner: build time-sorted SHA256^2 lists + per-edge SMT indexes."""
import argparse
from pathlib import Path
from time import perf_counter

from baseline.config import load_config, resolve_path, ensure_parent
from baseline.road_network import RoadNetwork
from baseline.dataset_builder import build_store_and_catalog
from baseline.persistence import save_index
from baseline.smt_mode import build_smt_baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = load_config(a.config)
    road = RoadNetwork()
    road.load(str(resolve_path(cfg, cfg['node_file'])), str(resolve_path(cfg, cfg['edge_file'])))
    store, _unused_catalog, source_files, ntr, nseg, insertion_seconds = build_store_and_catalog(
        road, cfg['trajectory_globs'], Path(cfg['_config_dir']))
    t0 = perf_counter()
    idx = build_smt_baseline(road, store, int(cfg.get('theta', 64)), source_files)
    extra_seconds = perf_counter() - t0
    index_file = resolve_path(cfg, cfg['index_file'])
    root_file = resolve_path(cfg, cfg['trusted_root_file'])
    n_bytes = save_index(idx, index_file)
    ensure_parent(root_file)
    root_file.write_text(idx.root_hash.hex() + '\n', encoding='ascii')
    print('mode: PER-EDGE-SMT (independent 256-level proofs; roots aggregated by spatial tree)')
    print('trajectories:', ntr, 'entries:', nseg, 'SMT leaves:', sum(len(x.values) for x in idx.edge_smts))
    print(f'Index Construction Time: {insertion_seconds + extra_seconds:.9f} s')
    print(f'Index Save Size: {n_bytes / 1024 / 1024:.3f} MiB')
    print('trusted root:', idx.root_hash.hex())
    print('index file:', index_file)
    print('root file:', root_file)

if __name__ == '__main__':
    main()
