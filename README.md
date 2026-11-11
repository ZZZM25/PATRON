# PATRON

Source code for the paper **"PATRON: Privacy-Preserving and Verifiable Trajectory Range Queries over Road Networks"**.

The repository contains our method (PATRON, in `KZG/`) and the two baselines used in the experiments (`RSA/` and `SMT/`).

## Repository structure

```
PATRON/
├── KZG/                  # PATRON (our method)
│   ├── preprocess/       # pipeline: split road edges at Geohash cell boundaries and map trajectories onto the split edges
│   ├── network/          # road-network reader and edge splitting
│   ├── trajectory/       # trajectory reader and edge entry/exit times
│   ├── geohash/          # binary Geohash grid and cell-size selection
│   ├── cell_index/       # sparse Merkle tree over Geohash cells; spatial query and verification
│   └── kzg/              # KZG commitments over time slots, index building, query-package generation, verification
├── RSA/                  # RSA-accumulator baseline: data preparation, index building, query/verification, experiment scripts
└── SMT/                  # sparse-Merkle-tree baseline (one SMT per edge, edge roots aggregated by a spatial tree)
    └── smt/              # SMT library code
```

## Requirements

- Python 3.10+
- `py_ecc` and `pyblst` (BLS12-381 operations for KZG)
- `gmpy2` (RSA baseline)

```bash
pip install py_ecc pyblst gmpy2
```

All other modules are from the Python standard library.

`KZG/kzg/_native.c` is an optional C extension (NTT, multi-scalar multiplication, polynomial aggregation) built against [blst](https://github.com/supranational/blst). If it is not compiled, the code automatically falls back to the pure-Python implementations.

## Data

We use three datasets. The data can be downloaded from the following websites:

- Xi'an and Chengdu: [DiDi GAIA Open Dataset](https://outreach.didichuxing.com/)
- Geolife (Beijing): [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=52367)
- Road networks: [OpenStreetMap](https://www.openstreetmap.org/)

Input format (after map matching):

- **Nodes** (`*_nodes.txt`): one node per line, `node_id lon lat`.
- **Edges** (`*_edges.txt`): one edge per line, `edge_id start_node end_node`.
- **Trajectories** (`*.json`): a list of trajectories; each trajectory is `[node_sequence, node_pairs, gps_groups, simplified_nodes]`, where each GPS point is `[timestamp, lat, lon]`.

## Running PATRON (`KZG/`)

**1. Preprocessing.** Split the road network into Geohash cells (about 50 m) and convert the trajectories to the split edges. This also builds the authenticated cell index.

```bash
cd KZG
PYTHONPATH=. python preprocess/main.py --nodes data/chengdu_nodes.txt --edges data/chengdu_edges.txt --trajectories data/trajectories.json --output output --cell-size 50
```

Outputs in `output/`: the split network (`new_nodes.txt`, `new_edges.txt`, ...), `processed_trajectories.csv`, and the cell index (`cell_index/manifest.json`, ...).

**2. Spatial query on the cell index** (optional check of the spatial part):

```bash
python -m cell_index.query_geo --index output/cell_index --min-lon 104.06 --max-lon 104.08 --min-lat 30.66 --max-lat 30.68
```

**3. Building, querying, and verifying.** These steps are provided as Python functions:

```python
from kzg.kzg import SRS
from kzg.builder import build_day
from kzg.batch_query_server import generate_batch_query_package
from kzg.batch_verifier import verify_batch_query_package

# Test SRS for simulation (single-party setup)
SRS.generate(domain_size=2048, max_multi_points=64).save("output/srs.bin")

# Data owner: build the commitments and the trusted anchor
build_day(nodes_path="output/new_nodes.txt", edges_path="output/new_edges.txt",
          trajectories_path="output/processed_trajectories.csv",
          cell_index_dir="output/cell_index", srs_path="output/srs.bin",
          output_database="output/day.sqlite", anchor_path="output/anchor.json")

# Service provider: answer a query (cell rectangle + time range) -> query package
# User: verify the package against the anchor
```

`kzg/period_builder.py` (`build_period`) builds one multi-day period instead of a single day.

## Running the baselines

**RSA** (run inside `RSA/`):

```bash
python prepare_rsg_dataset.py --nodes ... --edges ... --trajectories ... --dataset-date 2018-10-10 --output-dir rsg_50m_2018-10-10
python build_rsg_index.py --dataset-dir rsg_50m_2018-10-10 --date 2018-10-10
python run_rsg_50m_experiment_matrix.py --dataset-dir rsg_50m_2018-10-10 --date 2018-10-10
```

Other experiment scripts: `run_three_city_experiment.py` (query/verification/VO size on three datasets), `run_beijing_experiment.py` (Geolife), `run_build_scaling_experiment.py` (construction time and index size at 20%–100% of the data), and `run_update_experiment.py` (updates from a 60% base).

**SMT** (run inside `SMT/`): the scripts import the library in `SMT/smt/` as `baseline`, so rename `SMT/smt/` to `SMT/baseline/` first. The scripts read a JSON config file (keys such as `city`, `node_file`, `edge_file`, `trajectory_globs`, `index_file`, `trusted_root_file`, and `cell_partition` with `size_m: 50`).

```bash
python build_cell50_edge_smt_index.py --config config.json
python run_query_experiment.py --config config.json
```

`build_cell50_edge_smt_index.py` builds the index on the road network split into 50 m cells, as in our experiments; `build_edge_smt_index.py` builds it on the original road network.

Other experiment scripts: `run_beijing_experiment.py`, `run_build_scaling_experiment.py`, and `run_update_experiment.py`.
