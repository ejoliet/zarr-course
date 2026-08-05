# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 09 — Case study: 200M-object light-curve store (Zarr vs Iceberg)
# Full argument, scaling math, and verdict: **README Module 9**. This notebook runs the
# evidence. Key reframe: a light-curve table is dense 2D matrices per quantity
# (`flux[object, epoch]`) + a small per-object Iceberg catalog. Keys become indices —
# the long table stores object_id/epoch 8 trillion times; Zarr stores them zero times.

# %% [markdown]
# ## Run the benchmark (script, not cells — kernel state pollutes timing)

# %%
!cd ../scripts && uv run --extra extras python lc_bench.py

# %% [markdown]
# Five lanes: Zarr, three Parquet layouts (1M-row groups; 50k object-grain groups;
# 1M + page index), and real Iceberg via pyiceberg. Measured local NVMe, warm cache:
#
# | Workload | Zarr | pq1M | pq50k | pq1Mpi | Iceberg |
# |---|---|---|---|---|---|
# | write | **4.4 s** | 9.2 s | 23.7 s | 10.1 s | 19.4 s |
# | 50 random light curves | 2.03 s | **0.91 s** | 1.87 s | 1.17 s | 3.06 s |
# | one epoch, all objects | **0.083 s** | 0.223 s | 0.352 s | 0.235 s | 0.687 s |
# | per-object std full scan | 0.551 s | **0.515 s** | 0.609 s | 0.566 s | 0.574 s |
# | size (400 MB raw) | **250 MB** | 326 MB | 437 MB | 326 MB | 437 MB |
#
# W3 is the tell: object-sorted Parquet scans every row group for an epoch predicate;
# fixing it needs a *second copy* sorted by epoch. Zarr serves both axes from one layout.
# Everything else here is warm-cache-flattered — Parquet wins W2 locally because there is
# no request latency. Rerun on MinIO (README exercise 1) before trusting any ratio but W3.
# Row groups: 50k (25 objects) fetches 15x fewer bytes per curve, yet lost on every axis
# locally (+34% storage, 2.6x write, 2x slower W2) — it is a bet on bytes-over-the-wire,
# which only pays off on object storage. The page index is a wash locally too.

# %% [markdown]
# ## Real data: Multimodal Universe (streaming preview, no bulk download)
# TESS = dense-cadence positive test (real compression ratios!). PLAsTiCC = ragged
# negative control — find the NaN-fill fraction where the dense matrix loses.

# %%
RUN_MMU = False  # flip on with network access; downloads a small streaming preview
if RUN_MMU:
    from datasets import load_dataset
    import numpy as np
    tess = load_dataset("MultimodalUniverse/tess", split="train", streaming=True)
    ex = next(iter(tess.with_format("numpy")))
    print({k: getattr(v, "shape", type(v)) for k, v in ex.items()})

# %% [markdown]
# ## The write path & hybrid architecture (see README for the diagram)
# Iceberg wins ingest (epoch-append is cheap + ACID); Zarr wins reads. Run both:
# Iceberg hot table -> chunk-aligned compaction -> Zarr/Icechunk cold matrix.
# Write-safety mechanics of that compaction task: **notebook 12**.
