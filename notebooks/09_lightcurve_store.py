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
# | write | **3.62 s** | 7.78 s | 21.05 s | 9.40 s | 15.90 s |
# | 50 random light curves | 1.970 s | **0.859 s** | 1.677 s | 1.027 s | 2.489 s |
# | one epoch, all objects | **0.094 s** | 0.193 s | 0.319 s | 0.274 s | 0.590 s |
# | per-object std full scan | 0.476 s | 0.534 s | 0.553 s | 0.529 s | **0.461 s** |
# | size (400 MB raw) | **250 MB** | 326 MB | 437 MB | 326 MB | 437 MB |
#
# W3 is the tell: object-sorted Parquet scans every row group for an epoch predicate;
# fixing it needs a *second copy* sorted by epoch. Zarr serves both axes from one layout.
# Everything else here is warm-cache-flattered — Parquet wins W2 locally because there is
# no request latency. The MinIO cell below is where the ratios get their real shape.
# Row groups: 50k (25 objects) fetches 15x fewer bytes per curve, yet lost on every axis
# locally (+34% storage, 2.6x write, 2x slower W2) — it is a bet on bytes-over-the-wire.
# The bet pays off remotely, and only for point reads (see the MinIO cell below).

# %% [markdown]
# ## Same five lanes on MinIO — where the row-group bet actually settles
# `LC_BENCH_S3=<bucket>` moves Zarr, all three Parquet layouts and the Iceberg warehouse
# onto object storage (creds from `../.env`). Needs MinIO up — module 04.

# %%
!cd ../scripts && LC_BENCH_S3=zarr-lab uv run --extra extras python lc_bench.py

# %% [markdown]
# | Workload | Zarr | pq1M | pq50k | pq1Mpi | Iceberg |
# |---|---|---|---|---|---|
# | write | **6.20 s** | 7.35 s | 21.77 s | 8.06 s | 15.96 s |
# | 50 random light curves | 4.29 s | 4.08 s | **1.70 s** | 3.66 s | 3.28 s |
# | one epoch, all objects | **0.192 s** | 1.170 s | 0.916 s | 0.599 s | 1.053 s |
# | per-object std full scan | 1.174 s | 1.188 s | 1.394 s | **0.913 s** | 1.435 s |
#
# 50k row groups flip from worst to best on W2 (**2.4x** over 1M) — bytes fetched finally
# cost something. They still lose the full scan (W4 1.5x) and write by 3x; the page index
# is the best remote scan layout. Zarr's epoch-slice win *widens* to 3.1x. Zarr's poor W2
# is chunk geometry, not format: `chunks=(500, 500)` serves one 2,000-epoch curve from 4
# partial chunks covering 500 objects each (README exercise 4). The Parquet-vs-Parquet W3
# row is noise-dominated (pq1M ranged 0.741-1.192 s) — don't quote that ratio.
# Caveat: loopback MinIO is ~0.2 ms RTT vs 20-60 ms for real S3, so latency — the term
# favouring Zarr and small row groups most — is under-counted here.

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
