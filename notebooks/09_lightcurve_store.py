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
!cd ../scripts && uv run python lc_bench.py

# %% [markdown]
# Expected shape of results (measured during course development, local NVMe):
#
# | Workload | Zarr | Parquet | Ratio |
# |---|---|---|---|
# | write | 4.6 s | 11.9 s | 2.6x |
# | 50 random light curves | 0.84 s | 1.44 s | 1.7x |
# | one epoch, all objects | 0.078 s | 1.27 s | **16x** |
# | per-object std full scan | 3.5 s | 7.5 s | 2.2x |
#
# W3 is the tell: object-sorted Parquet scans every row group for an epoch predicate;
# fixing it needs a *second copy* sorted by epoch. Zarr serves both axes from one layout.

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
