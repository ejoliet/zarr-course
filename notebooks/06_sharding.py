# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 06 — Sharding: solving the millions-of-tiny-objects problem
# Zarr v3 shards pack many chunks into one storage object; inner chunks stay
# independently readable via byte ranges. Read granularity without object sprawl.

# %%
import zarr, numpy as np
z = zarr.create_array(store="../data/sharded.zarr", shape=(100, 2048, 2048),
                      shards=(10, 1024, 1024), chunks=(1, 256, 256),
                      dtype="float32", overwrite=True)
z[:10] = np.random.default_rng(4).normal(size=(10, 2048, 2048)).astype("float32")
z.info

# %%
!find ../data/sharded.zarr -type f | wc -l
!find ../data/cube.zarr -type f | wc -l   # unsharded, from notebook 01

# %% [markdown]
# For a Roman-scale mosaic this is 10^4 objects vs 10^7 (LIST cost, request cost).
# **Trade-off**: a shard must be rewritten wholesale — align shards to write units
# (e.g., one shard = one pipeline task's output). Full-scale geometry worked out in
# notebook 09 / README Module 9.
