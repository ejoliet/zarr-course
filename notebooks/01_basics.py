# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 01 — Core concepts: array, chunks, store
# Mental model: **Zarr = chunked N-D array in a key-value store; metadata is plain JSON;
# every chunk is an independent object.**

# %%
import zarr, numpy as np
z = zarr.create_array(store="../data/cube.zarr", shape=(100, 1024, 1024),
                      chunks=(10, 256, 256), dtype="float32", overwrite=True)
z[:] = np.random.default_rng(0).normal(size=z.shape).astype("float32")
z.info

# %% [markdown]
# The store is just files — inspect it with shell tools:

# %%
!cat ../data/cube.zarr/zarr.json
!ls ../data/cube.zarr/c/ | head -5
!du -sh ../data/cube.zarr

# %% [markdown]
# ## Exercise 1 — partial reads
# Reading one time slice touches only the chunks it intersects, not the whole store.
# Compare mentally with pulling plane 50 out of a FITS cube over HTTP.

# %%
import time
t0 = time.perf_counter(); plane = z[50]; dt = time.perf_counter() - t0
print(f"one 1024x1024 plane in {dt*1000:.0f} ms — touched {1024//256 * (1024//256)} chunks")

# %% [markdown]
# ## Exercise 2 — chunking is the access-pattern contract
# Re-chunk and count objects. Rule of thumb: chunk ≈ dominant access unit, 1–100 MB compressed.

# %%
z2 = zarr.create_array(store="../data/cube_bigchunk.zarr", shape=(100, 1024, 1024),
                       chunks=(100, 128, 128), dtype="float32", overwrite=True)
z2[:10] = 1.0
!find ../data/cube.zarr -type f | wc -l && find ../data/cube_bigchunk.zarr -type f | wc -l
