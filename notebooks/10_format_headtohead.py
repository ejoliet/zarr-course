# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 10 — Head-to-head vs HDF5, and the memory story
# Full honest reading in README Module 10. Headlines: same codec + single thread ≈ parity
# locally; the architectural wins are (1) codecs travel in metadata vs HDF5 plugins,
# (2) no global lock -> parallel decompression, (3) cloud reads: computed key + one GET
# vs serial B-tree walk.

# %%
!cd ../scripts && uv run python fmt_bench.py

# %% [markdown]
# On a multi-core Mac expect Zarr scan time to drop with threads while HDF5 stays flat
# or degrades (h5py global lock). Record your crossover — that ratio is the per-node
# pipeline speedup.

# %% [markdown]
# ## Out-of-core, enforced
# The script also prints RSS growth during a full on-disk reduction (~0 MB: chunks
# stream and drop). Prove the guarantee by capping worker memory below data size:

# %%
import dask.array as da
from dask.distributed import Client, LocalCluster
client = Client(LocalCluster(n_workers=2, memory_limit="500MB"))
import pathlib
assert pathlib.Path("../scripts/cube.zarr").exists(), "run the fmt_bench cell above first"
z = da.from_zarr("../scripts/cube.zarr")          # 1 GB on disk > 2x500MB cap
print("mean of 1 GB array under 1 GB total worker RAM:", float(z.mean().compute()))
client.close()
