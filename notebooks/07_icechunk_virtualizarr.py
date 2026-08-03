# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 07 — Icechunk & VirtualiZarr: the array data lake
# **Icechunk** = transactions, branches, time travel over Zarr (Iceberg's role, for tensors).
# **VirtualiZarr** = read existing NetCDF/HDF5 *as* Zarr via chunk references, no rewrite.

# %%
import icechunk, xarray as xr, numpy as np, pandas as pd, shutil
shutil.rmtree("../data/repo.icechunk", ignore_errors=True)
storage = icechunk.local_filesystem_storage("../data/repo.icechunk")
repo = icechunk.Repository.create(storage)
session = repo.writable_session("main")
ds = xr.Dataset({"flux": (("time", "y", "x"), np.zeros((10, 256, 256), "float32"))},
                coords={"time": pd.date_range("2026-01-01", periods=10)})
ds.to_zarr(session.store, mode="w", consolidated=False)
snap1 = session.commit("initial L3 mosaic")
snap1

# %% [markdown]
# ## Second commit + time travel

# %%
session = repo.writable_session("main")
ds2 = xr.open_zarr(session.store, consolidated=False)
(ds2.flux + 1.0).to_dataset(name="flux").to_zarr(session.store, mode="r+", consolidated=False)
snap2 = session.commit("recalibrated with new darks")
for c in repo.ancestry(branch="main"):
    print(c.id, "-", c.message)

# %%
old = xr.open_zarr(repo.readonly_session(snapshot_id=snap1).store, consolidated=False)
new = xr.open_zarr(repo.readonly_session(branch="main").store, consolidated=False)
print("pinned snapshot mean:", float(old.flux.mean()), "| main mean:", float(new.flux.mean()))

# %% [markdown]
# Reprocessing campaign = branch, not copy. Pin analyses to snapshot IDs like git SHAs.
#
# ## VirtualiZarr — legacy HDF5/NetCDF as Zarr (references only)

# %%
# make a small legacy NetCDF, then virtualize it (VirtualiZarr 2.x API: parser + registry)
import pathlib
xr.Dataset({"t": ("x", np.arange(100.0))}).to_netcdf("../data/legacy.nc", engine="h5netcdf")
from obstore.store import LocalStore
from virtualizarr import open_virtual_dataset
from virtualizarr.parsers import HDFParser
try:
    from obspec_utils.registry import ObjectStoreRegistry
except ImportError:  # older virtualizarr
    from virtualizarr.registry import ObjectStoreRegistry
registry = ObjectStoreRegistry({"file://": LocalStore()})
url = pathlib.Path("../data/legacy.nc").resolve().as_uri()
vds = open_virtual_dataset(url, registry=registry, parser=HDFParser())
vds  # ManifestArray = chunk references only, no data copied

# %% [markdown]
# Persist those references into an Icechunk repo and a whole archive becomes one
# queryable cloud-native cube — no conversion. FITS support is experimental (kerchunk);
# uncompressed image HDUs are byte-range addressable, worth a spike before bulk converting.
#
# ⚠️ Icechunk stores are not plain Zarr directories — readers need the icechunk library.
# Plain Zarr + consolidated metadata remains the most interoperable public-serving choice.
