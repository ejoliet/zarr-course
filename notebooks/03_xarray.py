# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 03 — xarray on Zarr: labeled cubes
# How you actually consume Zarr: labeled dims, lazy open, label-based cutouts.

# %%
import xarray as xr, numpy as np, pandas as pd
ds = xr.Dataset(
    {"flux": (("time", "y", "x"),
              np.random.default_rng(2).normal(size=(30, 1024, 1024)).astype("float32"))},
    coords={"time": pd.date_range("2026-01-01", periods=30, freq="D"),
            "y": np.arange(1024), "x": np.arange(1024)})
ds.flux.attrs["units"] = "MJy/sr"
ds.to_zarr("../data/labeled.zarr", mode="w",
           encoding={"flux": {"chunks": (5, 256, 256)}})
print("written")

# %% [markdown]
# ## Lazy open — milliseconds at any scale (metadata/data separation)

# %%
import time
t0 = time.perf_counter(); ds2 = xr.open_zarr("../data/labeled.zarr")
print(f"open_zarr: {(time.perf_counter()-t0)*1000:.1f} ms"); ds2

# %% [markdown]
# ## Label-based cutout = server-free SODA primitive
# Slicing by label pulls only intersecting chunks.

# %%
cut = ds2.flux.sel(time="2026-01-15", y=slice(100, 400), x=slice(100, 400))
float(cut.mean().compute())

# %% [markdown]
# ## Append along time — the daily-pipeline pattern

# %%
ds_new = xr.Dataset(
    {"flux": (("time", "y", "x"), np.ones((5, 1024, 1024), "float32"))},
    coords={"time": pd.date_range("2026-01-31", periods=5, freq="D"),
            "y": np.arange(1024), "x": np.arange(1024)})
ds_new.to_zarr("../data/labeled.zarr", append_dim="time")
print("time length now:", xr.open_zarr("../data/labeled.zarr").sizes["time"])

# %%
# hygiene: quick visual
import matplotlib.pyplot as plt
ds2.flux.isel(time=0, y=slice(0, 256), x=slice(0, 256)).plot(); plt.show()
