# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 04 — Zarr on S3 (MinIO emulation)
# Same code, object-store backend. One chunk = one S3 GET.
# %% [markdown]
# **MinIO check** — this notebook needs MinIO running (see `00_setup`). The cell below
# skips S3 cells gracefully if it is not up.

# %%
import os, socket
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")
def minio_up():
    try:
        socket.create_connection(("127.0.0.1", 9000), timeout=1).close()
        return True
    except OSError:
        return False
MINIO_OK = minio_up()
SO = {"key": os.getenv("AWS_ACCESS_KEY_ID"), "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
      "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL")}} if MINIO_OK else None
print("MinIO:", "up" if MINIO_OK else "DOWN — S3 cells will be skipped")

# %%
import xarray as xr
if MINIO_OK:
    ds = xr.open_zarr("../data/labeled.zarr")
    ds.to_zarr("s3://zarr-lab/labeled.zarr", mode="w", storage_options=SO)
    ds_s3 = xr.open_zarr("s3://zarr-lab/labeled.zarr", storage_options=SO)
    print(float(ds_s3.flux.isel(time=0).mean().compute()))

# %% [markdown]
# ## Chunks are objects — see them, then feel the latency trade

# %%
if MINIO_OK:
    import s3fs, time
    fs = s3fs.S3FileSystem(**{k: v for k, v in SO.items()})
    print(*fs.ls("zarr-lab/labeled.zarr")[:6], sep="\n")
    t0 = time.perf_counter()
    float(ds_s3.flux.isel(time=0).mean().compute())
    print(f"one timestep from MinIO: {time.perf_counter()-t0:.2f} s (vs local disk in 03)")

# %% [markdown]
# ## Consolidated metadata — one GET to open stores with many arrays

# %%
if MINIO_OK:
    import zarr
    zarr.consolidate_metadata(zarr.storage.FsspecStore.from_url(
        "s3://zarr-lab/labeled.zarr", storage_options=SO))
    print("consolidated — xarray picks it up automatically")
