# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 05 — Distributed compute: Dask + Zarr
# The pattern that maps 1:1 onto EKS workers. Watch the dashboard link while cells run.
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
from dask.distributed import Client, LocalCluster
client = Client(LocalCluster(n_workers=4, threads_per_worker=2))
client

# %% [markdown]
# ## Parallel write straight to object storage — every task writes its chunk, no locks

# %%
import dask.array as da, xarray as xr
cube = da.random.default_rng(3).normal(size=(100, 1024, 1024), chunks=(10, 256, 256)).astype("float32")
ds = xr.Dataset({"flux": (("time", "y", "x"), cube)})
target = ("s3://zarr-lab/big.zarr", SO) if MINIO_OK else ("../data/big.zarr", None)
ds.to_zarr(target[0], mode="w", storage_options=target[1])
print("wrote", target[0])

# %% [markdown]
# ## Parallel reduction — each worker reads only its own chunks

# %%
ds2 = xr.open_zarr(target[0], storage_options=target[1])
float(ds2.flux.std(dim="time").mean().compute())

# %% [markdown]
# ## Region writes — the Airflow-task pattern
# Each task owns a chunk-aligned slice of a pre-allocated store.
# **Correctness caveats (retries, partial failure) are the subject of notebook 12.**

# %%
import numpy as np
patch = xr.Dataset({"flux": (("time", "y", "x"), np.ones((10, 1024, 1024), "float32"))})
patch.to_zarr(target[0], region={"time": slice(0, 10),
              "y": slice(0, 1024), "x": slice(0, 1024)}, storage_options=target[1])
print("region [0:10] overwritten:", float(ds2.flux[0, 0, 0].compute()) == 1.0)

# %%
client.close()  # hygiene
