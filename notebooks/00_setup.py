# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 00 — Setup & environment check
# Course spine and full narrative: see the repo `README.md`. Each notebook is idempotent:
# **Restart & Run All must pass** — that is the done-criterion for every module.
#
# Setup (from repo root):
# ```bash
# uv sync
# cp .env.example .env          # local MinIO creds — .env is gitignored
# brew install minio/stable/minio minio/stable/mc
# minio server ~/minio-data --console-address ":9001" &
# mc alias set local http://127.0.0.1:9000 minioadmin minioadmin
# mc mb -p local/zarr-lab
# ```

# %%
import zarr, xarray, dask, numpy, icechunk, pyarrow
for m in (zarr, xarray, dask, numpy, icechunk, pyarrow):
    print(f"{m.__name__:12s} {m.__version__}")
assert zarr.__version__.startswith("3"), "This course requires zarr-python 3.x"

# %% [markdown]
# ## Notebook hygiene rules (read once)
# - Every write cell uses `mode="w"` / `overwrite=True` → safe to re-run out of order.
# - Dask `Client`s and `h5py.File`s are closed in the last cell of each notebook.
# - Benchmarks live in `scripts/` and are run via `!` — kernel state pollutes timings.

# %%
import socket
try:
    socket.create_connection(("127.0.0.1", 9000), timeout=1).close()
    print("MinIO: up — notebooks 04+ fully runnable")
except OSError:
    print("MinIO: down — notebooks 01-03 still work; start MinIO before 04")
