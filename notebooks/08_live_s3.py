# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 08 — Live AWS S3 (optional)
# Run only after 04–07 pass on MinIO. Empty `storage_options` = ambient AWS credentials.
#
# ```python
# ds.to_zarr("s3://<scratch-bucket>/zarr-lab/labeled.zarr", mode="w", storage_options={})
# ```
#
# **Checklist** (full ops checklist in README Module 8):
# 1. Bucket + compute co-located in `us-east-1` — cross-region kills the advantage.
# 2. Requests dominate cost for small chunks: shard first (06), chunks ≥ a few MB.
# 3. Add `crc32c` checksum codec — no integrity checking by default.
# 4. Clean up: `aws s3 rm --recursive` the lab prefix.

# %%
# Feel real-world scale for free: anonymous public Zarr (network + creds NOT required for MinIO work)
import xarray as xr
# example public store (Pangeo/NOAA catalogs list more):
# ds = xr.open_zarr("s3://mur-sst/zarr-v1", storage_options={"anon": True})
print("uncomment with a live connection")
