"""Light-curve store benchmark: Zarr dense (object x epoch) vs Parquet long table.

Subset: N objects x T epochs. Full problem: 200e6 x 40e3.
Workloads:
  W1: write full dataset
  W2: random single-object full light curve, x50
  W3: single-epoch slice across all objects
  W4: full-scan reduction (per-object std over time)
"""
import time, json, shutil, os
import numpy as np
import zarr
from zarr.codecs import BloscCodec
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads
import pyarrow.compute as pc

N, T = 20_000, 2_000          # subset scale (laptop-safe)
RNG = np.random.default_rng(0)
RESULTS = {}

def timeit(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    RESULTS[label] = round(dt, 3)
    print(f"{label:45s} {dt:8.3f} s")
    return out

# ---------- synthetic photometry (same bits for both stores) ----------
flux = RNG.normal(1000, 50, size=(N, T)).astype("float32")
ferr = RNG.gamma(2.0, 5.0, size=(N, T)).astype("float32")
dqa  = (RNG.random((N, T)) < 0.02).astype("uint16")

# ============================ ZARR ============================
shutil.rmtree("lc.zarr", ignore_errors=True)

def zarr_write():
    root = zarr.open_group("lc.zarr", mode="w")
    kw = dict(
        shape=(N, T),
        shards=(5_000, T),          # 1 shard object per 5k objects, full time
        chunks=(500, 500),          # inner read granularity
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
    )
    root.create_array("flux", dtype="float32", **kw)[:] = flux
    root.create_array("ferr", dtype="float32", **kw)[:] = ferr
    root.create_array("dqa",  dtype="uint16",  **kw)[:] = dqa
timeit("zarr  W1 write", zarr_write)

zg = zarr.open_group("lc.zarr", mode="r")
obj_ids = RNG.integers(0, N, size=50)

def zarr_curves():
    tot = 0.0
    for i in obj_ids:
        tot += float(zg["flux"][i, :].sum()) + float(zg["ferr"][i, :].sum())
    return tot
timeit("zarr  W2 50 random light curves", zarr_curves)

timeit("zarr  W3 one epoch, all objects",
       lambda: float(zg["flux"][:, 1234].mean()))

timeit("zarr  W4 per-object std (full scan)",
       lambda: float(np.std(zg["flux"][:], axis=1).mean()))

# ============================ PARQUET (Iceberg data layout) ============================
# Long table sorted by object_id (read-optimized layout Iceberg compaction would produce)
shutil.rmtree("lc_parquet", ignore_errors=True)

def pq_write():
    oid = np.repeat(np.arange(N, dtype="int64"), T)
    ep  = np.tile(np.arange(T, dtype="int32"), N)
    tbl = pa.table({
        "object_id": oid, "epoch": ep,
        "flux": flux.ravel(), "ferr": ferr.ravel(), "dqa": dqa.ravel(),
    })
    pq.write_table(tbl, "lc_long.parquet",
                   row_group_size=1_000_000, compression="zstd",
                   sorting_columns=[pq.SortingColumn(0)])
timeit("pq    W1 write (sorted by object)", pq_write)

ds = pads.dataset("lc_long.parquet")

def pq_curves():
    tot = 0.0
    for i in obj_ids:
        t = ds.to_table(filter=pc.field("object_id") == int(i),
                        columns=["flux", "ferr"])
        tot += float(pc.sum(t["flux"]).as_py()) + float(pc.sum(t["ferr"]).as_py())
    return tot
timeit("pq    W2 50 random light curves", pq_curves)

timeit("pq    W3 one epoch, all objects",
       lambda: float(pc.mean(
           ds.to_table(filter=pc.field("epoch") == 1234,
                       columns=["flux"])["flux"]).as_py()))

def pq_scan_std():
    # group-by std over 40M rows
    t = ds.to_table(columns=["object_id", "flux"])
    g = t.group_by("object_id").aggregate([("flux", "stddev")])
    return float(pc.mean(g["flux_stddev"]).as_py())
timeit("pq    W4 per-object std (full scan)", pq_scan_std)

# ---------- storage footprint ----------
def du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs)

RESULTS["zarr_bytes"] = du("lc.zarr")
RESULTS["pq_bytes"] = du("lc_long.parquet")
print(f"\nzarr size: {RESULTS['zarr_bytes']/1e6:.0f} MB   "
      f"parquet size: {RESULTS['pq_bytes']/1e6:.0f} MB   "
      f"raw dense: {(N*T*(4+4+2))/1e6:.0f} MB")
print(json.dumps(RESULTS, indent=1))
