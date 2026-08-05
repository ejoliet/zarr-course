"""Light-curve store benchmark: Zarr dense (object x epoch) vs Parquet long table.

Subset: N objects x T epochs. Full problem: 200e6 x 40e3.
Workloads:
  W1: write full dataset
  W2: random single-object full light curve, x50
  W3: single-epoch slice across all objects
  W4: full-scan reduction (per-object std over time)

Three lanes: zarr, raw parquet (hand-tuned = Iceberg best case), and real
Iceberg via pyiceberg (optional: `uv run --extra extras python lc_bench.py`).
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
# One object = T rows. Keep row groups object-aligned so a single-curve read
# touches 25 objects' worth of rows, not 500 (1M rows would).
OBJ_PER_ROW_GROUP = 25
ROW_GROUP = OBJ_PER_ROW_GROUP * T     # 50_000 rows
RNG = np.random.default_rng(0)
RESULTS = {}

REPS = 3        # reads only; single-run timings vary ~2x on a busy laptop

def timeit(label, fn, reps=1):
    """Median of `reps` runs. Writes use reps=1 (they rewrite the same path)."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    dt = sorted(ts)[len(ts) // 2]
    RESULTS[label] = round(dt, 3)
    spread = f"  (min {min(ts):.3f} max {max(ts):.3f})" if reps > 1 else ""
    print(f"{label:45s} {dt:8.3f} s{spread}")
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
timeit("zarr  W2 50 random light curves", zarr_curves, REPS)

timeit("zarr  W3 one epoch, all objects",
       lambda: float(zg["flux"][:, 1234].mean()), REPS)

timeit("zarr  W4 per-object std (full scan)",
       lambda: float(np.std(zg["flux"][:], axis=1).mean()), REPS)

# ============================ PARQUET (Iceberg data layout) ============================
# Long table sorted by object_id (read-optimized layout Iceberg compaction would produce)
LONG = pa.table({
    "object_id": np.repeat(np.arange(N, dtype="int64"), T),
    "epoch": np.tile(np.arange(T, dtype="int32"), N),
    "flux": flux.ravel(), "ferr": ferr.ravel(), "dqa": dqa.ravel(),
})

# Row-group granularity is the whole ballgame for W2: one object is T rows, so a
# 1M-row group holds 500 objects and a single-curve read must touch all of them.
# Three layouts, same bytes: coarse groups, object-grain groups (25 objects), and
# coarse groups + Parquet page index (page-level pruning inside the row group).
PQ_LAYOUTS = [
    ("pq1M",   dict(row_group_size=1_000_000)),
    ("pq50k",  dict(row_group_size=ROW_GROUP)),
    ("pq1Mpi", dict(row_group_size=1_000_000, write_page_index=True)),
]
for tag, kw in PQ_LAYOUTS:
    path = f"lc_long_{tag}.parquet"
    os.path.exists(path) and os.remove(path)
    timeit(f"{tag} W1 write (sorted by object)",
           lambda path=path, kw=kw: pq.write_table(
               LONG, path, compression="zstd",
               sorting_columns=[pq.SortingColumn(0)], **kw))
    ds = pads.dataset(path)

    def pq_curves(ds=ds):
        tot = 0.0
        for i in obj_ids:
            t = ds.to_table(filter=pc.field("object_id") == int(i),
                            columns=["flux", "ferr"])
            tot += float(pc.sum(t["flux"]).as_py()) + float(pc.sum(t["ferr"]).as_py())
        return tot
    timeit(f"{tag} W2 50 random light curves", pq_curves, REPS)

    timeit(f"{tag} W3 one epoch, all objects",
           lambda ds=ds: float(pc.mean(
               ds.to_table(filter=pc.field("epoch") == 1234,
                           columns=["flux"])["flux"]).as_py()), REPS)

    def pq_scan_std(ds=ds):
        # group-by std over N*T rows
        t = ds.to_table(columns=["object_id", "flux"])
        g = t.group_by("object_id").aggregate([("flux", "stddev")])
        return float(pc.mean(g["flux_stddev"]).as_py())
    timeit(f"{tag} W4 per-object std (full scan)", pq_scan_std, REPS)
    RESULTS[f"{tag}_bytes"] = os.path.getsize(path)

# ============================ REAL ICEBERG (pyiceberg) ============================
# Same bits, same sort, but written through a catalog. Differences that matter:
#  - row_group_size becomes a table property: write.parquet.row-group-limit
#    (rows, default 1_048_576). write.parquet.row-group-size-bytes is accepted
#    and then ignored by pyiceberg 0.11 ("not implemented" warning), so bytes
#    tuning does nothing -- set the row limit.
#  - pyiceberg does not sort for you: SortOrder is metadata only, and there is no
#    rewrite_data_files/compaction. Sort the Arrow table yourself or W2 dies.
#  - Iceberg has no unsigned types, so dqa (uint16) must widen to int32
try:
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.expressions import EqualTo
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.table.sorting import SortOrder, SortField
    from pyiceberg.schema import Schema
    from pyiceberg.types import NestedField, LongType, IntegerType, FloatType
except ImportError:
    print("\n(pyiceberg not installed; skipping Iceberg lane. "
          "uv run --extra extras python lc_bench.py)")
    ice = None
else:
    shutil.rmtree("ice_warehouse", ignore_errors=True)
    os.path.exists("iceberg_catalog.db") and os.remove("iceberg_catalog.db")
    os.makedirs("ice_warehouse")
    cat = SqlCatalog("lab", uri="sqlite:///iceberg_catalog.db",
                     warehouse=f"file://{os.path.abspath('ice_warehouse')}")
    cat.create_namespace_if_not_exists("lc")
    ICE_SCHEMA = Schema(
        NestedField(1, "object_id", LongType(), required=False),
        NestedField(2, "epoch", IntegerType(), required=False),
        NestedField(3, "flux", FloatType(), required=False),
        NestedField(4, "ferr", FloatType(), required=False),
        NestedField(5, "dqa", IntegerType(), required=False),
    )

    def ice_write():
        # LONG is already object-sorted by construction; pyiceberg won't sort
        tbl = LONG.set_column(4, "dqa", pc.cast(LONG["dqa"], pa.int32()))
        t = cat.create_table(
            "lc.curves", schema=ICE_SCHEMA,
            sort_order=SortOrder(SortField(source_id=1,
                                           transform=IdentityTransform())),
            properties={
                "write.parquet.compression-codec": "zstd",
                "write.parquet.row-group-limit": str(ROW_GROUP),
                "write.target-file-size-bytes": str(2 * 1024**3),   # one file
            })
        t.append(tbl)
        return t
    ice = timeit("ice   W1 write (catalog append)", ice_write)

    def ice_curves():
        tot = 0.0
        for i in obj_ids:
            t = ice.scan(row_filter=EqualTo("object_id", int(i)),
                         selected_fields=("flux", "ferr")).to_arrow()
            tot += float(pc.sum(t["flux"]).as_py()) + float(pc.sum(t["ferr"]).as_py())
        return tot
    timeit("ice   W2 50 random light curves", ice_curves, REPS)

    timeit("ice   W3 one epoch, all objects",
           lambda: float(pc.mean(ice.scan(row_filter=EqualTo("epoch", 1234),
                                          selected_fields=("flux",))
                                 .to_arrow()["flux"]).as_py()), REPS)

    def ice_scan_std():
        t = ice.scan(selected_fields=("object_id", "flux")).to_arrow()
        g = t.group_by("object_id").aggregate([("flux", "stddev")])
        return float(pc.mean(g["flux_stddev"]).as_py())
    timeit("ice   W4 per-object std (full scan)", ice_scan_std, REPS)

# ---------- storage footprint ----------
def du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs)

RESULTS["zarr_bytes"] = du("lc.zarr")
print(f"\nzarr size: {RESULTS['zarr_bytes']/1e6:.0f} MB   "
      + "   ".join(f"{tag}: {RESULTS[tag + '_bytes']/1e6:.0f} MB"
                   for tag, _ in PQ_LAYOUTS)
      + f"   raw dense: {(N*T*(4+4+2))/1e6:.0f} MB")
if ice is not None:
    RESULTS["ice_bytes"] = du("ice_warehouse")
    print(f"iceberg warehouse: {RESULTS['ice_bytes']/1e6:.0f} MB "
          f"(data + manifests + metadata)")
print(json.dumps(RESULTS, indent=1))
