"""Light-curve store benchmark: Zarr dense (object x epoch) vs Parquet long table.

Subset: N objects x T epochs. Full problem: 200e6 x 40e3.
Workloads:
  W1: write full dataset
  W2: random single-object full light curve, x50
  W3: single-epoch slice across all objects
  W4: full-scan reduction (per-object std over time)

Three lanes: zarr, raw parquet (hand-tuned = Iceberg best case), and real
Iceberg via pyiceberg (optional: `uv run --extra extras python lc_bench.py`).

Set LC_BENCH_S3=<bucket> to run every lane against MinIO/S3 instead of local
disk (creds from ../.env). That is the regime where row-group granularity
actually matters, because bytes and round-trips stop being free.
"""
import time, json, shutil, os
import numpy as np
import zarr
from zarr.codecs import BloscCodec
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as pads
import pyarrow.compute as pc

N = int(os.getenv("LC_BENCH_N", 20_000))      # subset scale (laptop-safe);
T = int(os.getenv("LC_BENCH_T", 2_000))       # shrink for smoke tests, grow for ex. 2
# One object = T rows. Keep row groups object-aligned so a single-curve read
# touches 25 objects' worth of rows, not 500 (1M rows would).
OBJ_PER_ROW_GROUP = 25
ROW_GROUP = OBJ_PER_ROW_GROUP * T     # 50_000 rows
RNG = np.random.default_rng(0)
RESULTS = {}

# ---------- backend: local disk, or MinIO/S3 when LC_BENCH_S3 is set ----------
S3_BUCKET = os.getenv("LC_BENCH_S3")
if S3_BUCKET:
    import s3fs
    from dotenv import load_dotenv
    from pyarrow.fs import S3FileSystem
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
    KEY, SECRET = os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"]
    ENDPOINT = os.environ["AWS_ENDPOINT_URL"]
    SO = {"key": KEY, "secret": SECRET,        # region_name only silences MinIO's
          "client_kwargs": {"endpoint_url": ENDPOINT,   # "unable to resolve region"
                            "region_name": "us-east-1"}}
    S3FS = s3fs.S3FileSystem(**SO)
    S3FS.mkdirs(S3_BUCKET, exist_ok=True)
    PQ_FS = S3FileSystem(access_key=KEY, secret_key=SECRET, region="us-east-1",
                         endpoint_override=ENDPOINT, scheme="http")
    PREFIX = f"{S3_BUCKET}/lc_bench/"          # every path below is PREFIX-relative
    print(f"backend: {ENDPOINT} bucket={S3_BUCKET}")
else:
    PQ_FS, PREFIX = None, ""
    print("backend: local filesystem")

def rm(path):
    if S3_BUCKET:
        S3FS.exists(PREFIX + path) and S3FS.rm(PREFIX + path, recursive=True)
    elif os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)

def zarr_store(name, **kw):
    if S3_BUCKET:
        return zarr.storage.FsspecStore.from_url(f"s3://{PREFIX}{name}",
                                                 storage_options=SO, **kw)
    return name

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
rm("lc.zarr")

def zarr_write():
    root = zarr.open_group(zarr_store("lc.zarr"), mode="w")
    kw = dict(
        shape=(N, T),
        shards=(min(5_000, N), T),        # 1 shard object per 5k objects, full time
        chunks=(min(500, N), min(500, T)),  # inner read granularity
        compressors=BloscCodec(cname="zstd", clevel=3, shuffle="shuffle"),
    )
    root.create_array("flux", dtype="float32", **kw)[:] = flux
    root.create_array("ferr", dtype="float32", **kw)[:] = ferr
    root.create_array("dqa",  dtype="uint16",  **kw)[:] = dqa
timeit("zarr  W1 write", zarr_write)

zg = zarr.open_group(zarr_store("lc.zarr", read_only=True), mode="r")
obj_ids = RNG.integers(0, N, size=50)
EPOCH = min(1234, T - 1)          # the epoch W3 slices

def zarr_curves():
    tot = 0.0
    for i in obj_ids:
        tot += float(zg["flux"][i, :].sum()) + float(zg["ferr"][i, :].sum())
    return tot
timeit("zarr  W2 50 random light curves", zarr_curves, REPS)

timeit("zarr  W3 one epoch, all objects",
       lambda: float(zg["flux"][:, EPOCH].mean()), REPS)

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
    rm(path)
    timeit(f"{tag} W1 write (sorted by object)",
           lambda path=path, kw=kw: pq.write_table(
               LONG, PREFIX + path, filesystem=PQ_FS, compression="zstd",
               sorting_columns=[pq.SortingColumn(0)], **kw))
    ds = pads.dataset(PREFIX + path, filesystem=PQ_FS)

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
               ds.to_table(filter=pc.field("epoch") == EPOCH,
                           columns=["flux"])["flux"]).as_py()), REPS)

    def pq_scan_std(ds=ds):
        # group-by std over N*T rows
        t = ds.to_table(columns=["object_id", "flux"])
        g = t.group_by("object_id").aggregate([("flux", "stddev")])
        return float(pc.mean(g["flux_stddev"]).as_py())
    timeit(f"{tag} W4 per-object std (full scan)", pq_scan_std, REPS)
    RESULTS[f"{tag}_bytes"] = (S3FS.du(PREFIX + path) if S3_BUCKET
                               else os.path.getsize(path))

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
    # catalog stays local sqlite either way; only the warehouse moves
    rm("ice_warehouse")
    os.path.exists("iceberg_catalog.db") and os.remove("iceberg_catalog.db")
    if S3_BUCKET:
        ice_props = {"warehouse": f"s3://{PREFIX}ice_warehouse",
                     "s3.endpoint": ENDPOINT, "s3.access-key-id": KEY,
                     "s3.secret-access-key": SECRET, "s3.region": "us-east-1"}
    else:
        os.makedirs("ice_warehouse")
        ice_props = {"warehouse": f"file://{os.path.abspath('ice_warehouse')}"}
    cat = SqlCatalog("lab", uri="sqlite:///iceberg_catalog.db", **ice_props)
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
           lambda: float(pc.mean(ice.scan(row_filter=EqualTo("epoch", EPOCH),
                                          selected_fields=("flux",))
                                 .to_arrow()["flux"]).as_py()), REPS)

    def ice_scan_std():
        t = ice.scan(selected_fields=("object_id", "flux")).to_arrow()
        g = t.group_by("object_id").aggregate([("flux", "stddev")])
        return float(pc.mean(g["flux_stddev"]).as_py())
    timeit("ice   W4 per-object std (full scan)", ice_scan_std, REPS)

# ---------- storage footprint ----------
def du(path):
    if S3_BUCKET:
        return S3FS.du(PREFIX + path)
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
