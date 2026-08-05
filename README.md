# zarr-course

Hands-on Zarr v3 course: data lakes, distributed compute, and astronomy-scale workloads.
Local-first (macOS, `uv`, MinIO S3 emulation), benchmarked claims, Roman-SSC-flavored examples.

## Quickstart

```bash
git clone https://github.com/ejoliet/zarr-course && cd zarr-course
uv sync
cp .env.example .env                      # local MinIO creds only; .env is gitignored
brew install minio/stable/minio minio/stable/mc
minio server ~/minio-data --console-address ":9001" &
mc alias set local http://127.0.0.1:9000 minioadmin minioadmin && mc mb -p local/zarr-lab
uv run jupyter lab notebooks/
```

## Layout

```
notebooks/    13 modules as .ipynb, paired with git-diffable .py (jupytext percent)
scripts/      benchmarks + crash demo — scripts, not cells: kernel state pollutes timing
viewer/       module 13: no-build-step zarrita.js browser reader (open index.html)
data/         generated stores (gitignored)
```

## Notebook index

| # | Notebook | Module | Needs |
|---|---|---|---|
| 00 | `00_setup` | environment check, hygiene rules | — |
| 01 | `01_basics` | arrays, chunks, store-as-files | — |
| 02 | `02_groups_codecs` | hierarchy = your MEF, codec shoot-out | — |
| 03 | `03_xarray` | labeled cubes, lazy open, append | — |
| 04 | `04_s3_minio` | object-store backend, consolidated metadata | MinIO |
| 05 | `05_dask` | parallel write/read, region writes | MinIO (falls back local) |
| 06 | `06_sharding` | tiny-objects problem | — |
| 07 | `07_icechunk_virtualizarr` | transactions, time travel, virtualization | — |
| 08 | `08_live_s3` | optional live AWS + ops checklist | AWS |
| 09 | `09_lightcurve_store` | 200M-object case study vs Iceberg | runs `scripts/lc_bench.py` |
| 10 | `10_format_headtohead` | vs HDF5/FITS, out-of-core memory | runs `scripts/fmt_bench.py` |
| 11 | `11_wcs` | sky-coordinate cutouts, GWCS pattern | — |
| 12 | `12_write_safety` | Airflow retry corruption + Icechunk fix | runs `scripts/crash_task.py` |
| 13 | `viewer/index.html` | serving & browser visualization | MinIO (anonymous) |

Done-criterion per notebook: **Restart & Run All passes.**

The rest of this README is the course spine — the full narrative, measured benchmark
tables, scaling math, decision guides, and verdicts that the notebooks reference.

---


> Step-by-step, local-first course on Zarr v3 for data lakes and distributed compute.
> Target: macOS, `uv`, zarr-python 3.x. Local disk → MinIO (S3 emulation) → optional live S3.

**Audience**: data engineer already fluent in Parquet, FITS/ASDF, S3, Dask, Airflow.
**Time**: ~10–14 hours across 13 modules + browser viewer. Each module is self-contained and verifiable.

---

## Why Zarr (framing for your work)

- **Zarr = chunked, compressed N-D arrays in a key-value store** (disk, S3, memory, zip). Think "Parquet for tensors": Parquet owns tables; Zarr owns cubes (image stacks, spectral cubes, time × y × x grids).
- Every chunk is an independent object → massively parallel reads/writes from Dask/EKS workers, no single-file lock like HDF5/FITS.
- Metadata is plain JSON → inspectable with `cat`, diffable, no binary header parsing.
- Ecosystem on top: **xarray** (labeled dims), **Icechunk** (ACID + time travel), **VirtualiZarr/kerchunk** (expose existing FITS/HDF5/NetCDF as Zarr without rewriting).

**Roman/IPAC relevance**: L2/L3 image stacks and spectral cubes as cloud-native cutout-able stores; Dask-on-EKS parallel pipelines writing region-wise; Icechunk as versioned array lake next to Iceberg tables.

---

## Module 0 — Setup (macOS, uv)

**Goal**: reproducible env, running in 10 minutes.

```bash
mkdir zarr-course && cd zarr-course
uv init --python 3.12
uv add "zarr>=3" xarray dask[distributed] s3fs numcodecs matplotlib \
       netcdf4 h5netcdf virtualizarr icechunk rich
```

MinIO (local S3 emulation) — used from Module 4 on:

```bash
brew install minio/stable/minio minio/stable/mc
# or: docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
```

Local-only credentials file `.env` (gitignored — never commit):

```bash
# .env  (add to .gitignore now)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ENDPOINT_URL=http://127.0.0.1:9000
```

> ⚠️ Add `.env` and `*.zarr/` to `.gitignore` before anything else. Standing invariant: gitignore before secrets.

**Verify**:

```bash
uv run python -c "import zarr; print(zarr.__version__)"   # expect 3.x
```

**Docs**: https://zarr.readthedocs.io · https://zarr.dev/blog/zarr-python-3-release/

---

## Module 1 — Core concepts: array, chunks, store

**Goal**: create a Zarr array, then read its layout with plain filesystem tools.

```python
# m1_basics.py
import zarr, numpy as np

# A 3D "cube": 100 time slices of 2048x2048 float32
z = zarr.create_array(
    store="data/cube.zarr",
    shape=(100, 2048, 2048),
    chunks=(10, 512, 512),
    dtype="float32",
)
z[:] = np.random.default_rng(0).normal(size=z.shape).astype("float32")
print(z.info)
```

Now inspect the store as files — this is the key mental model:

```bash
uv run python m1_basics.py
cat data/cube.zarr/zarr.json        # metadata: shape, chunks, codecs — plain JSON
ls data/cube.zarr/c/ | head          # chunk objects, one per (t,y,x) chunk grid cell
du -sh data/cube.zarr
```

**Exercises**:

1. Read a single slice `z[50]` and note only 16 chunks (one time-plane row of the grid) are touched — not the whole file. Compare with what a FITS read of plane 50 over HTTP would require.
2. Change `chunks=(100, 128, 128)` and re-inspect. Count chunk files. Rule of thumb: chunk = your dominant access pattern; target 1–100 MB compressed per chunk.

**Docs**: Quickstart https://zarr.readthedocs.io/en/stable/quickstart.html · Chunk guide https://zarr.readthedocs.io/en/stable/user-guide/performance.html

---

## Module 2 — Groups, hierarchy, compression codecs

**Goal**: build a mission-style hierarchy and tune compression.

```python
# m2_groups.py
import zarr, numpy as np
from zarr.codecs import BloscCodec

root = zarr.open_group("data/mission.zarr", mode="w")
l2 = root.create_group("L2")
sci = l2.create_array(
    "sci", shape=(50, 4096, 4096), chunks=(1, 1024, 1024), dtype="float32",
    compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),
)
err = l2.create_array("err", shape=(50, 4096, 4096),
                      chunks=(1, 1024, 1024), dtype="float32")
sci.attrs.update({"unit": "MJy/sr", "instrument": "WFI-like", "level": "L2"})
sci[:] = np.random.default_rng(1).gamma(2.0, size=sci.shape).astype("float32")
print(root.tree())
```

**Exercises**:

1. Benchmark `zstd` clevel 1 vs 5 vs 9 and `lz4` on write time + `du -sh`. Note astronomy float noise compresses poorly — shuffle helps.
2. Attributes land in `zarr.json` as JSON — this is where FITS-header-like metadata lives. Compare mentally to ASDF: ASDF has richer schema/typed trees; Zarr attrs are flat JSON but store-native.

**Docs**: https://zarr.readthedocs.io/en/stable/user-guide/groups.html · codecs: https://zarr.readthedocs.io/en/stable/user-guide/arrays.html

---

## Module 3 — xarray on Zarr: labeled cubes

**Goal**: the way you will actually consume Zarr — labeled dims, lazy loading.

```python
# m3_xarray.py
import xarray as xr, numpy as np, pandas as pd

ds = xr.Dataset(
    {"flux": (("time", "y", "x"),
              np.random.default_rng(2).normal(size=(30, 1024, 1024)).astype("float32"))},
    coords={"time": pd.date_range("2026-01-01", periods=30, freq="D"),
            "y": np.arange(1024), "x": np.arange(1024)},
)
ds.flux.attrs["units"] = "MJy/sr"
ds.to_zarr("data/labeled.zarr", mode="w",
           encoding={"flux": {"chunks": (5, 256, 256)}})

# Lazy open — reads only metadata
ds2 = xr.open_zarr("data/labeled.zarr")
cutout = ds2.flux.sel(time="2026-01-15", y=slice(100, 400), x=slice(100, 400))
print(cutout.mean().compute())
```

**Exercises**:

1. Time the `open_zarr` call — it's milliseconds regardless of data size. That's the metadata/data separation.
2. Append along time: `ds_new.to_zarr("data/labeled.zarr", append_dim="time")`. This is the daily-pipeline-output pattern.
3. Server-free cutout service: slicing by label pulls only intersecting chunks. This is the Zarr answer to SODA-style cutouts — no service needed, just range reads.

**Docs**: https://docs.xarray.dev/en/stable/user-guide/io.html#zarr

---

## Module 4 — Zarr on S3 (MinIO emulation)

**Goal**: same code, object-store backend. Zero AWS cost.

Start MinIO and create a bucket:

```bash
mkdir -p ~/minio-data
minio server ~/minio-data --console-address ":9001" &   # console at http://127.0.0.1:9001
mc alias set local http://127.0.0.1:9000 minioadmin minioadmin
mc mb local/zarr-lab
```

```python
# m4_s3.py
import xarray as xr, os

storage_options = {
    "key": os.environ["AWS_ACCESS_KEY_ID"],
    "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
    "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL"]},
}

ds = xr.open_zarr("data/labeled.zarr")
ds.to_zarr("s3://zarr-lab/labeled.zarr", mode="w",
           storage_options=storage_options)

ds_s3 = xr.open_zarr("s3://zarr-lab/labeled.zarr",
                     storage_options=storage_options)
print(ds_s3.flux.isel(time=0).mean().compute())
```

Run with the env loaded: `set -a; source .env; set +a; uv run python m4_s3.py`

**Exercises**:

1. `mc ls -r local/zarr-lab/labeled.zarr/ | head` — see chunks as objects. One chunk = one S3 GET. Chunk size now directly controls request count and latency.
2. Time a single-timestep read from MinIO vs local disk. Then re-chunk to `(1, 1024, 1024)` and repeat — fewer, larger GETs usually win on object stores.
3. Consolidated metadata: for stores with many arrays, one metadata object avoids N small GETs on open. `zarr.consolidate_metadata(store)`; xarray reads it automatically.

> 💡 LocalStack also works (you already use it in the Airflow sandbox) — same `endpoint_url` pattern on port 4566. MinIO is lighter for pure-S3 work.

**Docs**: s3fs endpoint config https://s3fs.readthedocs.io/en/latest/ · MinIO https://min.io/docs/minio/macos/

---

## Module 5 — Distributed compute: Dask + Zarr

**Goal**: parallel read/compute/write — the pattern that maps to EKS workers.

```python
# m5_dask.py
import xarray as xr, dask.array as da, os
from dask.distributed import Client, LocalCluster

if __name__ == "__main__":
    client = Client(LocalCluster(n_workers=4, threads_per_worker=2))
    print(client.dashboard_link)   # watch it: http://127.0.0.1:8787

    so = {"key": os.environ["AWS_ACCESS_KEY_ID"],
          "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
          "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL"]}}

    # 1) Parallel write of a big synthetic cube straight to MinIO
    cube = da.random.default_rng(3).normal(
        size=(200, 2048, 2048), chunks=(10, 512, 512)).astype("float32")
    ds = xr.Dataset({"flux": (("time", "y", "x"), cube)})
    ds.to_zarr("s3://zarr-lab/big.zarr", mode="w", storage_options=so)

    # 2) Parallel reduction — each worker reads its own chunks, no coordination
    ds2 = xr.open_zarr("s3://zarr-lab/big.zarr", storage_options=so)
    print(ds2.flux.std(dim="time").mean().compute())
```

**Exercises**:

1. Watch the dashboard during the write: every task writes its chunk independently. This is why Zarr beats HDF5 for concurrent pipeline output — no file lock, no MPI-IO.
2. **Region writes** (the Airflow-task pattern): each task writes its slice of a pre-allocated store with `ds.to_zarr(..., region={"time": slice(i, i+10)})`. Sketch how N KubernetesPodOperator tasks could each own a time region of one store.
3. Scale thought experiment: same code, swap `LocalCluster` for `KubeCluster` on EKS. Only the client setup changes.

**Docs**: https://docs.dask.org/en/stable/array-best-practices.html · region writes https://docs.xarray.dev/en/stable/user-guide/io.html#appending-to-existing-zarr-stores

---

## Module 6 — Sharding (Zarr v3 feature)

**Goal**: solve the "millions of tiny objects" problem.

Small chunks are great for cutouts but terrible for object-store file counts. Zarr v3 sharding packs many chunks into one storage object while keeping them independently readable via byte ranges.

```python
# m6_sharding.py
import zarr, numpy as np

z = zarr.create_array(
    store="data/sharded.zarr",
    shape=(100, 4096, 4096),
    shards=(10, 2048, 2048),   # one object holds 4x4x... inner chunks
    chunks=(1, 512, 512),      # read granularity stays fine-grained
    dtype="float32",
)
z[:10] = np.random.default_rng(4).normal(size=(10, 4096, 4096)).astype("float32")
```

**Exercises**:

1. Count objects: `find data/sharded.zarr -type f | wc -l` vs the unsharded equivalent. For a Roman-scale mosaic this is the difference between 10^4 and 10^7 S3 objects (LIST cost, request cost, small-object overhead).
2. Trade-off to note: shards are read-efficient but a shard must be rewritten wholesale — pick shards aligned to write units (e.g., one shard = one pipeline task's output).

**Docs**: https://zarr.readthedocs.io/en/stable/user-guide/performance.html#sharding

---

## Module 7 — Icechunk & VirtualiZarr: the array data lake

**Goal**: version control for arrays, and virtualizing legacy files.

**Icechunk** = transactional layer over Zarr: commits, branches, tags, time travel — Iceberg's role, for tensors.

```python
# m7_icechunk.py
import icechunk, xarray as xr, numpy as np, pandas as pd

storage = icechunk.local_filesystem_storage("data/repo.icechunk")
repo = icechunk.Repository.create(storage)

session = repo.writable_session("main")
ds = xr.Dataset(
    {"flux": (("time", "y", "x"),
              np.zeros((10, 256, 256), dtype="float32"))},
    coords={"time": pd.date_range("2026-01-01", periods=10)},
)
ds.to_zarr(session.store, mode="w", consolidated=False)
snap1 = session.commit("initial L3 mosaic")

# Second commit: reprocessing
session = repo.writable_session("main")
ds2 = xr.open_zarr(session.store, consolidated=False)
# ... modify and write ...
# session.commit("recalibrated with new darks")

print(list(repo.ancestry(branch="main")))   # commit history — time travel via snapshot IDs
```

**VirtualiZarr** = read existing NetCDF/HDF5 (and via plugins, TIFF) *as* Zarr without converting — chunk references only:

```python
# m7_virtual.py — virtualize a NetCDF file
from virtualizarr import open_virtual_dataset
vds = open_virtual_dataset("some_legacy_file.nc")
# Persist references into an Icechunk repo → query the archive as one cloud-native cube
```

**Exercises**:

1. Make three commits to the Icechunk repo, then open the dataset at an old snapshot. This is reproducibility for pipeline reruns: pin analysis to a snapshot ID like a git SHA.
2. Map concepts: Iceberg snapshot ↔ Icechunk snapshot; Iceberg manifest ↔ chunk manifest. Decide where each fits: Iceberg for catalogs/tables, Icechunk for cubes — side by side in the same bucket.
3. Note for FITS: kerchunk/VirtualiZarr have experimental FITS reference support; uncompressed FITS image HDUs are byte-range-addressable so virtualization is feasible. Worth a spike for IRSA holdings before any bulk conversion.

> ⚠️ Icechunk stores are not plain Zarr directories — readers need the icechunk library. Plain Zarr + consolidated metadata stays the most interoperable choice for public serving.

**Docs**: https://icechunk.io · https://virtualizarr.readthedocs.io · kerchunk https://fsspec.github.io/kerchunk/

---

## Module 8 (optional) — Live AWS S3

**Goal**: validate on real S3. Only after Modules 4–7 pass on MinIO.

Changes required — remove the endpoint override, use real credentials (Kion/CloudTamer session as usual):

```python
so = {}   # empty: s3fs picks up ambient AWS credentials + region
ds.to_zarr("s3://<your-scratch-bucket>/zarr-lab/labeled.zarr",
           mode="w", storage_options=so)
```

**Checklist**:

1. Bucket and compute in `us-east-1` (co-locate; cross-region kills Zarr's advantage).
2. Read a public reference dataset for scale feel, e.g. anonymous access to a Pangeo/NOAA public Zarr store (`storage_options={"anon": True}`).
3. Cost sanity: requests dominate for small chunks. Sharding (Module 6) + chunks ≥ a few MB before pushing volume.
4. Clean up: `aws s3 rm --recursive` the lab prefix when done.

**Production ops checklist** (system-engineer gap — verify before any real deployment):

- Integrity: no checksums by default — add the `crc32c` checksum codec to every array; consider bucket-level checksum verification on PUT.
- Cost model: GET/PUT request counts dominate for small chunks; shard (Module 6) and lifecycle cold shards to Infrequent Access — never Glacier for anything range-read.
- Throughput: S3 rate limits are per prefix (~5,500 GET/s); shard-file naming already spreads prefixes, but verify under Dask-wide fan-out.
- Access: IAM prefix policies per store; Icechunk repos need list+get+put on their prefix only.
- Backup: a Zarr store is "just objects" — bucket replication works, but replicate Icechunk repos atomically at snapshot boundaries, not mid-commit.

---

## Module 9 — Case study: 200M-object light-curve store (Zarr vs Iceberg)

**Goal**: decide, with evidence, how to store 200M objects × 40,000 epochs × ~20 columns — and whether Zarr earns its place.

### The reframe that decides everything

A light-curve table is not really a table. It is a **dense 2D matrix per physical quantity**:

```
flux[object, epoch]     float32   200e6 × 40e3
ferr[object, epoch]     float32
dqa [object, epoch]     uint16
time[epoch]             float64   (shared axis, 40e3 values — stored ONCE)
```

Split the 20 columns by shape:

| Column kind | Examples | Store |
|---|---|---|
| Per-(object, epoch) | flux, ferr, dqa, background, x/y shift | **Zarr matrices** |
| Per-object scalars | object_id, ra, dec, class | **Iceberg catalog table** |
| Per-epoch scalars | mjd, exposure_id, zeropoint | small Zarr 1D arrays or Iceberg |

The bridge is one Iceberg table mapping `object_id → row_index`. In the long-table layout, `object_id` and `epoch` are repeated **8 trillion times** (8×10^12 rows × 12 bytes of keys ≈ 96 TB before encoding). In Zarr they are array indices — stored zero times.

> 💡 Precondition: this works because survey cadence is dense and shared (GBTDS-style: same fields, same epochs). If cadence is ragged per object, build one matrix per field with its own epoch axis — not one global matrix. Fully irregular sparse photometry → stay in Parquet/Iceberg.

### Subset benchmark (runnable)

Script: `lc_bench.py` (shipped with this course). Subset 20,000 objects × 2,000 epochs, three quantities, same bits into both layouts:

- **Zarr**: sharded arrays, `shards=(5000, T)`, inner `chunks=(500, 500)`, zstd
- **Parquet**, three layouts of the same object-sorted long table, zstd — the read-optimized physical layout Iceberg compaction produces (Iceberg's read path *is* Parquet; it adds catalog/ACID, not speed): `pq1M` = 1M-row groups, `pq50k` = 50k-row groups (object-grain: one object is T=2,000 rows, so 25 objects per group), `pq1Mpi` = 1M-row groups **plus the Parquet page index** (`write_page_index=True`)
- **Iceberg**, for real, via `pyiceberg` + a local SQLite catalog — same bits, written through a catalog

```bash
uv run --extra extras python lc_bench.py   # drop --extra extras to skip the Iceberg lane
```

Measured (local NVMe, warm page cache, median of 3 reads, this machine — rerun on yours):

| Workload | Zarr | pq1M | pq50k | pq1Mpi | Iceberg | Zarr vs best table |
|---|---|---|---|---|---|---|
| W1 write full dataset | **4.4 s** | 9.2 s | 23.7 s | 10.1 s | 19.4 s | 2.1× |
| W2 50 random full light curves | 2.03 s | **0.91 s** | 1.87 s | 1.17 s | 3.06 s | 0.45× (Parquet wins) |
| W3 one epoch across all objects | **0.083 s** | 0.223 s | 0.352 s | 0.235 s | 0.687 s | **2.7×** |
| W4 per-object std (full scan) | 0.551 s | **0.515 s** | 0.609 s | 0.566 s | 0.574 s | ~1× |
| Storage (400 MB raw) | **250 MB** | 326 MB | 437 MB | 326 MB | 437 MB | 1.3× |

> ⚠️ These replaced an earlier single-shot run that showed Zarr winning W2 (0.84 s vs 1.44 s) and a 16× W3 gap. Single runs on a busy laptop vary ~2×, which is why the script now takes a median of 3 — and warm-cache local NVMe is the *worst* case for the Zarr argument: no request latency, so Parquet's point lookups look great and only W3's structural penalty survives. Nothing but W3, W1 and storage is a robust local signal; the ratios that transfer to 200M objects are the ones measured on object storage (exercise 1).

### Why the numbers look like this

1. **W3 is the tell, and it is the only structural read gap.** Parquet is sorted by object; an epoch predicate matches rows in *every* row group → full-column scan. Zarr computes chunk coordinates from the index and reads exactly one chunk-column. To fix W3 in Parquet you need a *second copy* sorted by epoch. **Zarr serves both axes from one layout; Parquet must pick one.**
2. **Row-group granularity: measure it, don't reason about it.** A 1M-row group holds 500 objects, so one light curve decompresses ~8 MB of `flux`+`ferr` to return 16 KB. Object-grain 50k-row groups (25 objects, 0.53 MB/group) cut that 15× — and lost on *every* axis here: W2 1.87 s vs 0.91 s, +34% storage (437 vs 326 MB — zstd gets a 20× smaller compression context and dictionaries reset per group), 2.6× write time. Warm page cache makes bytes-read nearly free, so all that is left is per-scan planning over 800 row groups instead of 40. Writing the Parquet page index (`pq1Mpi`) is a wash locally too: same size, +10% write, no read win. **Small row groups are a bet on bytes-over-the-wire being the bottleneck** (cold S3, high latency) — real for a 200M-object store on object storage, invisible on a laptop. Find your crossover with exercise 1 before committing a layout.
3. **W2 is where sorted Parquet genuinely wins here** — row-group stats are very good at point lookups, and Zarr pays for reading `500×500` chunks to serve one row of 2,000. On S3 the balance shifts: Parquet needs footer + page-index + data-page round-trips per file touched; Zarr needs one computed-key range GET per chunk, no metadata reads. Chunk the object axis finer (`chunks=(50, 2_000)`) if single-curve serving is the primary workload.
4. **Keys are free.** The 76 MB Parquet spends on encoded `object_id`/`epoch` columns, Zarr spends on nothing.
5. **W1 is Zarr's clearest win** (2.1× the fastest table lane, 5.4× the 50k one) and it is the one that scales: Zarr writes are embarrassingly parallel per shard, so on Dask/EKS the gap grows with worker count. Iceberg's write cost — catalog transaction + manifest/metadata on top of the same Parquet — is real but bounded.

### What using real Iceberg changes (and what it doesn't)

The `ice` lane is not a relabelled Parquet run — going through a catalog moves three things:

| Hand-written Parquet | Through `pyiceberg` 0.11 |
|---|---|
| `row_group_size=50_000` (rows) | table property `write.parquet.row-group-limit` (rows, default 1,048,576). `write.parquet.row-group-size-bytes` is **accepted and ignored** — it warns `not implemented`, so byte-based tuning silently does nothing |
| `sorting_columns=[SortingColumn(0)]` | `SortOrder` is metadata only. PyIceberg does not sort on write and has no `rewrite_data_files`/compaction — sort the Arrow table yourself or W2 collapses |
| `dqa` as `uint16` | Iceberg has no unsigned types → widen to `int32` (that column doubles) |
| `pads.dataset(...).to_table(filter=...)` | `table.scan(row_filter=EqualTo(...)).to_arrow()`, which re-plans manifests per scan — 1.6× the pq50k point-read time it is layout-identical to (3.06 s vs 1.87 s), 3.4× pq1M |

None of it changes the *layout* argument: Iceberg's read path is these same Parquet files, so the hand-tuned Parquet lane remains the honest best case for Iceberg, and the `ice` lane shows the overhead a catalog adds on top.

### Scaling to 200M × 40k

| | Zarr dense | Iceberg long table |
|---|---|---|
| flux column | 32 TB raw → ~20 TB zstd | same data + 8×10^12 key pairs |
| One light curve | ~20 range GETs (chunks 500×2000) | fine *if* compacted + sorted by object |
| One epoch, all objects | ~400 chunk reads, parallel | full scan or second sort copy |
| Matrix ops (period search, PCA, ML batches) | native — `arr[i0:i1, :]` → NumPy/GPU | export/reshape 8T rows first |
| SQL / joins / BI | ✗ (export needed) | native |
| Row-level deletes | ✗ awkward | native |
| New per-epoch column | trivial (add array) | schema evolution (also fine) |

Recommended full-scale geometry: `shards=(25_000 objects, 40_000 epochs)` ≈ 2–4 GB objects (8,000 shards per quantity — S3-friendly count), inner `chunks=(500, 2_000)` ≈ 4 MB raw. One object's curve = 20 × ~2 MB range reads; batch 1,000 objects and the same reads amortize to near-scan throughput.

### The write path (the honest hard part)

Iceberg wins ingest: appending one epoch = append 200M rows to an epoch-partitioned table, cheap and transactional. Zarr appending one epoch means partially rewriting every chunk column (time-chunk width 2,000). So don't fight it — run the lakehouse pattern:

```
per-epoch pipeline output ──► Iceberg "hot" table (epoch-partitioned, ACID)
                                     │  every ~2,000 epochs (chunk-aligned)
                                     ▼  compaction job (Dask/Airflow)
                              Zarr/Icechunk dense matrix ("cold", read-optimized)
Queries: recent epochs → hot table;  bulk science / ML / curve serving → matrix
```

This is Iceberg-as-WAL, Zarr-as-columnar-serving — the exact hot/cold split Iceberg itself uses internally, one level up. Icechunk gives the matrix side commits and time travel, so a reprocessing campaign is a branch, not a copy.

### Iceberg-on-MinIO subset (moving the real Iceberg lane onto object storage)

`lc_bench.py` already runs the Iceberg lane against a local SQLite catalog and a `file://` warehouse. Point the same catalog at MinIO to add request latency, which is the variable that actually decides this comparison:

```bash
uv add "pyiceberg[sql-sqlite,s3fs]"    # already in the `extras` group
```

```python
from pyiceberg.catalog.sql import SqlCatalog
cat = SqlCatalog("lab", **{
    "uri": "sqlite:///iceberg_catalog.db",
    "warehouse": "s3://zarr-lab/warehouse",
    "s3.endpoint": "http://127.0.0.1:9000",
    "s3.access-key-id": "minioadmin",
    "s3.secret-access-key": "minioadmin",
})
# reuse ICE_SCHEMA / ice_write() from lc_bench.py verbatim; only the warehouse moves
```

**Exercises**:

1. Rerun `lc_bench.py` with all stores pointed at MinIO. Hypothesis to test, not assume: W2/W3/W4 gaps move in Zarr's favor once bytes and round-trips cost something (Zarr needs no metadata reads), and the 50k-row-group layout finally beats 1M because it fetches 15× less per curve. The warm-local numbers above say the opposite — find where the crossover is.
2. Grow the subset 10× (200k × 4k) and plot W2–W4 vs size. The ratios are what transfer to 200M — absolute times are just your disk.
3. Simulate the hybrid: land 100 epochs in the Iceberg table, then write a compaction task that flushes them as a chunk-aligned region write into the Zarr store.

### Real data: Multimodal Universe (replaces synthetic input)

https://github.com/MultimodalUniverse/MultimodalUniverse — 100 TB of astronomical data as HuggingFace datasets (native HDF5 at Flatiron; ~1k-example streaming previews per survey). Two time-series sets map exactly onto this module's hypothesis:

| Dataset | Cadence | Role in the test |
|---|---|---|
| **TESS** (160k LCs) | dense, regular, shared grid per sector | Positive test — GBTDS analog; validates chunking + *real* compression ratios |
| **PLAsTiCC** (3.5M) | ragged, sparse, multi-band | Negative control — measures NaN-fill cost where the dense matrix loses |

```bash
uv add datasets
```

```python
from datasets import load_dataset
tess = load_dataset("MultimodalUniverse/tess", split="train", streaming=True)
ex = next(iter(tess.with_format("numpy")))
# ex["lightcurve"] holds time/flux/flux_err arrays — stack ~1k of these into
# the (object, epoch) matrix and rerun lc_bench.py in place of the RNG data
```

**Exercises**:

1. Rerun W1–W4 with TESS flux. Synthetic Gaussian noise mis-predicts zstd ratios; real photometry is the number that matters for the 200M-scale storage estimate.
2. PLAsTiCC dense-fill experiment: grid a preview batch onto a common time axis, record the fill fraction, and find the NaN percentage at which Parquet long-table storage beats the padded matrix. That threshold is your written GO/NO-GO criterion for the design.
3. Bonus (ties to Module 7): the full MMU distribution is HDF5 — virtualize one file with VirtualiZarr instead of converting, and read it through the Zarr API.

> 💡 For true Roman GBTDS cadence, graduate to OpenUniverse2024 sims later; MMU is the fastest realistic data to get today. Cite per-survey licenses if any result leaves your laptop.

### Verdict

Zarr is a good fit — **for the per-epoch measurement block, not the whole 20-column table**. The dense (object × epoch) matrix eliminates 8 trillion stored keys, serves both access axes from one layout, and is the only option that hands light curves to NumPy/GPU code at full I/O speed. Keep object metadata and hot ingest in Iceberg. If your cadence turns out ragged across the survey (not per-field), the case collapses and Iceberg alone wins — check that first.

---

## Module 10 — Head-to-head vs HDF5/FITS, and the memory story

**Goal**: measured (not asserted) comparison with the array formats you already use, plus an explicit larger-than-RAM demonstration. Script: `fmt_bench.py`.

### Zarr vs HDF5 — what's actually true

Same 1 GB cube, same chunking, measured on a 1-CPU sandbox (rerun on your Mac for the parallel story):

| Config | 1-thread scan |
|---|---|
| HDF5 + gzip (the no-plugin default) | 9.2 s |
| HDF5 + blosc-zstd (`hdf5plugin`) | 3.2 s |
| Zarr + blosc-zstd | 2.7 s |

Honest reading:

1. **Same codec, single thread: near parity.** Locally, chunked HDF5 is a fine format. The 3.5× gap most people see is gzip vs zstd — a codec-ecosystem gap (HDF5 needs plugins your readers must install; Zarr codecs travel in the metadata), not raw-speed magic.
2. **Parallelism is the architectural difference.** h5py serializes every HDF5 call behind one global lock — threads decompress one at a time. Zarr chunks are independent objects with GIL-releasing codecs. Exercise 1 below measures this where it can show up.
3. **Cloud is the decisive difference.** Reading HDF5 over S3 means walking the file's internal B-tree: dependent, serial, small reads before any data arrives. Zarr computes the chunk key and issues one GET — nothing to walk. This is why Modules 4–5 work well and why "HDF5 on S3" ended up spawning kerchunk (Module 7) — whose whole job is extracting HDF5's chunk index once so it can be read *as if* it were Zarr.

**FITS in the same frame**: fine locally via memmap; on object storage, a classic FITS image HDU is one contiguous payload — no independent chunks, so no partial parallel reads unless tiled-compressed. Same remedy as HDF5: virtualize byte ranges (Module 7 spike) or convert hot data.

### Memory: out-of-core by default

Measured in the same run: reducing the full 1 GB on-disk array through Dask showed **~0 MB RSS growth** — chunks stream through workers and are dropped; the array is never materialized. The same code processes a 32 TB flux matrix (Module 9) on a laptop; only chunk size × active workers bounds memory. This is the property NumPy/FITS-into-RAM workflows lack and the reason Module 3's `open_zarr` is instant at any scale.

**Exercises**:

1. Run `fmt_bench.py` on your Mac (8+ cores). Expect: Zarr scan time drops with threads; HDF5 stays flat or degrades under lock contention. Record the crossover — that ratio is your per-node pipeline speedup.
2. Put `cube.h5` and `cube.zarr` on MinIO and time reading one chunk from each (h5py+`ros3`/fsspec vs zarr+s3fs). Count the requests each makes first — that is point 3 above, made concrete.
3. Cap Dask memory (`memory_limit="500MB"` per worker) and rerun the Module 9 full-scan on a dataset larger than the cap. Confirm it completes — the out-of-core guarantee, enforced.

### Where each claimed benefit lives in this course

| Benefit | Evidence |
|---|---|
| vs Parquet/Iceberg (tables) | Module 9 benchmark, epoch slice (3× warm-local, structural: Parquet must full-scan) |
| vs HDF5 (arrays) | this module: codec portability, lock-free parallel reads, cloud access |
| vs FITS | Module 1 ex.1, this module, Module 7 virtualization spike |
| Memory / out-of-core | this module RSS demo, Module 3 lazy open |
| Compute distribution | Module 5 Dask + region writes, Module 6 sharding |
| Data lakes | Module 7 Icechunk, Module 9 hybrid architecture |
| Coexisting with other formats | Module 7 VirtualiZarr/kerchunk, Module 9 Iceberg+Zarr split |
| Sky-coordinate access (WCS/GWCS) | Module 11, validated roundtrip |
| Pipeline write correctness (Airflow) | Module 12, validated corruption + fix |
| VO serving & visualization | Module 13 |

---

## Module 11 — WCS: sky-coordinate access (the astronomer's module)

**Goal**: answer "how do I cut out by RA/Dec?" — the first objection to index-only arrays.

There is no IVOA/Zarr WCS standard. The working pattern: store the WCS *with* the array, reconstruct it at read time, and let it translate sky → index → chunk slice. Validated example (`val_wcs.py`, ships with course):

```python
import zarr, numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

g = zarr.open_group("mosaic.zarr", mode="w")
img = g.create_array("sci", shape=(4096, 4096), chunks=(512, 512), dtype="float32")
img.attrs["fits_wcs_header"] = dict(my_wcs.to_header())   # JSON-native, travels with data

# Reader: sky -> pixel -> chunk-aligned slice (only touched chunks are read)
w = WCS(dict(zarr.open_group("mosaic.zarr")["sci"].attrs["fits_wcs_header"]))
x, y = w.world_to_pixel(SkyCoord(269.48*u.deg, -28.62*u.deg))
cutout = img[int(y)-64:int(y)+64, int(x)-64:int(x)+64]
```

Roundtrip verified: sky → pixel → sky separation = 0.0 mas. The cutout read touches only intersecting chunks — this *is* the SODA `CIRCLE` primitive, minus the service.

**Roman reality — GWCS, not FITS WCS**: L2 products carry GWCS (distortion, full pipelines) serialized in ASDF, and GWCS objects are not JSON. Pattern: serialize with `asdf` to bytes and store as a `uint8` sidecar array in the group (`grp.create_array("wcs_asdf", ...)`); readers `asdf.open(BytesIO(bytes))` and get the full transform. Attrs hold a cheap approximate FITS-SIP header for tools that only speak FITS WCS.

**Exercises**:

1. Run `val_wcs.py`; then repeat the cutout on the MinIO copy and confirm request count = chunks intersected, nothing more.
2. Store a `romancal` L2 GWCS via the uint8-sidecar pattern and verify roundtrip `world_to_pixel` against the original ASDF file.
3. Time-axis WCS: for the Module 9 matrix, the "WCS" is the shared `time[epoch]` coordinate — same idea, one array.

**Docs**: GWCS https://gwcs.readthedocs.io · astropy WCS https://docs.astropy.org/en/stable/wcs/

---

## Module 12 — Airflow ingestion done safely (the write-correctness module)

**Goal**: the failure mode Module 5 glossed over — and its fix. Validated (`val_tx.py`).

### The problem, demonstrated

A KubernetesPodOperator task region-writes 30 rows; the pod is OOM-killed after 15:

```python
z[:] = 1.0                    # good committed state (epoch batch 0)
z[0:15, :] = new[0:15]        # task dies here; retry never lands
```

Measured result: readers now see **mixed epochs `[1.0, 2.0]` with no error, no flag, nothing**. Plain Zarr has no transactions — every chunk PUT is immediately live. In a DAG with retries and concurrent readers this is silent data corruption.

### The fix, demonstrated

Same failure under Icechunk:

```python
s = repo.writable_session("main")
zarr.open_array(s.store, path="flux", mode="r+")[0:15, :] = 2.0
# crash before s.commit() -> readers on main still see [1.0] only
```

Measured: uncommitted writes are invisible; the retry re-writes the *full* region in a fresh session and commits atomically — readers flip from all-`1.0` to all-`2.0`, never mixed.

### The DAG pattern

```python
# AIDEV-NOTE: one task = one chunk-aligned region = one atomic commit
def flush_epoch_batch(batch_id: int, run_id: str, try_number: int):
    repo = icechunk.Repository.open(storage)          # S3-backed in prod
    session = repo.writable_session("main")
    arr = zarr.open_array(session.store, path="flux", mode="r+")
    t0 = batch_id * TIME_CHUNK                        # chunk-aligned: no read-modify-write
    arr[:, t0:t0 + TIME_CHUNK] = load_batch_from_iceberg_hot_table(batch_id)
    session.commit(f"flux batch={batch_id} run_id={run_id} try={try_number}")
```

Three properties make this retry-proof: **chunk-aligned** regions (no partial-chunk read-modify-write, no cross-task chunk sharing), **deterministic** payload (retry writes identical bytes), **atomic commit** carrying `run_id` — which doubles as provenance (gap #7): `repo.ancestry()` is your lineage log, joinable to OpenLineage events by run_id.

If Icechunk is off the table (interop constraint), the fallback is write-to-staging-prefix + consolidate + atomic pointer flip — more moving parts, same idea, you own the failure matrix.

**Exercises**:

1. Run `val_tx.py`. Then add a concurrent-writer test: two sessions committing to `main` — observe Icechunk's conflict rejection instead of last-write-wins.
2. Wire the pattern into a real 3-task Airflow DAG against MinIO with `retries=2`, and `kill -9` one task mid-write. Verify `main` never exposes a partial batch.
3. Decide your commit granularity: per-task commits serialize on the branch; batching commits per DAG-run trades latency for less contention. Measure both.

---

## Module 13 — Serving & visualization (VO + browsers)

**Goal**: how the outside world reads the store — the adoption barrier Module 4 didn't cover.

### Serving without a server

A Zarr store in a public/anonymous bucket is already a data service: any HTTP client that can do range GETs is a full-capability reader. What VO integration honestly looks like today (no IVOA Zarr standard exists):

| Layer | Pattern |
|---|---|
| Discovery | ObsCore record as usual; `access_url` → store root, `access_format` → a declared Zarr media type |
| Fine-grained links | DataLink `#this` → store root; `#cutout` → a thin SODA service |
| SODA | Still worth a thin service for standards-compliant clients: it resolves CIRCLE/POLYGON via Module 11's WCS and either streams a FITS cutout (legacy clients) or 303-redirects to a byte-range recipe (Zarr-aware clients) |

The SODA service becomes a translator, not a data mover — your UWS/cutout architecture slots in unchanged, with the CutoutEngine reading Zarr instead of FITS.

### Looking at it

DS9, Firefly, and jdaviz do not read Zarr natively today. Working paths:

1. **Browser-native**: `zarrita.js` reads Zarr v3 over HTTP ranges directly — a vanilla-JS, no-backend viewer over a public store is exactly your locked stack (spike candidate: `popcube`?). Anywidget/jupyter-scatter cover notebooks.
2. **Notebook**: xarray → matplotlib/hvplot is the default; jdaviz accepts in-memory arrays, so `open_zarr` → `Spectrum1D`/`NDData` bridges it.
3. **Legacy tools**: convert-on-demand — the SODA cutout above emits small FITS for DS9/Firefly users. Cheap, keeps everyone served during transition.

**Exercises**:

1. `mc anonymous set download local/zarr-lab` then read the Module 4 store with *plain* `curl` range requests — compute the chunk key by hand once; that's the whole protocol.
2. 30-line HTML page with zarrita.js rendering one chunk of the MinIO store to a canvas. No build step needed.
3. Draft the ObsCore + DataLink records for the Module 9 light-curve store; decide what `#this` means when the "product" is a matrix.

**Docs**: zarrita.js https://github.com/manzt/zarrita.js · SODA https://www.ivoa.net/documents/SODA/

---

## Decision guide: when Zarr vs your other formats

| Data shape | Use | Why |
|---|---|---|
| Tabular catalogs, SQL access | Parquet / Iceberg | Columnar, predicate pushdown, engines |
| Spatially partitioned catalogs | HATS + LSDB | HEALPix-aware cross-match |
| N-D cubes, image stacks, grids | **Zarr (+ xarray)** | Chunked parallel array I/O |
| Versioned cube lake, ACID | **Icechunk** | Transactions + time travel on Zarr |
| Single-file archival delivery w/ schema | ASDF / FITS | Standards, validation, portability |
| Legacy NetCDF/HDF5 archive, cloud reads | VirtualiZarr/kerchunk refs | No rewrite, Zarr-speed access |

Known limits to keep in mind: no built-in schema validation (vs ASDF), many-object sprawl without sharding, Zarr v2↔v3 ecosystem still settling — pin `zarr>=3` in every project.

---

## References

- Zarr v3 spec: https://zarr-specs.readthedocs.io
- zarr-python docs: https://zarr.readthedocs.io
- xarray Zarr I/O: https://docs.xarray.dev/en/stable/user-guide/io.html#zarr
- Icechunk: https://icechunk.io
- VirtualiZarr: https://virtualizarr.readthedocs.io
- Pangeo guide (cloud-native patterns): https://pangeo.io
- s3fs (MinIO/endpoint config): https://s3fs.readthedocs.io

## Next Steps

1. Run Modules 0–3 in one sitting (~90 min) — core mental model.
2. Modules 4–5 next session — the S3 + Dask patterns that map to your EKS work.
3. Spike after finishing: virtualize one real IRSA NetCDF/HDF5 (or FITS) file with VirtualiZarr and time cutout reads vs direct file access — concrete GO/NO-GO evidence for Roman use.
