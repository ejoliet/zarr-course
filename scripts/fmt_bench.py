"""Module 10 benchmark: Zarr vs HDF5 (same data, same chunking) + out-of-core memory demo.

Run on a multi-core machine (macOS): thread-scaling results are meaningless on 1 CPU.
  uv add "zarr>=3" h5py hdf5plugin "dask[array]" numpy psutil
  uv run python fmt_bench.py
"""
import time, numpy as np, h5py, hdf5plugin, zarr, dask.array as da, psutil
from zarr.codecs import BloscCodec

SHAPE, CHUNK = (256, 1024, 1024), (16, 256, 256)   # 1 GB float32
zstd_h5 = hdf5plugin.Blosc(cname="zstd", clevel=4, shuffle=hdf5plugin.Blosc.SHUFFLE)

data = np.random.default_rng(0).normal(size=SHAPE).astype("float32")
with h5py.File("cube.h5", "w") as f:
    f.create_dataset("gz", data=data, chunks=CHUNK, compression="gzip", compression_opts=4)
    f.create_dataset("zs", data=data, chunks=CHUNK, **zstd_h5)
z = zarr.create_array("cube.zarr", shape=SHAPE, chunks=CHUNK, dtype="float32",
                      compressors=BloscCodec(cname="zstd", clevel=4, shuffle="shuffle"),
                      overwrite=True)
z[:] = data
del data

def bench(label, arr, workers):
    t0 = time.perf_counter()
    float(arr.std(axis=0).mean().compute(scheduler="threads", num_workers=workers))
    print(f"{label:44s} {time.perf_counter()-t0:6.2f} s")

f = h5py.File("cube.h5", "r")
for w in (1, 4, 8):
    bench(f"hdf5 gzip        {w}-thread scan", da.from_array(f["gz"], chunks=CHUNK), w)
    bench(f"hdf5 blosc-zstd  {w}-thread scan", da.from_array(f["zs"], chunks=CHUNK), w)
    bench(f"zarr blosc-zstd  {w}-thread scan", da.from_zarr("cube.zarr"), w)

# Out-of-core: peak RSS while reducing the on-disk array
proc = psutil.Process()
base = proc.memory_info().rss
float(da.from_zarr("cube.zarr").mean().compute(scheduler="threads", num_workers=4))
grow = proc.memory_info().rss - base
print(f"\narray on disk: 1024 MB | RSS growth during full reduction: {grow/1e6:.0f} MB")
