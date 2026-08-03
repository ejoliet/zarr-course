# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 02 — Groups, hierarchy, compression
# Build a mission-style hierarchy. **This is your multi-extension FITS**:
#
# | MEF concept | Zarr concept |
# |---|---|
# | HDU (SCI/ERR/DQ) | array in a group |
# | header keywords | `attrs` (JSON) |
# | file | store (directory / S3 prefix) |
#
# What you lose vs ASDF: schema validation, typed trees. What you gain: chunked cloud reads.

# %%
import zarr, numpy as np
from zarr.codecs import BloscCodec
root = zarr.open_group("../data/mission.zarr", mode="w")
l2 = root.create_group("L2")
kw = dict(shape=(50, 1024, 1024), chunks=(1, 512, 512), dtype="float32",
          compressors=BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"))
sci = l2.create_array("sci", **kw)
err = l2.create_array("err", **kw)
dq  = l2.create_array("dq", shape=kw["shape"], chunks=kw["chunks"], dtype="uint16")
sci.attrs.update({"unit": "MJy/sr", "instrument": "WFI-like", "level": "L2"})
sci[:] = np.random.default_rng(1).gamma(2.0, size=sci.shape).astype("float32")
root.tree()

# %% [markdown]
# ## Exercise — codec shoot-out on astronomy-like floats
# Shuffle matters for noisy floats; higher clevel often buys little.

# %%
import time, shutil, os
def du(p): return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)
data = np.random.default_rng(2).gamma(2.0, size=(10, 1024, 1024)).astype("float32")
for cname, lvl in [("zstd", 1), ("zstd", 5), ("zstd", 9), ("lz4", 5)]:
    p = f"../data/codec_{cname}{lvl}.zarr"
    t0 = time.perf_counter()
    a = zarr.create_array(p, shape=data.shape, chunks=(1, 512, 512), dtype="float32",
                          compressors=BloscCodec(cname=cname, clevel=lvl, shuffle="shuffle"),
                          overwrite=True)
    a[:] = data
    print(f"{cname}-{lvl}: write {time.perf_counter()-t0:5.2f}s  size {du(p)/1e6:6.1f} MB")
