# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 12 — Airflow ingestion done safely (write correctness)
# Plain Zarr has **no transactions**: every chunk PUT is immediately live. A task retry
# that died mid region-write leaves silently mixed data. This notebook demonstrates the
# corruption in-process, then the process-kill variant via `scripts/crash_task.py`
# (a kernel cannot kill itself), then the Icechunk fix.

# %% [markdown]
# ## 1. The corruption, in-process

# %%
import zarr, numpy as np
z = zarr.create_array("../data/plain.zarr", shape=(100, 1000), chunks=(10, 1000),
                      dtype="float32", fill_value=0, overwrite=True)
z[:] = 1.0                                   # good state: epoch batch 0
z[0:15, :] = 2.0                             # task writes half its 30-row region... "crash"
seen = np.unique(zarr.open_array("../data/plain.zarr", mode="r")[0:30, 0])
print("reader sees:", seen, "<- mixed epochs, no error, no flag")

# %% [markdown]
# ## 2. The real thing: kill -9 an actual writer process mid-write

# %%
import subprocess, sys
p = subprocess.run([sys.executable, "../scripts/crash_task.py", "../data/plain2.zarr"])
print("writer exit code:", p.returncode, "(non-zero = died mid-write, like an OOM-killed pod)")
seen = np.unique(zarr.open_array("../data/plain2.zarr", mode="r")[0:30, 0])
print("store now contains:", seen)

# %% [markdown]
# ## 3. The fix: Icechunk sessions — uncommitted writes are invisible

# %%
import icechunk
repo = icechunk.Repository.create(icechunk.in_memory_storage())
s = repo.writable_session("main")
za = zarr.create_array(s.store, name="flux", shape=(100, 1000), chunks=(10, 1000),
                       dtype="float32", fill_value=0)
za[:] = 1.0
s.commit("epoch batch 0")

s2 = repo.writable_session("main")
zarr.open_array(s2.store, path="flux", mode="r+")[0:15, :] = 2.0   # crash before commit
ro = repo.readonly_session(branch="main")
print("main readers see:", np.unique(zarr.open_array(ro.store, path="flux", mode="r")[0:30, 0]))

# %%
s3 = repo.writable_session("main")                                  # the retry
zarr.open_array(s3.store, path="flux", mode="r+")[0:30, :] = 2.0    # full region, deterministic
cid = s3.commit("flux batch=1 run_id=manual__2026-08-03 try=2")     # atomic + provenance
ro2 = repo.readonly_session(branch="main")
print("after atomic commit:", np.unique(zarr.open_array(ro2.store, path="flux", mode="r")[0:30, 0]))
print("lineage:", [c.message for c in repo.ancestry(branch="main")])

# %% [markdown]
# ## The DAG pattern (three retry-proof properties)
# ```python
# # AIDEV-NOTE: one task = one chunk-aligned region = one atomic commit
# def flush_epoch_batch(batch_id, run_id, try_number):
#     session = icechunk.Repository.open(storage).writable_session("main")
#     arr = zarr.open_array(session.store, path="flux", mode="r+")
#     t0 = batch_id * TIME_CHUNK                    # 1) chunk-aligned
#     arr[:, t0:t0+TIME_CHUNK] = load_batch(batch_id)   # 2) deterministic payload
#     session.commit(f"flux batch={batch_id} run_id={run_id} try={try_number}")  # 3) atomic
# ```
# Commit log = provenance, joinable to OpenLineage by run_id. Exercise: wire this into a
# 3-task DAG against MinIO with retries=2 and kill one pod mid-write; verify `main` never
# exposes a partial batch. Also test two concurrent sessions committing -> conflict
# rejection instead of last-write-wins.
