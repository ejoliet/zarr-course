"""Module 12 validation: (1) plain Zarr partial-write corruption is silent,
(2) Icechunk makes the same failure invisible to readers until commit."""
import zarr, numpy as np, icechunk, shutil

# --- 1) plain zarr: simulated task crash mid region-write ---
shutil.rmtree("plain.zarr", ignore_errors=True)
z = zarr.create_array("plain.zarr", shape=(100, 1000), chunks=(10, 1000),
                      dtype="float32", fill_value=0)
z[:] = 1.0                              # committed good state
new = np.full((30, 1000), 2.0, "float32")
z[0:15, :] = new[0:15]                  # task writes half its region...
# ...pod OOM-killed here. Retry never happens.
z2 = zarr.open_array("plain.zarr", mode="r")
seen = np.unique(z2[0:30, 0])
print("plain zarr reader sees mixed epochs:", seen, "<- silent corruption")

# --- 2) icechunk: same failure, invisible until commit ---
repo = icechunk.Repository.create(icechunk.in_memory_storage())
s = repo.writable_session("main")
za = zarr.create_array(s.store, name="flux", shape=(100, 1000), chunks=(10, 1000),
                       dtype="float32", fill_value=0)
za[:] = 1.0
snap = s.commit("epoch batch 0 (good state)")

s2 = repo.writable_session("main")
zb = zarr.open_array(s2.store, path="flux", mode="r+")
zb[0:15, :] = 2.0                       # crash before s2.commit()
# reader on main:
ro = repo.readonly_session(branch="main")
zr = zarr.open_array(ro.store, path="flux", mode="r")
print("icechunk reader on main sees:", np.unique(zr[0:30, 0]), "<- uncommitted write invisible")

# retry: new session, full region, atomic commit
s3 = repo.writable_session("main")
zc = zarr.open_array(s3.store, path="flux", mode="r+")
zc[0:30, :] = 2.0
print("retry commit:", s3.commit("epoch batch 1, run_id=manual__2026-08-03, attempt=2")[:12])
ro2 = repo.readonly_session(branch="main")
print("after commit:", np.unique(zarr.open_array(ro2.store, path="flux", mode="r")[0:30, 0]))
