"""Simulates an OOM-killed pipeline task: writes HALF its region, then dies hard.
Used by notebook 12. Usage: python crash_task.py <store_path>"""
import sys, os
import numpy as np
import zarr

store = sys.argv[1]
z = zarr.create_array(store, shape=(100, 1000), chunks=(10, 1000),
                      dtype="float32", fill_value=0, overwrite=True)
z[:] = 1.0                      # good committed state (epoch batch 0)
z[0:15, :] = 2.0                # task's region is [0:30] -- it gets halfway...
os._exit(1)                     # ...and the pod is OOM-killed. No cleanup, no exception.
