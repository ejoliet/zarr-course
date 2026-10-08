"""Module 11 validation: WCS in Zarr attrs -> sky-coordinate chunk-aligned cutout."""
import zarr, numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# Build a mosaic-like image with a WCS
w = WCS(naxis=2)
w.wcs.crpix = [2048.5, 2048.5]
w.wcs.cdelt = [-0.11/3600, 0.11/3600]   # Roman-like 0.11"/pix
w.wcs.crval = [269.5, -28.6]            # bulge-ish field
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

g = zarr.open_group("mosaic.zarr", mode="w")
img = g.create_array("sci", shape=(4096, 4096), chunks=(512, 512), dtype="float32")
img[:] = np.random.default_rng(0).normal(size=(4096, 4096)).astype("float32")
g["sci"].attrs["fits_wcs_header"] = dict(w.to_header())   # JSON-native

# --- reader side: reconstruct WCS, sky -> pixel -> slice ---
g2 = zarr.open_group("mosaic.zarr", mode="r")
w2 = WCS(dict(g2["sci"].attrs["fits_wcs_header"]))
target = SkyCoord(269.48*u.deg, -28.62*u.deg)
x, y = w2.world_to_pixel(target)
half = 64
cut = g2["sci"][int(y)-half:int(y)+half, int(x)-half:int(x)+half]
print("pixel:", round(float(x),1), round(float(y),1), "| cutout:", cut.shape)
# roundtrip check
back = w2.pixel_to_world(x, y)
print("roundtrip sep (mas):", round(target.separation(back).to(u.mas).value, 6))
