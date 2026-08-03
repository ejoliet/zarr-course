# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %% [markdown]
# # 11 — WCS: sky-coordinate access (the astronomer's module)
# No IVOA/Zarr WCS standard exists. Working pattern: WCS travels **with** the array
# (attrs or ASDF sidecar), reader reconstructs it, sky -> pixel -> chunk-aligned slice.

# %%
import zarr, numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

w = WCS(naxis=2)
w.wcs.crpix = [2048.5, 2048.5]
w.wcs.cdelt = [-0.11/3600, 0.11/3600]     # Roman-like 0.11 arcsec/pix
w.wcs.crval = [269.5, -28.6]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

g = zarr.open_group("../data/mosaic.zarr", mode="w")
img = g.create_array("sci", shape=(4096, 4096), chunks=(512, 512), dtype="float32")
img[:] = np.random.default_rng(0).normal(size=(4096, 4096)).astype("float32")
img.attrs["fits_wcs_header"] = dict(w.to_header())
print("WCS stored in attrs (JSON-native)")

# %% [markdown]
# ## Reader side: this IS the SODA CIRCLE primitive, minus the service

# %%
g2 = zarr.open_group("../data/mosaic.zarr", mode="r")
w2 = WCS(dict(g2["sci"].attrs["fits_wcs_header"]))
target = SkyCoord(269.48 * u.deg, -28.62 * u.deg)
x, y = w2.world_to_pixel(target)
cut = g2["sci"][int(y)-64:int(y)+64, int(x)-64:int(x)+64]
sep = target.separation(w2.pixel_to_world(x, y)).to(u.mas).value
print(f"pixel ({x:.1f},{y:.1f})  cutout {cut.shape}  roundtrip sep {sep:.6f} mas")

# %%
import matplotlib.pyplot as plt
plt.imshow(cut, origin="lower"); plt.title("sky-coordinate cutout"); plt.show()

# %% [markdown]
# ## Roman reality: GWCS, not FITS WCS
# L2 GWCS (distortion, full transform chains) is not JSON. Pattern:
# serialize with `asdf` to bytes -> store as `uint8` sidecar array in the group ->
# readers `asdf.open(BytesIO(...))` and get the full transform. Keep an approximate
# FITS-SIP header in attrs for legacy tools.
# Exercise: do this with a real `romancal` L2 product and verify `world_to_pixel`
# against the original ASDF — the Roman-credibility check.
