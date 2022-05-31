import xarray
from matplotlib.colors import ListedColormap
import earthpy.plot as ep

class_bins = [canopy_HARV.min().values, 2, 10, 20, np.inf]

height_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
height_cmap = ListedColormap(height_colors)

category_names = [
    "No Vegetation",
    "Bare Area",
    "Low Canopy",
    "Medium Canopy",
    "Tall Canopy",
]

category_indices = list(range(len(category_names)))


canopy_height_classified = xarray.apply_ufunc(
    np.digitize,  
    canopy_HARV,  
    class_bins   
)

plt.style.use("default")
plt.figure()
im = canopy_height_classified.plot(cmap=height_cmap, add_colorbar=False)
return im