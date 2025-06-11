import timeit
import numpy as np
import fasterraster as fr
import slidepy as sp
from pathlib import Path
from fasterraster.fasterraster import Raster
from fasterraster.fasterraster.flo import Flo

# NTESTS = 10

# # Load grid files
# dir = Path('./test_data/')
# dem = Raster()
# dem.read(dir / 'dem.bil')
# mask = Raster()
# mask.read(dir / 'mask.bil')
# ssem = Raster()
# ssem.read(dir / 'ssem.bil')
# vel = Flo(dir / 'vel.flo')
# # vel = fr.Flo(dir / 'vel.flo')

# # prep velocity grids
# fr.multiplyFloMask(vel.raster, mask.raster) # 0 velocity values outslide of landslide extent
# u, v = fr.flo_to_u_v(vel.raster)            # split velocity grid into u & v components

# # regular python implementation of com function
# def py_com(dem, u, v, ssem, cell_size, epochs):

#     dem_cpy = dem.copy()

#     dl = 2. * cell_size
#     rows = dem_cpy.shape[0] - 2
#     cols = dem_cpy.shape[1] - 2

#     # calculate depth
#     h = dem_cpy - ssem

#     for i in range(epochs):
#         for i in range(1, rows):
#             for j in range(1, cols):
#                 dem_cpy[i,j] -= ((h[i,j] * (u[i,j-1] - u[i,j+1]) / dl) + (u[i,j] * (h[i,j-1] - h[i,j+1]) / dl)) + ((h[i,j] * (v[i+1,j] - v[i-1,j]) / dl) + (v[i,j] * (h[i+1,j] - h[i-1,j]) / dl))
#         for i in range(1, rows):
#             for j in range(1, cols):
#                 h[i,j] = dem_cpy[i,j] - ssem[i,j]
#     return dem_cpy

# # regular numpy implementation of com function
# def np_com(dem, u, v, ssem, cell_size, epochs):
    
#     dem_cpy = dem.copy()

#     # calculate depth
#     h = dem_cpy - ssem

#     # calculate vel gradients
#     du = -1 * np.gradient(u, axis=1) / cell_size
#     dv = np.gradient(v, axis=1) / cell_size

#     for i in range(epochs):
#         # calculate depth gradient
#         dh_v, dh_u = np.gradient(h)
#         dh_u = -1 * dh_u / cell_size
#         dh_v = dh_v / cell_size

#         # calculate dz
#         dz_u = (h * du) + (u * dh_u)
#         dz_v = (h * dv) + (v * dh_v)
#         dz = dz_u + dz_v

#         # update dem & depth
#         dem_cpy = dem_cpy - dz
#         h = dem_cpy - ssem
    
#     return dem_cpy

# # Time Conservation of mass simulation using regular python
# time = timeit.timeit(lambda: py_com(dem.raster, u, v, ssem.raster, dem.XDIM, 1), number=1)
# print(f'python COM took {time:.3f} seconds')

# # Time Conservation of mass simulation using numpy
# time = timeit.timeit(lambda: np_com(dem.raster, u, v, ssem.raster, dem.XDIM, 1), number=1)
# print(f'numpy COM took {time:.3f} seconds')

# # Time Conservation of mass simulation using open-MP and SIMD for numt-threads
# num_threads = [1,2,4,8]
# for numt in num_threads:
#     time = timeit.timeit(lambda: sp.com_mp(dem.raster, u, v, ssem.raster, dem.XDIM, 1, numt), number=NTESTS)
#     print(f'MP COM averaged {time/NTESTS:.3f} seconds for {numt} threads')

# for numt in num_threads:
#     time = timeit.timeit(lambda: sp.com_sse(dem.raster, u, v, ssem.raster, dem.XDIM, 1, numt), number=NTESTS)
#     print(f'SSE COM averaged {time/NTESTS:.3f} seconds for {numt} threads')
from slidepy import generate_vel
# from slidepy.fasterraster import Flo
import rasterio
import numpy as np

# 1. Generate raw velocity array from two DEMs (2019 → 2021)
vel_array = generate_vel(
    dem_early="dem_2019.tif", 
    dem_late="dem_2021.tif", 
    time_diff=2.0,            # in years
    mask="mask.bil"           # only compute inside landslide
)
# vel_array shape = (H, W, 2)

# 2. Open DEM to grab transform & crs
with rasterio.open("dem.bil") as src:
    transform = src.transform
    crs = src.crs
    nodata = src.nodata
    height = src.height
    width = src.width

# 3. Convert (H, W, 2) → (2, H, W)
vel_data = np.moveaxis(vel_array, -1, 0).astype(np.float32)

# 4. Wrap in Flo
flo_obj = Flo(fname=None, data=vel_data)
flo_obj.transform = transform
flo_obj.crs = crs
flo_obj.nodata = 0.0

# 5. Save to disk
flo_obj.to_file("vel.flo")

# 6. (Optional) Verify
v2 = Flo("vel.flo")
print("Shape:", v2.data.shape)         # should be (2, H, W)
print("Pixel size:", v2.transform[0])   # same as DEM
print("CRS:", v2.crs)                   # same as DEM
