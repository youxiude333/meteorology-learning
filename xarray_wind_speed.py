# -*- coding: utf-8 -*-
"""
物理正确的风速合成示例（可复制直接运行）
场景：中国区域不同经纬度格点，用 10m 纬向风 u10 与经向风 v10 合成合风速。
物理公式：wind = sqrt(u10^2 + v10^2)      # u=纬向风, v=经向风
依赖：numpy, xarray（不依赖 cartopy，可直接运行）
运行：python xarray_wind_speed.py
"""

import numpy as np
import xarray as xr

# ========== 1. 定义网格：不同经纬度（中国区域 70~139°E, 15~55°N） ==========
lon = np.arange(70, 140, 1.0)      # 经度 70~139°E，共 70 个格点
lat = np.arange(15, 56, 1.0)       # 纬度 15~55°N，共 41 个格点
time = np.arange(0, 6)             # 6 个时次（示意）

print("网格规模：time × lat × lon =", len(time), "×", len(lat), "×", len(lon))

# ========== 2. 生成两个风分量（u=纬向风, v=经向风） ==========
# u10：纬向风（东西向），正值为西风。叠加均值约 5 m/s 模拟西风带。
# v10：经向风（南北向），正值为南风，均值约 0。
rng = np.random.default_rng(42)     # 固定随机种子 → 结果可复现
u10 = xr.DataArray(
    rng.normal(0, 2, (len(time), len(lat), len(lon))) + 5.0,  # 平均约 5 m/s 西风
    dims=("time", "lat", "lon"),
    coords={"time": time, "lat": lat, "lon": lon},
    name="u10",
    attrs={"units": "m s-1", "long_name": "10m eastward wind (u)"},
)
v10 = xr.DataArray(
    rng.normal(0, 2, (len(time), len(lat), len(lon))),        # 南北风，均值约 0
    dims=("time", "lat", "lon"),
    coords={"time": time, "lat": lat, "lon": lon},
    name="v10",
    attrs={"units": "m s-1", "long_name": "10m northward wind (v)"},
)

print("\nu10 形状:", u10.shape, "| 单位:", u10.attrs["units"])
print("v10 形状:", v10.shape, "| 单位:", v10.attrs["units"])

# ========== 3. 物理正确的风速合成：wind = sqrt(u^2 + v^2) ==========
wind = (u10**2 + v10**2) ** 0.5
wind.attrs = {"units": "m s-1", "long_name": "10m wind speed"}

print("\n合风速 wind 形状:", wind.shape, "| 单位:", wind.attrs["units"])
print("u10 范围: %.2f ~ %.2f m/s" % (u10.values.min(), u10.values.max()))
print("v10 范围: %.2f ~ %.2f m/s" % (v10.values.min(), v10.values.max()))
print("wind 范围: %.2f ~ %.2f m/s" % (wind.values.min(), wind.values.max()))

# ========== 4. 物理正确性验证：手动复算某一点 ==========
# 取 (time=0, lat=30, lon=105) 这个格点，人工用公式核验 wind 是否一致
u0 = float(u10.isel(time=0).sel(lat=30, lon=105, method="nearest"))
v0 = float(v10.isel(time=0).sel(lat=30, lon=105, method="nearest"))
w0 = float(wind.isel(time=0).sel(lat=30, lon=105, method="nearest"))
manual = (u0**2 + v0**2) ** 0.5
print("\n验证格点 (lat=30, lon=105, time=0):")
print("  u10 = %.4f, v10 = %.4f" % (u0, v0))
print("  公式 √(u²+v²) = %.4f" % manual)
print("  wind 实际值    = %.4f" % w0)
print("  两者一致？", "是 ✓" if abs(w0 - manual) < 1e-9 else "否 ✗")

# ========== 5. 应用：某点时间序列 + 全区域平均风速 ==========
ts = wind.sel(lat=30, lon=105, method="nearest")   # 该点随时间变化
print("\n(lat=30, lon=105) 的 6 个时次风速:", np.round(ts.values, 2))
region_mean = wind.mean(dim=["time", "lat", "lon"])  # 全场平均风速
print("全区域平均风速: %.2f m/s" % float(region_mean))

# ========== 6.（可选）画一张空间分布图 ==========
# 需要 matplotlib + cartopy。若已安装，取消注释即可运行：
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# field = wind.isel(time=0)
# fig = plt.figure(figsize=(9, 5))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
# cf = ax.contourf(field.lon, field.lat, field,
#                  transform=ccrs.PlateCarree(), cmap="viridis", levels=15)
# ax.coastlines(linewidth=0.8)
# ax.add_feature(cfeature.BORDERS, linewidth=0.5)
# ax.set_title("10m wind speed (m/s)")
# plt.colorbar(cf, orientation="horizontal", shrink=0.8)
# plt.show()

print("\n运行完成。")
