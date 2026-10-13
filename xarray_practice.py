# -*- coding: utf-8 -*-
"""
xarray 综合练习脚本（迁移自主对话学习内容）
流程：读数据 → 区域平均时间序列 → 气候态距平 → 画两张图
依赖：xarray, matplotlib, cartopy
运行前：把下面 open_dataset 的路径换成你自己的 ERA5 文件
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. 读数据（替换为你的文件）
ds = xr.open_dataset("era5_2t_2020_2024.nc")
t2m = ds["t2m"] - 273.15            # K → ℃

# 2. 中国区域平均时间序列
cn = t2m.sel(lat=slice(15, 55), lon=slice(70, 140)).mean(dim=["lat", "lon"])

# 3. 逐日气候态 + 距平
clim_day = cn.groupby("time.dayofyear").mean("time")
anom = cn.groupby("time.dayofyear") - clim_day

# 4. 画时间序列图
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
cn.plot(ax=axes[0], color="tab:red")
axes[0].set_title("China area-mean T2m (℃)")
anom.plot(ax=axes[1], color="tab:blue")
axes[1].set_title("Anomaly (℃)")
axes[1].axhline(0, color="k", lw=0.8)
plt.tight_layout()
plt.savefig("series.png", dpi=200)
plt.show()

# 5. 画 2020-07-01 空间场
field = t2m.sel(time="2020-07-01", method="nearest")
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
cf = ax.contourf(field.lon, field.lat, field, transform=ccrs.PlateCarree(),
                 cmap="RdBu_r", levels=20)
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.set_title("T2m 2020-07-01 (℃)")
plt.colorbar(cf, orientation="horizontal", shrink=0.8)
plt.savefig("map.png", dpi=200)
plt.show()
