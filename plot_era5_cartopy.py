# -*- coding: utf-8 -*-
"""
Cartopy 画 ERA5 真实数据的空间温度图
依赖：xarray, matplotlib, cartopy, netcdf4
运行前：把 DATA_FILE 改成你的 ERA5 文件路径
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# ★ 改成你的 ERA5 文件路径 ★
# ============================================================
DATA_FILE = r"C:\Users\86137\Downloads\dbda254db6bf7a56ffa3e6a8797f34c.nc"

# 要画哪个时间步（可以改成你数据里有的时间，比如 "2005-01-15"）
PLOT_TIME = "2005-01-15"

# ============================================================
# 1. 读取数据
# ============================================================
print("读取数据:", DATA_FILE)
ds = xr.open_dataset(DATA_FILE)
print("变量:", list(ds.data_vars))

# 取 t2m（2米温度），选指定时间，取最近的时次
t2m = ds["t2m"].sel(valid_time=PLOT_TIME, method="nearest")

# 温度从开尔文转摄氏度
t2m_c = t2m - 273.15

# 取经纬度
lons = ds["longitude"].values
lats = ds["latitude"].values

print("画图时间:", t2m_c["valid_time"].values)
print("温度范围: %.1f ~ %.1f ℃" % (float(t2m_c.min()), float(t2m_c.max())))

# ============================================================
# 2. 创建画布和地图投影
# ============================================================
# PlateCarree 是最常用的等经纬度投影（经纬度直接当 xy 坐标）
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置显示范围（中国区域，可根据需要调整）
ax.set_extent([73, 135, 15, 54], crs=ccrs.PlateCarree())

# ============================================================
# 3. 画填色等值线图
# ============================================================
# contourf = filled contour（填色等值线）
# transform=ccrs.PlateCarree() 告诉 cartopy 数据是经纬度格式
levels = np.arange(-30, 35, 2.5)  # 色标范围：-30到35℃，间隔2.5
cf = ax.contourf(lons, lats, t2m_c.values,
                  levels=levels,
                  cmap="RdYlBu_r",       # 颜色：红-黄-蓝（反转，蓝冷红热）
                  transform=ccrs.PlateCarree(),
                  extend="both")          # 色标两端加箭头

# ============================================================
# 4. 加地理要素
# ============================================================
# 海岸线
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
# 国界线
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, linestyle=":")
# 河流（可选）
ax.add_feature(cfeature.RIVERS.with_scale("50m"), linewidth=0.4, color="steelblue")

# ============================================================
# 5. 加经纬度网格线和标签
# ============================================================
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
gl.top_labels = False      # 顶部不显示经度标签
gl.right_labels = False    # 右侧不显示纬度标签
gl.xlabel_style = {"size": 10}
gl.ylabel_style = {"size": 10}

# ============================================================
# 6. 加色标和标题
# ============================================================
cbar = plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("2m Temperature (℃)", fontsize=11)

title_time = str(t2m_c["valid_time"].values)[:16]
plt.title("ERA5 2m Temperature  %s" % title_time, fontsize=14, pad=12)

# ============================================================
# 7. 保存图片 + 显示
# ============================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "era5_t2m_map.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print("图片已保存:", out_path)

plt.show()

# 关闭文件
ds.close()
print("完成。")
