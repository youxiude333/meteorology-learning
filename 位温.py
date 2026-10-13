# -*- coding: utf-8 -*-
"""
ERA5 气压层数据：位温计算与绘图
依赖：xarray, numpy, matplotlib, cartopy, netcdf4
运行前：把 DATA_FILE 改成你下载的气压层 nc 文件路径
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
DATA_FILE = r"C:\Users\86137\Downloads\f5d42a728cfec60420e14219c84bc534.nc" # ← 改成你的文件
# ============================================================

print("读取数据:", DATA_FILE)
ds = xr.open_dataset(DATA_FILE)
print("变量:", list(ds.data_vars))
print("坐标:", list(ds.coords))
print("气压层:", ds["level"].values if "level" in ds.coords else "看上面坐标名")

# 气压层坐标名可能是 level 或 pressure_level，自动识别
lev_name = "level" if "level" in ds.coords else "pressure_level"
print("气压坐标名:", lev_name)

# 时间坐标名可能是 time 或 valid_time，自动识别
time_name = "valid_time" if "valid_time" in ds.coords else "time"
print("时间坐标名:", time_name)

# 取温度（变量名 t，单位 K）
T = ds["t"]
print("\n温度形状:", T.shape, "维度:", T.dims)

# ============================================================
# 计算位温：θ = T × (1000/P)^0.286
# P 就是气压层坐标，单位 hPa
# ============================================================
P = ds[lev_name]  # 气压坐标，hPa
theta = T * (1000.0 / P) ** 0.286
theta.attrs = {"units": "K", "long_name": "Potential Temperature"}
print("\n位温计算完成，形状:", theta.shape)
print("各层平均位温（验证：高度越高位温越大）:")
for i, lev in enumerate(P.values):
    print("  %4.0f hPa: θ = %.2f K" % (lev, float(theta.isel({lev_name: i}).mean())))

# ============================================================
# 图1：500hPa 位温空间图
# ============================================================
print("\n画图1：500hPa 位温空间图...")
theta_500 = theta.sel({lev_name: 500}, method="nearest").isel({time_name: 0})  # 取第一个时次

fig1 = plt.figure(figsize=(12, 8))
ax1 = plt.axes(projection=ccrs.PlateCarree())
ax1.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

lons = ds["longitude"].values
lats = ds["latitude"].values
levels = np.arange(260, 340, 5)
cf = ax1.contourf(lons, lats, theta_500.values, levels=levels,
                   cmap="RdYlBu_r", transform=ccrs.PlateCarree(), extend="both")
ax1.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
ax1.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, linestyle=":")

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = plt.colorbar(cf, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label("位温 θ (K)", fontsize=11)
time_str = str(theta_500[time_name].values)[:16] if time_name in theta_500.coords else ""
plt.title("ERA5 500hPa 位温  %s" % time_str, fontsize=14, pad=12)

out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_500hPa_map.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print("  已保存:", out1)

# ============================================================
# 图2：龙泉站位温垂直廓线（判断稳定度）
# ============================================================
print("\n画图2：龙泉站位温垂直廓线...")
theta_lq = theta.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})
T_lq = T.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

# 左图：温度廓线
ax2a.plot(T_lq.values - 273.15, P.values, "b-o", markersize=4)
ax2a.set_xlabel("温度 (℃)", fontsize=11)
ax2a.set_ylabel("气压 (hPa)", fontsize=11)
ax2a.set_title("温度垂直廓线", fontsize=13)
ax2a.invert_yaxis()  # 气压从大到小（从地面到高空）
ax2a.grid(True, linestyle=":", alpha=0.5)

# 右图：位温廓线
ax2b.plot(theta_lq.values, P.values, "r-o", markersize=4)
ax2b.set_xlabel("位温 θ (K)", fontsize=11)
ax2b.set_title("位温垂直廓线", fontsize=13)
ax2b.grid(True, linestyle=":", alpha=0.5)

# 判断稳定度：位温随高度增加=稳定，减小=不稳定
# 注意气压减小=高度增加，所以看 P 减小时 θ 的变化
theta_vals = theta_lq.values
P_vals = P.values
dtheta_dz = np.diff(theta_vals) / np.diff(P_vals)  # dθ/dP，因为P减小=高度增加，所以符号要反
# 位温随高度增加 = dθ/dP < 0（P减小θ增大）
stable = np.all(dtheta_dz < 0)
fig2.suptitle("龙泉站 (28.08°N, 119.14°E)  位温随高度%s → %s" %
              ("增加" if stable else "存在减小层", "整体稳定" if stable else "存在不稳定层"),
              fontsize=14, y=1.02)

plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_profile_longquan.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print("  已保存:", out2)

# ============================================================
# 图3：沿 110°E 的纬度-高度位温剖面图
# ============================================================
print("\n画图3：沿110°E 纬度-高度位温剖面图...")
theta_section = theta.sel(longitude=110, method="nearest").isel({time_name: 0})

fig3 = plt.figure(figsize=(12, 7))
ax3 = plt.axes()

lats_sec = ds["latitude"].values
levs_sec = P.values
# 填色图：x=纬度，y=气压（高度）
levels_sec = np.arange(260, 350, 5)
cf3 = ax3.contourf(lats_sec, levs_sec, theta_section.values,
                    levels=levels_sec, cmap="RdYlBu_r", extend="both")

ax3.invert_yaxis()
ax3.set_xlabel("纬度 (°N)", fontsize=11)
ax3.set_ylabel("气压 (hPa)", fontsize=11)
ax3.set_title("沿 110°E 位温纬度-高度剖面图  %s" % time_str, fontsize=13, pad=12)
ax3.grid(True, linestyle=":", alpha=0.4)

cbar3 = plt.colorbar(cf3, ax=ax3, shrink=0.8, pad=0.02)
cbar3.set_label("位温 θ (K)", fontsize=11)

plt.tight_layout()
out3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_lat_height_section.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print("  已保存:", out3)

plt.show()
ds.close()
print("\n全部完成。3张图已保存到脚本同目录。")
