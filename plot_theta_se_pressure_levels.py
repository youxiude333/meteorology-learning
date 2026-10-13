# -*- coding: utf-8 -*-
"""
假相当位温 θse 计算与绘图
依赖：xarray, numpy, matplotlib, cartopy, netcdf4
（可选）metpy：用于对比验证，没装也能跑
数据：ERA5 气压层，需包含 Temperature(t) 和 Specific humidity(q)
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
DATA_FILE = r"C:\Users\86137\Downloads\era5_pressure_levels_with_q.nc"  # ← 改成你的文件
# ============================================================

print("读取数据:", DATA_FILE)
ds = xr.open_dataset(DATA_FILE)
print("变量:", list(ds.data_vars))
print("坐标:", list(ds.coords))

lev_name = "level" if "level" in ds.coords else "pressure_level"
time_name = "valid_time" if "valid_time" in ds.coords else "time"
print("气压坐标名:", lev_name, "时间坐标名:", time_name)
T = ds["t"]        # 温度, K
q = ds["q"]        # 比湿, kg/kg
P = ds[lev_name]   # 气压, hPa

print("\n温度形状:", T.shape, "比湿形状:", q.shape)
print("气压层:", P.values)

# ============================================================
# 1. 算位温 θ
# ============================================================
theta = T * (1000.0 / P) ** 0.286
theta.attrs = {"units": "K", "long_name": "Potential Temperature"}

# ============================================================
# 2. 算假相当位温 θse —— 简化公式
#    θse ≈ θ × exp(Lv × q / (Cp × T))
# ============================================================
Lv = 2.5e6    # 汽化潜热, J/kg
Cp = 1004.0   # 定压比热, J/(kg·K)
theta_se_simple = theta * np.exp(Lv * q / (Cp * T))
theta_se_simple.attrs = {"units": "K", "long_name": "Equivalent Potential Temp (simple)"}

# ============================================================
# 3. 算假相当位温 θse —— Bolton (1980) 精确公式
# ============================================================
def bolton_theta_se(T, P, q):
    """
    Bolton (1980) 假相当位温（业务标准）
    输入：T(K), P(hPa), q(kg/kg)  可以是 numpy 数组或 xarray DataArray
    输出：θse (K)
    """
    # 混合比 r = q/(1-q)
    r = q / (1.0 - q)

    # 饱和水汽压 (Tetens 公式), hPa
    es = 6.112 * np.exp(17.67 * (T - 273.15) / (T - 29.65))

    # 实际水汽压 e = r*P/(0.622+r)
    e = r * P / (0.622 + r)

    # 露点温度 Td (反算 Tetens 公式)
    ln_e_es = np.log(e / 6.112)
    Td = 243.5 * ln_e_es / (17.67 - ln_e_es) + 273.15

    # LCL 凝结温度 TL (Bolton 公式)
    TL = 1.0 / (1.0 / (Td - 56.0) + np.log(T / Td) / 800.0) + 56.0

    # Bolton 1980 假相当位温 (eq.43)
    theta_se = T * (1000.0 / P) ** (0.2854 * (1.0 - 0.28 * r)) * \
               np.exp((3.376 / TL - 0.00254) * r * (1.0 + 0.81 * r))

    return theta_se

theta_se_bolton = xr.apply_ufunc(
    bolton_theta_se, T, P, q,
    dask="allowed", keep_attrs=True
)
theta_se_bolton.attrs = {"units": "K", "long_name": "Equivalent Potential Temp (Bolton 1980)"}

# ============================================================
# 4. 三种算法对比（取一个点验证）
# ============================================================
print("\n" + "=" * 60)
print("三种算法对比（取龙泉站、第一个时次、各层）")
print("=" * 60)
t_point = T.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})
q_point = q.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})
theta_point = theta.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})
ths_point = theta_se_simple.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})
thb_point = theta_se_bolton.sel(latitude=28.08, longitude=119.14, method="nearest").isel({time_name: 0})

print("%-6s %-8s %-8s %-8s %-10s %-10s %-8s" %
      ("P(hPa)", "T(K)", "q(g/kg)", "θ(K)", "θse简化", "θseBolton", "差值(K)"))
print("-" * 70)
for i, lev in enumerate(P.values):
    print("%-6.0f %-8.1f %-8.2f %-8.1f %-10.1f %-10.1f %-8.1f" %
          (lev, float(t_point[i]), float(q_point[i])*1000,
           float(theta_point[i]), float(ths_point[i]),
           float(thb_point[i]), float(thb_point[i] - theta_point[i])))

print("\n观察：θse(Bolton) 比 θ 大，差值随高度减小（高层水汽少，潜热贡献小）")
print("      简化公式和 Bolton 在低层差异较大，高层接近")

# ============================================================
# 图1：500hPa 假相当位温空间图
# ============================================================
print("\n画图1：500hPa 假相当位温空间图...")
thb_500 = theta_se_bolton.sel({lev_name: 500}, method="nearest").isel({time_name: 0})

fig1 = plt.figure(figsize=(12, 8))
ax1 = plt.axes(projection=ccrs.PlateCarree())
ax1.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

lons = ds["longitude"].values
lats = ds["latitude"].values
levels = np.arange(280, 360, 5)
cf = ax1.contourf(lons, lats, thb_500.values, levels=levels,
                   cmap="RdYlBu_r", transform=ccrs.PlateCarree(), extend="both")
ax1.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8)
ax1.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6, linestyle=":")

gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5)
gl.top_labels = False
gl.right_labels = False

cbar = plt.colorbar(cf, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label("假相当位温 θse (K)", fontsize=11)
time_str = str(thb_500[time_name].values)[:16] if time_name in thb_500.coords else ""
plt.title("ERA5 500hPa 假相当位温 (Bolton 1980)  %s" % time_str, fontsize=14, pad=12)

out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_se_500hPa_map.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print("  已保存:", out1)

# ============================================================
# 图2：龙泉站 θ vs θse 垂直廓线对比（判断对流稳定度）
# ============================================================
print("\n画图2：龙泉站 θ vs θse 垂直廓线...")
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

# 左图：位温廓线
ax2a.plot(theta_point.values, P.values, "b-o", markersize=4, label="位温 θ")
ax2a.plot(thb_point.values, P.values, "r-s", markersize=4, label="假相当位温 θse")
ax2a.set_xlabel("温度 (K)", fontsize=11)
ax2a.set_ylabel("气压 (hPa)", fontsize=11)
ax2a.set_title("θ vs θse 垂直廓线", fontsize=13)
ax2a.invert_yaxis()
ax2a.legend(fontsize=10)
ax2a.grid(True, linestyle=":", alpha=0.5)

# 右图：θse 随高度变化率（判断对流不稳定）
dthse_dp = np.diff(thb_point.values) / np.diff(P.values)  # dθse/dP
# P 减小 = 高度增加，所以 dθse/dP < 0 表示 θse 随高度增加（稳定）
# dθse/dP > 0 表示 θse 随高度减小（对流不稳定）
mid_P = (P.values[:-1] + P.values[1:]) / 2
colors = ["red" if x > 0 else "blue" for x in dthse_dp]
ax2b.barh(mid_P, dthse_dp * 1000, height=30, color=colors, alpha=0.7)
ax2b.axvline(0, color="black", linewidth=0.8)
ax2b.set_xlabel("dθse/dP × 1000 (K/hPa)", fontsize=11)
ax2b.set_title("θse 随高度变化率\n(红=对流不稳定层, 蓝=稳定层)", fontsize=13)
ax2b.grid(True, linestyle=":", alpha=0.5)

fig2.suptitle("龙泉站 (28.08°N, 119.14°E)  位温与假相当位温对比", fontsize=14, y=1.02)
plt.tight_layout()
out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_se_profile_longquan.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print("  已保存:", out2)

# ============================================================
# 图3：沿 110°E 纬度-高度 θse 剖面图
# ============================================================
print("\n画图3：沿110°E 纬度-高度 θse 剖面图...")
thb_section = theta_se_bolton.sel(longitude=110, method="nearest").isel({time_name: 0})

fig3 = plt.figure(figsize=(12, 7))
ax3 = plt.axes()

levels_sec = np.arange(280, 360, 5)
cf3 = ax3.contourf(lats, P.values, thb_section.values,
                    levels=levels_sec, cmap="RdYlBu_r", extend="both")
# 叠加 θse 等值线，看梯度
cs3 = ax3.contour(lats, P.values, thb_section.values,
                   levels=levels_sec, colors="black", linewidths=0.4, alpha=0.5)

ax3.invert_yaxis()
ax3.set_xlabel("纬度 (°N)", fontsize=11)
ax3.set_ylabel("气压 (hPa)", fontsize=11)
ax3.set_title("沿 110°E 假相当位温纬度-高度剖面图  %s" % time_str, fontsize=13, pad=12)
ax3.grid(True, linestyle=":", alpha=0.4)

cbar3 = plt.colorbar(cf3, ax=ax3, shrink=0.8, pad=0.02)
cbar3.set_label("假相当位温 θse (K)", fontsize=11)

plt.tight_layout()
out3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "theta_se_lat_height_section.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
print("  已保存:", out3)

plt.show()
ds.close()
print("\n全部完成。3张图已保存到脚本同目录。")
print("\n学习要点：")
print("1. θse 永远 ≥ θ，差值 = 潜热贡献，水汽越多差值越大")
print("2. θse 随高度减小的层 = 对流不稳定层，容易产生雷暴")
print("3. θse 高值区 = 暖湿气流，低值区 = 干冷空气，密集带 = 锋面")
