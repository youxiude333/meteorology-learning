# -*- coding: utf-8 -*-
"""
【假相当位温 θse 计算与绘图】
功能：读取 ERA5 气压层数据（温度+比湿），计算假相当位温，画3张图
依赖：xarray, numpy, matplotlib, cartopy, netcdf4
运行前：把 DATA_FILE 改成你下载的 nc 文件路径
输出：3张图（500hPa空间图、龙泉站垂直廓线、110°E纬度高度剖面图）
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 【第1步】读取数据
# ============================================================
DATA_FILE = r"C:\Users\86137\Downloads\你的文件名.nc"  # ← 改成你的文件

print("正在读取数据：", DATA_FILE)
ds = xr.open_dataset(DATA_FILE)
print("变量：", list(ds.data_vars))
print("坐标：", list(ds.coords))

# 自动识别坐标名（ERA5 可能是 time 或 valid_time）
time_name = "valid_time" if "valid_time" in ds.coords else "time"
lev_name = "level" if "level" in ds.coords else "pressure_level"
print("时间坐标名：", time_name, " 气压层坐标名：", lev_name)

# 读取温度（K）和比湿（kg/kg）
T = ds["t"]            # 温度，单位 K
q = ds["q"]            # 比湿，单位 kg/kg
P = ds[lev_name]       # 气压，单位 hPa

print("\n温度形状：", T.shape, " 维度：", T.dims)
print("比湿形状：", q.shape, " 维度：", q.dims)

# ============================================================
# 【第2步】计算位温 θ（基础，先算出来）
# 公式：θ = T × (1000/P)^0.286
# ============================================================
theta = T * (1000.0 / P) ** 0.286
theta.attrs = {"units": "K", "long_name": "Potential Temperature"}
print("\n位温计算完成，形状：", theta.shape)

# ============================================================
# 【第3步】计算假相当位温 θse
# ============================================================

# --- 方法A：简化公式（好理解，误差约1-2K）---
# 公式：θse ≈ θ × exp(Lv × q / (Cp × T))
# 物理意义：把水汽凝结潜热全部加到位温上
Lv = 2.5e6       # 水的汽化潜热，J/kg
Cp = 1004.0      # 干空气定压比热，J/(kg·K)

theta_se_simple = theta * np.exp(Lv * q / (Cp * T))
theta_se_simple.attrs = {"units": "K", "long_name": "Equivalent Potential Temp (simple)"}

# --- 方法B：Bolton 1980 精确公式（业务标准，推荐用这个）---
def calc_theta_se_bolton(T, q, P):
    """
    Bolton 1980 假相当位温精确公式
    输入：T(K), q(kg/kg), P(hPa)
    输出：θse(K)
    """
    # 1. 计算露点温度 Td（从比湿反推）
    e = q * P / (0.622 + q)
    Td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112)) + 273.15

    # 2. 计算凝结温度 Tl（抬升凝结高度的温度）
    Tl = 1.0 / (1.0 / (Td - 56.0) + np.log(T / Td) / 800.0) + 56.0

    # 3. 计算混合比 rv
    rv = q / (1.0 - q)

    # 4. 位温 θ
    theta = T * (1000.0 / P) ** 0.286

    # 5. Bolton 公式
    theta_se = theta * np.exp((3.376 / Tl - 0.00254) * rv * (1.0 + 0.81 * rv))
    return theta_se

theta_se_bolton = calc_theta_se_bolton(T, q, P)
theta_se_bolton.attrs = {"units": "K", "long_name": "Equivalent Potential Temp (Bolton 1980)"}

# 打印两种方法的对比（龙泉站附近）
lat_lq, lon_lq = 28.08, 119.14  # 龙泉市坐标
t_lq = theta_se_simple.sel(latitude=lat_lq, longitude=lon_lq, method="nearest").isel({time_name: 0})
b_lq = theta_se_bolton.sel(latitude=lat_lq, longitude=lon_lq, method="nearest").isel({time_name: 0})
print("\n=== 龙泉站 两种方法对比（第一个时次）===")
for i, lev in enumerate(P.values):
    print(f"  {lev:6.0f} hPa: 简化={float(t_lq.isel({lev_name: i})):.1f}K  Bolton={float(b_lq.isel({lev_name: i})):.1f}K")

# 后面统一用 Bolton 精确版
theta_se = theta_se_bolton

# ============================================================
# 【第4步】画图1：500hPa 假相当位温空间图
# ============================================================
print("\n正在画图1：500hPa 假相当位温空间图...")
theta_se_500 = theta_se.sel({lev_name: 500}, method="nearest").isel({time_name: 0})

fig1 = plt.figure(figsize=(12, 8))
ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax1.set_extent([73, 135, 4, 54], crs=ccrs.PlateCarree())

contour = ax1.contourf(
    theta_se_500.longitude, theta_se_500.latitude, theta_se_500.values,
    levels=np.arange(280, 360, 5), cmap="RdYlBu_r",
    transform=ccrs.PlateCarree(), extend="both"
)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
cbar1 = plt.colorbar(contour, ax=ax1, shrink=0.8, pad=0.02)
cbar1.set_label("假相当位温 θse (K)", fontsize=12)
ax1.set_title("500hPa 假相当位温空间分布（Bolton 1980）", fontsize=14)
plt.tight_layout()
plt.savefig("theta_se_500hPa_map.png", dpi=150, bbox_inches="tight")
print("  已保存：theta_se_500hPa_map.png")

# ============================================================
# 【第5步】画图2：龙泉站 位温 vs 假相当位温 垂直廓线
# ============================================================
print("正在画图2：龙泉站位温与假相当位温垂直廓线...")
theta_lq = theta.sel(latitude=lat_lq, longitude=lon_lq, method="nearest").isel({time_name: 0})
theta_se_lq = theta_se.sel(latitude=lat_lq, longitude=lon_lq, method="nearest").isel({time_name: 0})

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# 左图：温度廓线
T_lq = T.sel(latitude=lat_lq, longitude=lon_lq, method="nearest").isel({time_name: 0})
ax2a.plot(T_lq.values - 273.15, P.values, "b-o", markersize=4)
ax2a.set_ylim(1000, 100)
ax2a.set_xlabel("温度 (°C)", fontsize=12)
ax2a.set_ylabel("气压 (hPa)", fontsize=12)
ax2a.set_title("龙泉站 温度垂直廓线", fontsize=13)
ax2a.grid(True, alpha=0.3)

# 右图：位温 vs 假相当位温
ax2b.plot(theta_lq.values, P.values, "b-o", markersize=4, label="位温 θ（干）")
ax2b.plot(theta_se_lq.values, P.values, "r-s", markersize=4, label="假相当位温 θse（湿）")
ax2b.set_xlabel("温度 (K)", fontsize=12)
ax2b.set_title("龙泉站 位温 vs 假相当位温", fontsize=13)
ax2b.legend(fontsize=11)
ax2b.grid(True, alpha=0.3)

# 标注对流不稳定层（θse 随高度减小）
dtheta_se_dp = np.diff(theta_se_lq.values) / np.diff(P.values)
unstable_levels = np.where(dtheta_se_dp > 0)[0]
if len(unstable_levels) > 0:
    for idx in unstable_levels:
        ax2b.axhspan(P.values[idx], P.values[idx+1], alpha=0.15, color="red")
    ax2b.text(0.02, 0.05, "红色区域=对流不稳定\n(θse随高度减小)", transform=ax2b.transAxes,
              fontsize=10, color="red", verticalalignment="bottom")

plt.tight_layout()
plt.savefig("theta_se_profile_longquan.png", dpi=150, bbox_inches="tight")
print("  已保存：theta_se_profile_longquan.png")

# ============================================================
# 【第6步】画图3：沿110°E 假相当位温 纬度-高度剖面图
# ============================================================
print("正在画图3：沿110°E 假相当位温纬度-高度剖面图...")
theta_se_cross = theta_se.sel(longitude=110.0, method="nearest").isel({time_name: 0})

fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(1, 1, 1)

contour3 = ax3.contourf(
    theta_se_cross.latitude, theta_se_cross[lev_name], theta_se_cross.values,
    levels=np.arange(280, 370, 5), cmap="RdYlBu_r", extend="both"
)
ax3.set_ylim(1000, 100)
cbar3 = plt.colorbar(contour3, ax=ax3, shrink=0.8, pad=0.02)
cbar3.set_label("假相当位温 θse (K)", fontsize=12)
ax3.set_xlabel("纬度 (°N)", fontsize=12)
ax3.set_ylabel("气压 (hPa)", fontsize=12)
ax3.set_title("沿 110°E 假相当位温 纬度-高度剖面图", fontsize=14)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("theta_se_lat_height_section.png", dpi=150, bbox_inches="tight")
print("  已保存：theta_se_lat_height_section.png")

# ============================================================
# 【第7步】输出关键结论
# ============================================================
print("\n" + "="*60)
print("【关键结论】")
print("="*60)
print(f"1. 500hPa θse 范围：{float(theta_se_500.min()):.1f} ~ {float(theta_se_500.max()):.1f} K")
print(f"2. 龙泉站地面(1000hPa) θse = {float(theta_se_lq.isel({lev_name: 0})):.1f} K")
print(f"3. 龙泉站 θse 与 θ 差值（低层）= {float(theta_se_lq.isel({lev_name: 0}) - theta_lq.isel({lev_name: 0})):.1f} K")
print("   （差值越大，说明低层水汽越充沛）")
print(f"4. 龙泉站 θse 随高度变化：", end="")
if float(theta_se_lq.isel({lev_name: -1})) > float(theta_se_lq.isel({lev_name: 0})):
    print("整体随高度增加 → 对流稳定")
else:
    print("存在随高度减小层 → 可能对流不稳定")
print("\n全部完成！3张图已保存到脚本同目录。")
plt.show()
