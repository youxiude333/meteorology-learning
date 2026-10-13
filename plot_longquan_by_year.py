# -*- coding: utf-8 -*-
"""
浙江龙泉 6月温度时间序列（按年分面，每年一个子图，x轴显示具体日期）
依赖：xarray, matplotlib, numpy, netcdf4
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
DATA_FILE = r"C:\Users\86137\Downloads\dbda254db6bf7a56ffa3e6a8797f34c.nc"
CITY_NAME = "浙江龙泉"
LAT = 28.08
LON = 119.14
# ============================================================

print("读取数据...")
ds = xr.open_dataset(DATA_FILE)
t2m_c = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest") - 273.15

actual_lat = float(t2m_c["latitude"].values)
actual_lon = float(t2m_c["longitude"].values)
times = t2m_c["valid_time"].values
temps = t2m_c.values

# 按年分组
years = sorted(set(int(str(t)[:4]) for t in times))
print("包含年份:", years)
print("每年时次数:", {y: sum(1 for t in times if int(str(t)[:4]) == y) for y in years})

# ========== 画图：每年一个子图 ==========
fig, axes = plt.subplots(len(years), 1, figsize=(14, 3.2 * len(years)), sharey=True)
if len(years) == 1:
    axes = [axes]

all_max = float(np.max(temps))
all_min = float(np.min(temps))

for idx, year in enumerate(years):
    ax = axes[idx]
    # 筛选这一年的数据
    mask = np.array([int(str(t)[:4]) == year for t in times])
    y_times = times[mask]
    y_temps = temps[mask]

    # 折线（每年内部是连续的6月数据，用折线能看出日变化）
    ax.plot(y_times, y_temps, color="#c0392b", linewidth=1.0, marker="o", markersize=2.5, alpha=0.8)

    # 这一年的最高/最低
    y_max = float(np.max(y_temps))
    y_min = float(np.min(y_temps))
    y_max_t = y_times[int(np.argmax(y_temps))]
    y_min_t = y_times[int(np.argmin(y_temps))]

    ax.scatter([y_max_t], [y_max], color="red", s=50, zorder=5)
    ax.scatter([y_min_t], [y_min], color="blue", s=50, zorder=5)

    # 子图标题：年份 + 最高最低
    ax.set_title("%d年  最高%.1f℃(%s)  最低%.1f℃(%s)" %
                 (year, y_max, str(y_max_t)[5:16], y_min, str(y_min_t)[5:16]),
                 fontsize=11, loc="left")

    # x 轴：显示月-日 时:分
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_ylabel("温度(℃)", fontsize=9)

# 总标题
fig.suptitle("%s (%.2f°N, %.2f°E)  6月逐6小时2米温度（按年分面）" %
              (CITY_NAME, actual_lat, actual_lon),
              fontsize=14, y=0.995)

axes[-1].set_xlabel("日期 (月-日 时:分)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.98])

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "longquan_t2m_by_year.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print("\n图片已保存:", out_path)

plt.show()
ds.close()
print("完成。")
