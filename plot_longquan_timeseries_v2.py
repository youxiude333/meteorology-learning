# -*- coding: utf-8 -*-
"""
浙江省龙泉市 2米温度时间序列图（改进版：散点图 + 中文 + 完整日期对照表）
依赖：xarray, matplotlib, numpy, netcdf4
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ========== 解决中文显示 ==========
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
DATA_FILE = r"C:\Users\86137\Downloads\dbda254db6bf7a56ffa3e6a8797f34c.nc"
CITY_NAME = "浙江龙泉"
LAT = 28.08
LON = 119.14
# ============================================================

print("读取数据:", DATA_FILE)
ds = xr.open_dataset(DATA_FILE)

t2m_point = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest")
t2m_c = t2m_point - 273.15

actual_lat = float(t2m_c["latitude"].values)
actual_lon = float(t2m_c["longitude"].values)
times = t2m_c["valid_time"].values
temps = t2m_c.values

print("实际选中格点: lat=%.2f, lon=%.2f" % (actual_lat, actual_lon))
print("时次数:", len(temps))
print("时间范围: %s ~ %s" % (str(times[0])[:10], str(times[-1])[:10]))

# ========== 打印时间分布（看看数据是不是连续的）==========
print("\n=== 数据时间分布（按年统计时次数）===")
years = [int(str(t)[:4]) for t in times]
for y in sorted(set(years)):
    cnt = years.count(y)
    print("  %d年: %d 个时次" % (y, cnt))

# ========== 统计 ==========
mean_t = float(np.mean(temps))
max_t = float(np.max(temps))
min_t = float(np.min(temps))
max_idx = int(np.argmax(temps))
min_idx = int(np.argmin(temps))

print("\n=== %s 温度统计 ===" % CITY_NAME)
print("平均温度: %.2f ℃" % mean_t)
print("最高温度: %.2f ℃  (%s)" % (max_t, str(times[max_idx])[:10]))
print("最低温度: %.2f ℃  (%s)" % (min_t, str(times[min_idx])[:10]))

# ========== 打印完整日期-温度对照表 ==========
print("\n=== 完整日期-温度对照表（共%d条）===" % len(temps))
print("日期             温度(℃)")
print("-" * 30)
for t, temp in zip(times, temps):
    print("%s    %.2f" % (str(t)[:16], temp))

# ========== 画图（散点图，不连线）==========
fig, ax = plt.subplots(figsize=(14, 6))

# 散点：每个点一个时次，不连线
ax.scatter(times, temps, s=12, color="#c0392b", alpha=0.7, zorder=3)

# 平均线
ax.axhline(mean_t, color="gray", linestyle="--", linewidth=1, label="平均 %.1f℃" % mean_t)

# 最高/最低点标注
ax.scatter([times[max_idx]], [max_t], color="red", s=80, edgecolor="black", zorder=5)
ax.annotate("最高 %.1f℃\n%s" % (max_t, str(times[max_idx])[:10]),
            xy=(times[max_idx], max_t),
            xytext=(20, 10), textcoords="offset points",
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

ax.scatter([times[min_idx]], [min_t], color="blue", s=80, edgecolor="black", zorder=5)
ax.annotate("最低 %.1f℃\n%s" % (min_t, str(times[min_idx])[:10]),
            xy=(times[min_idx], min_t),
            xytext=(20, -30), textcoords="offset points",
            fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"))

ax.set_xlabel("时间", fontsize=11)
ax.set_ylabel("2米温度 (℃)", fontsize=11)
ax.set_title("%s (%.2f°N, %.2f°E)  2米温度时间序列（共%d个时次）" %
             (CITY_NAME, actual_lat, actual_lon, len(temps)),
             fontsize=13, pad=12)

# x 轴：每年标年份，每季度标小刻度
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right")
ax.set_ylim(min_t - 3, max_t + 3)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "longquan_t2m_timeseries_v2.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print("\n图片已保存:", out_path)

plt.show()
ds.close()
print("完成。")
