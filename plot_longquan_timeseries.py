# -*- coding: utf-8 -*-
"""
浙江省龙泉市 2米温度时间序列图（用全部600个时次）
依赖：xarray, matplotlib, numpy, netcdf4
运行前：把 DATA_FILE 改成你的 ERA5 文件路径
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# ★ 改成你的 ERA5 文件路径 ★
# ============================================================
DATA_FILE = r"C:\Users\86137\Downloads\dbda254db6bf7a56ffa3e6a8797f34c.nc"

# 龙泉市城区中心坐标（用户自定义，可改成其他城市）
CITY_NAME = "浙江龙泉"
LAT = 28.08   # 北纬
LON = 119.14  # 东经

# ============================================================
# 1. 读取数据
# ============================================================
print("读取数据:", DATA_FILE)
ds = xr.open_dataset(DATA_FILE)

# 取龙泉市最近格点的 t2m（注意：ERA5 的坐标叫 latitude/longitude，时间叫 valid_time）
t2m_point = ds["t2m"].sel(latitude=LAT, longitude=LON, method="nearest")

# 转成摄氏度
t2m_c = t2m_point - 273.15

# 看看实际选中的格点坐标
actual_lat = float(t2m_c["latitude"].values)
actual_lon = float(t2m_c["longitude"].values)
print("实际选中格点: lat=%.2f, lon=%.2f" % (actual_lat, actual_lon))
print("时次数:", len(t2m_c))
print("时间范围:", str(t2m_c["valid_time"].values[0])[:10], "~", str(t2m_c["valid_time"].values[-1])[:10])

# ============================================================
# 2. 简单统计
# ============================================================
times = t2m_c["valid_time"].values
temps = t2m_c.values

mean_t = float(np.mean(temps))
max_t = float(np.max(temps))
min_t = float(np.min(temps))
max_idx = int(np.argmax(temps))
min_idx = int(np.argmin(temps))

print("\n=== %s 温度统计 ===" % CITY_NAME)
print("平均温度: %.2f ℃" % mean_t)
print("最高温度: %.2f ℃  (%s)" % (max_t, str(times[max_idx])[:10]))
print("最低温度: %.2f ℃  (%s)" % (min_t, str(times[min_idx])[:10]))

# ============================================================
# 3. 画图
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

# 折线图
ax.plot(times, temps, color="#c0392b", linewidth=0.8, alpha=0.8)

# 平均线
ax.axhline(mean_t, color="gray", linestyle="--", linewidth=1, label="平均 %.1f℃" % mean_t)

# 标注最高/最低点
ax.scatter([times[max_idx]], [max_t], color="red", s=60, zorder=5)
ax.annotate("最高 %.1f℃\n%s" % (max_t, str(times[max_idx])[:10]),
            xy=(times[max_idx], max_t),
            xytext=(15, 10), textcoords="offset points",
            fontsize=9, color="red",
            arrowprops=dict(arrowstyle="->", color="red"))

ax.scatter([times[min_idx]], [min_t], color="blue", s=60, zorder=5)
ax.annotate("最低 %.1f℃\n%s" % (min_t, str(times[min_idx])[:10]),
            xy=(times[min_idx], min_t),
            xytext=(15, -25), textcoords="offset points",
            fontsize=9, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue"))

# 坐标轴设置
ax.set_xlabel("Time", fontsize=11)
ax.set_ylabel("2m Temperature (℃)", fontsize=11)
ax.set_title("%s (%.2f°N, %.2f°E)  2m Temperature Time Series" %
             (CITY_NAME, actual_lat, actual_lon), fontsize=13, pad=12)

# x 轴时间格式：每年标一个
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

# 网格
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="upper right")

# 温度范围留点边
ax.set_ylim(min_t - 3, max_t + 3)

plt.tight_layout()

# ============================================================
# 4. 保存 + 显示
# ============================================================
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "longquan_t2m_timeseries.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print("\n图片已保存:", out_path)

plt.show()

ds.close()
print("完成。")
