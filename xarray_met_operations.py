# -*- coding: utf-8 -*-
"""
气象常用运算代码总结（可直接运行）
场景：用模拟的 ERA5 风格三维数据 (time × lat × lon)，演示 NWP/气象处理里最常用的运算。
依赖：numpy, xarray
运行：python xarray_met_operations.py
"""

import numpy as np
import pandas as pd
import xarray as xr

# ========== 0. 造一份模拟数据：3 年逐日 × 中国区域 ==========
rng = np.random.default_rng(42)
lat = np.arange(20, 50, 2.0)          # 纬度 20~48°N
lon = np.arange(100, 130, 2.0)        # 经度 100~128°E
time = pd.date_range("2020-01-01", periods=365*3, freq="D")  # 3 年逐日

# 温度场：加一个年循环（夏季暖冬季冷）+ 随机扰动
t2m = 15 + 10*np.sin(2*np.pi*(np.arange(365*3)/365.25 - 0.25))   # 年循环
t2m = t2m[:, None, None] + rng.normal(0, 1.5, (len(time), len(lat), len(lon)))
t2m = xr.DataArray(t2m, dims=("time", "lat", "lon"),
                   coords={"time": time, "lat": lat, "lon": lon},
                   name="t2m", attrs={"units": "degC", "long_name": "2m temperature"})
u10 = xr.DataArray(rng.normal(5, 2, (len(time), len(lat), len(lon))),
                   dims=("time", "lat", "lon"), coords=t2m.coords, name="u10")
v10 = xr.DataArray(rng.normal(0, 2, (len(time), len(lat), len(lon))),
                   dims=("time", "lat", "lon"), coords=t2m.coords, name="v10")
print("模拟数据 t2m 形状:", t2m.shape, "(time×lat×lon)")

# ========== 1. 维度规约：沿某一轴做统计 ==========
print("\n--- 1. 维度规约 ---")
print("mean(dim='time') 后 dims:", t2m.mean(dim="time").dims)   # 沿时间平均 → lat,lon
print("std(dim='time')  后 dims:", t2m.std(dim="time").dims)
print("max(dim=['lat','lon']) 后 dims:", t2m.max(dim=["lat","lon"]).dims)  # → time
print("全区域中位数: %.2f ℃" % float(t2m.median()))

# ========== 2. 区域平均 / 时间序列 ==========
print("\n--- 2. 区域平均与时间序列 ---")
cn_ts = t2m.mean(dim=["lat", "lon"])           # 对空间求平均 → 一条时间曲线
print("区域平均时间序列 dims:", cn_ts.dims, "长度:", len(cn_ts))
print("区域平均后前5天温度:", np.round(cn_ts.values[:5], 2))

# ========== 3. groupby：按月/季节/年统计 ==========
print("\n--- 3. groupby 分组 ---")
monthly = t2m.groupby("time.month").mean("time")     # 月气候态
print("月气候态 dims:", monthly.dims, "| 各月平均温度(℃):", np.round(monthly.mean(dim=["lat","lon"]).values, 1))
seasonal = t2m.groupby("time.season").mean("time")  # 季节气候态
print("季节气候态 dims:", seasonal.dims, "季节:", list(seasonal.season.values))
# 注意：groupby("time.month") 是 xarray 对 time 坐标的特殊解析（库自带）

# ========== 4. resample：时间重采样（小时→日→月） ==========
print("\n--- 4. resample 时间重采样 ---")
# 日数据 → 月平均
monthly_res = t2m.resample(time="1ME").mean()
print("resample('1ME') 后 time 个数:", len(monthly_res.time), "(原", len(t2m.time), "天)")
# 日数据 → 年平均
annual = t2m.resample(time="1YE").mean()
print("年平均温度:", np.round(annual.mean(dim=["lat","lon"]).values, 1), "℃")

# ========== 5. 滚动平均：平滑时间序列 ==========
print("\n--- 5. 滚动平滑 ---")
smooth = cn_ts.rolling(time=31, center=True).mean()   # 31 天滑动平均
print("原始区域平均末值: %.2f, 31天滑动平均末值: %.2f" %
      (float(cn_ts[-1]), float(smooth[-1])))

# ========== 6. 差分 / 水平梯度 ==========
print("\n--- 6. 差分与梯度 ---")
dTdt = t2m.differentiate("time")        # dT/dt（时间变化率）
dTdlon = t2m.differentiate("lon")       # dT/dlon（水平梯度）
print("dT/dtime 形状:", dTdt.shape, "| dT/dlon 形状:", dTdlon.shape)

# ========== 7. 派生量：物理公式合成 ==========
print("\n--- 7. 派生量计算 ---")
wind = (u10**2 + v10**2) ** 0.5        # 风速合成（物理正确）
wind.attrs = {"units": "m s-1", "long_name": "10m wind speed"}
print("合成风速 wind 形状:", wind.shape, "平均风速: %.2f m/s" % float(wind.mean()))

# ========== 8. 气候态与距平 ==========
print("\n--- 8. 气候态与距平 ---")
# 逐日气候态：把 3 年同一天的温度平均 → 得到一个"365 天"的气候态
clim_day = t2m.groupby("time.dayofyear").mean("time")
print("逐日气候态 dims:", clim_day.dims)
# 距平：实际值 − 气候态（判断偏暖/偏冷）
anom = t2m.groupby("time.dayofyear") - clim_day
print("温度距平范围: %.2f ~ %.2f ℃" % (float(anom.min()), float(anom.max())))

# ========== 9. 写出结果 ==========
print("\n--- 9. 写出结果 ---")
out = xr.Dataset({"t2m_clim": clim_day.mean(dim=["lat","lon"]),
                  "wind_mean": wind.mean(dim=["time","lat","lon"])})
print("可写出 Dataset 变量:", list(out.data_vars))
# out.to_netcdf("met_result.nc")   # 需要 netcdf4
# cn_ts.to_dataframe().to_csv("ts.csv")   # 转 CSV

print("\n全部气象运算示例运行完成。")
