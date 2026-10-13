# -*- coding: utf-8 -*-
"""
Python / NumPy / Xarray 语法速查表（可直接运行版）
运行：python syntax_cheatsheet.py
每个示例都打印输出，跑一遍就能看到每个函数的实际效果。
依赖：numpy, xarray, pandas
"""

import numpy as np
import xarray as xr
import pandas as pd

print("=" * 60)
print("一、Python 自带语法")
print("=" * 60)

# 1.1 打印与格式化
print("\n--- 1.1 打印 ---")
print("hello")                          # 普通打印
print("a =", 3.14, "b =", 2.71)      # 打印多个值
print("保留2位小数: %.2f" % 3.14159) # 格式化输出
print("保留3位小数:", round(3.14159, 3))

# 1.2 类型转换与查询
print("\n--- 1.2 类型与长度 ---")
arr = [0, 1, 2, 3]
print("长度 len:", len(arr))
print("类型 type:", type(np.array(arr)))
print("转浮点数 float:", float("3.14"))
print("绝对值 abs:", abs(-5))

# 1.3 数据容器
print("\n--- 1.3 数据容器 ---")
lst = [10, 20, 30]                    # 列表
dct = {"name": "t2m", "unit": "K"}    # 字典
tpl = ("time", "lat", "lon")           # 元组
print("列表:", lst)
print("字典:", dct, "取值:", dct["name"])
print("元组:", tpl)

# 1.4 切片
print("\n--- 1.4 切片 ---")
data = [0, 1, 2, 3, 4, 5, 6, 7]
print("原数组:", data)
print("第0个:", data[0])
print("前4个:", data[:4])
print("第1到3个:", data[1:4])
print("倒序:", data[::-1])

# 1.5 条件判断
print("\n--- 1.5 条件 ---")
x = 5
if x > 0:
    print("x 是正数")
else:
    print("x 是负数或零")

print("=" * 60)
print("二、NumPy 函数")
print("=" * 60)

# 2.1 生成数组
print("\n--- 2.1 生成数组 ---")
print("arange(4):", np.arange(4))                     # [0,1,2,3]
print("arange(1,5):", np.arange(1, 5))               # [1,2,3,4]
print("arange(0,10,2):", np.arange(0, 10, 2))         # [0,2,4,6,8]
print("linspace(20,50,4):", np.linspace(20, 50, 4))  # [20,30,40,50]
print("zeros(4):", np.zeros(4))                       # [0,0,0,0]
print("ones(4):", np.ones(4))                         # [1,1,1,1]

# 2.2 随机数
print("\n--- 2.2 随机数 ---")
rng = np.random.default_rng(42)
print("randn(3):", np.round(np.random.randn(3), 3))  # 标准正态
print("rand(3):", np.round(np.random.rand(3), 3))     # 均匀[0,1)
print("normal(5,2,size=3):", np.round(rng.normal(5, 2, size=3), 3))

# 2.3 统计函数
print("\n--- 2.3 统计 ---")
test = np.array([1, 2, 3, 4, 5])
print("数组:", test)
print("均值 mean:", np.mean(test))
print("标准差 std:", round(np.std(test), 3))
print("最小值 min:", np.min(test))
print("最大值 max:", np.max(test))
print("中位数 median:", np.median(test))
print("求和 sum:", np.sum(test))

# 2.4 数组运算
print("\n--- 2.4 数组运算 ---")
a = np.array([1, 2, 3, 4])
print("原数组:", a)
print("乘2:", a * 2)
print("加10:", a + 10)
print("平方:", a ** 2)
print("开方:", np.round(a ** 0.5, 3))
print("保留2位小数:", np.round(np.array([3.14159, 2.71828]), 2))
print("近似相等判断:", np.allclose([1, 2], [1.0001, 2.0001]))

print("=" * 60)
print("三、Xarray 方法与属性")
print("=" * 60)

# 3.1 构造 DataArray
print("\n--- 3.1 构造 DataArray ---")
time = np.arange(0, 6)
lat = np.linspace(20, 50, 3)
lon = np.linspace(100, 130, 4)
data = np.random.randn(len(time), len(lat), len(lon))

t2m = xr.DataArray(
    data,
    dims=("time", "lat", "lon"),
    coords={"time": time, "lat": lat, "lon": lon},
    name="t2m",
    attrs={"units": "K", "long_name": "2m temperature"},
)
print("DataArray 创建成功，shape:", t2m.shape)

# 3.2 查看结构
print("\n--- 3.2 查看结构 ---")
print("dims 维度名:", t2m.dims)
print("shape 形状:", t2m.shape)
print("coords 坐标:", list(t2m.coords))
print("attrs 属性:", t2m.attrs)
print("name 变量名:", t2m.name)
print("values 类型:", type(t2m.values))

# 3.3 索引与切片
print("\n--- 3.3 索引 ---")
print("isel(time=0) shape:", t2m.isel(time=0).shape)
print("sel(lat=30) 最近纬度:", t2m.sel(lat=30, method="nearest").coords["lat"].values)
print("sel(lat=slice(20,40)) lat数:", len(t2m.sel(lat=slice(20, 40)).lat))

# 3.4 维度规约
print("\n--- 3.4 维度规约 ---")
print("mean(dim='time') 后 dims:", t2m.mean(dim="time").dims)
print("std(dim='time') 后 dims:", t2m.std(dim="time").dims)
print("max(dim=['lat','lon']) 后 dims:", t2m.max(dim=["lat", "lon"]).dims)
print("全区域中位数:", round(float(t2m.median()), 3))

# 3.5 分组与重采样
print("\n--- 3.5 分组与重采样 ---")
# 造一个带日期的时间序列
dates = pd.date_range("2020-01-01", periods=36, freq="D")
ts_data = np.random.randn(36)
ts = xr.DataArray(ts_data, dims=("time",), coords={"time": dates}, name="t2m")

monthly = ts.resample(time="1ME").mean()
print("resample 月平均后长度:", len(monthly), "(原36天)")
print("rolling 7天平均后末值:", round(float(ts.rolling(time=7, center=True).mean()[-1]), 3))

# 3.6 坐标与属性操作
print("\n--- 3.6 坐标与属性 ---")
t2m_copy = t2m.copy()
t2m_renamed = t2m_copy.rename({"lat": "latitude"})
print("rename 后 dims:", t2m_renamed.dims)
t2m_with_attr = t2m.assign_coords(source="ERA5")
print("assign_coords 后 coords:", list(t2m_with_attr.coords))

# 3.7 合并与拼接
print("\n--- 3.7 合并与拼接 ---")
ds_a = xr.Dataset({"a": (("time",), np.arange(4))}, coords={"time": np.arange(4)})
ds_b = xr.Dataset({"b": (("time",), np.arange(4) * 2)}, coords={"time": np.arange(4)})
merged = xr.merge([ds_a, ds_b])
print("merge 后变量:", list(merged.data_vars))

ds_c = xr.Dataset({"a": (("time",), np.arange(4))}, coords={"time": np.arange(4)})
ds_d = xr.Dataset({"a": (("time",), np.arange(4) + 10)}, coords={"time": np.arange(4, 8)})
concat = xr.concat([ds_c, ds_d], dim="time")
print("concat 后 time 长度:", len(concat.time), "值:", concat["a"].values)

# 3.8 转换与写出
print("\n--- 3.8 转换 ---")
print("to_pandas 类型:", type(ts.to_pandas()))
print("to_dataframe 前3行:")
print(ts.to_dataframe().head(3))

print("=" * 60)
print("四、常用气象物理公式")
print("=" * 60)

# 风速合成
u = np.array([3.0, 4.0, 5.0])
v = np.array([4.0, 3.0, 12.0])
wind = (u**2 + v**2) ** 0.5
print("风速合成 wind = √(u²+v²):", np.round(wind, 2), "m/s")

# 温度换算
t_k = 285.15
t_c = t_k - 273.15
print(f"温度换算: {t_k}K = {t_c:.1f}℃")

# 区域平均
field = xr.DataArray(np.random.randn(10, 5, 5), dims=("time", "lat", "lon"))
region_ts = field.mean(dim=["lat", "lon"])
print("区域平均后 dims:", region_ts.dims, "长度:", len(region_ts))

print("\n" + "=" * 60)
print("全部运行完成！")
print("=" * 60)
