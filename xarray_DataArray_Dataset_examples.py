# -*- coding: utf-8 -*-
"""
DataArray 与 Dataset 学习示例（可复制直接运行，零外部数据依赖）
运行：python xarray_DataArray_Dataset_examples.py
输出会分段打印，配合注释理解每个例子在做什么。
"""

import numpy as np
import xarray as xr

print("=" * 60)
print("Part A｜DataArray 学习示例")
print("=" * 60)

# ============================================================
# A1. 创建 DataArray（一维 + 三维）
# ============================================================
print("\n--- A1. 创建 DataArray ---")
# 一维：单条时间序列
time1d = xr.DataArray(
    np.array([10.0, 12.0, 14.5, 11.0]),
    dims=["time"],
    coords={"time": ["00Z", "06Z", "12Z", "18Z"]},
    name="t2m",
    attrs={"units": "degC", "long_name": "2m temperature"},
)
print(time1d)

# 三维：time × lat × lon（气象场最常见形态）
time = np.arange(0, 24)          # 24 小时
lat = np.linspace(20, 50, 4)     # 4 个纬度
lon = np.linspace(100, 130, 5)   # 5 个经度
data3d = np.random.randn(len(time), len(lat), len(lon))
t2m = xr.DataArray(
    data3d,
    dims=("time", "lat", "lon"),
    coords={"time": time, "lat": lat, "lon": lon},
    name="t2m",
    attrs={"units": "K", "long_name": "2m temperature"},
)
print(t2m)

# ============================================================
# A2. 查看结构：dims / coords / attrs / name
# ============================================================
print("\n--- A2. 查看结构 ---")
print("dims:", t2m.dims)
print("shape:", t2m.shape)
print("coords:", t2m.coords)
print("attrs:", t2m.attrs)
print("name:", t2m.name)

# ============================================================
# A3. 坐标操作：重命名、改值、添加辅助坐标
# ============================================================
print("\n--- A3. 坐标操作 ---")
t2m2 = t2m.copy()
t2m2 = t2m2.rename({"lat": "latitude", "lon": "longitude"})  # 改坐标名
print("重命名后 dims:", t2m2.dims)

# 添加一个与维度无关的辅助坐标（如数据集标识）
t2m2 = t2m2.assign_coords(source="ERA5")
print("辅助坐标:", t2m2.coords)

# ============================================================
# A4. 属性(attrs)操作：设置、更新、删除
# ============================================================
print("\n--- A4. attrs 操作 ---")
t2m3 = t2m.copy()
t2m3.attrs["frequency"] = "hourly"          # 新增一个属性
t2m3.attrs["units"] = "degC"               # 更新已有属性
print("更新后 attrs:", t2m3.attrs)
# 删除某个属性
del t2m3.attrs["long_name"]
print("删除 long_name 后:", t2m3.attrs)

# ============================================================
# A5. 名字(name)与数值转换
# ============================================================
print("\n--- A5. name 与转换 ---")
t2m4 = t2m.copy()
t2m4 = t2m4.rename("temperature")          # 改变量名
print("改后 name:", t2m4.name)

arr = t2m.values                            # → numpy 数组
print("values 类型:", type(arr), "shape:", arr.shape)
ser = time1d.to_pandas()                    # → pandas Series
print("to_pandas 类型:", type(ser))

# ============================================================
# A6. 算术运算：标量、数组、广播对齐
# ============================================================
print("\n--- A6. 算术运算 ---")
t2m_c = t2m - 273.15                        # 标量：K → ℃
print("减标量后范围: %.2f ~ %.2f" % (t2m_c.min(), t2m_c.max()))

# 与另一变量相乘：xarray 自动按坐标对齐
u10 = xr.DataArray(
    np.random.randn(len(time), len(lat), len(lon)),
    dims=("time", "lat", "lon"),
    coords=t2m.coords,
    name="u10",
)
wind = (t2m**2 + u10**2) ** 0.5             # 派生量
print("合成变量 wind:", wind.name, wind.shape)

# ============================================================
# A7. 索引：位置(.isel) vs 标签(.sel)
# ============================================================
print("\n--- A7. 索引 ---")
print("isel(time=0) shape:", t2m.isel(time=0).shape)      # 取第 1 时次
print("sel(lat=30) 最近纬度:", t2m.sel(lat=30, method="nearest").coords["lat"].values)
print("sel(lat=slice(20,40)) 纬度范围切片:")
print(t2m.sel(lat=slice(20, 40)).coords["lat"].values)

# ============================================================
# A8. 维度规约（平均/最大/求和等）
# ============================================================
print("\n--- A8. 维度规约 ---")
print("mean(dim='time') 后 dims:", t2m.mean(dim="time").dims)   # lat, lon
print("std(dim='time') 后 dims:", t2m.std(dim="time").dims)
print("max(dim=['lat','lon']) 后 dims:", t2m.max(dim=["lat", "lon"]).dims)  # time
print("中位数:", float(t2m.median()))

# ============================================================
# A9. 时间序列提取：对空间平均得到一条曲线
# ============================================================
print("\n--- A9. 空间平均时间序列 ---")
ts = t2m.mean(dim=["lat", "lon"])           # 只剩 time 维度
print("时间序列 dims:", ts.dims, "长度:", len(ts))

print()
print("=" * 60)
print("Part B｜Dataset 学习示例")
print("=" * 60)

# ============================================================
# B1. 创建 Dataset：多个变量共用坐标
# ============================================================
print("\n--- B1. 创建 Dataset ---")
ds = xr.Dataset(
    {
        "t2m": (("time", "lat", "lon"), np.random.randn(len(time), len(lat), len(lon)) + 280),
        "msl": (("time", "lat", "lon"), np.random.randn(len(time), len(lat), len(lon)) + 1013),
    },
    coords={"time": time, "lat": lat, "lon": lon},
)
ds["t2m"].attrs = {"units": "K"}
ds["msl"].attrs = {"units": "hPa"}
print(ds)

# ============================================================
# B2. 查看结构
# ============================================================
print("\n--- B2. 查看结构 ---")
print("变量 variables:", list(ds.data_vars))
print("维度 dims:", ds.dims)
print("坐标 coords:", list(ds.coords))

# ============================================================
# B3. 添加 / 删除变量
# ============================================================
print("\n--- B3. 添加/删除变量 ---")
ds["rh"] = (("time", "lat", "lon"), np.random.rand(len(time), len(lat), len(lon)))  # 新增
print("添加 rh 后变量:", list(ds.data_vars))
del ds["rh"]                                  # 删除
print("删除 rh 后变量:", list(ds.data_vars))

# ============================================================
# B4. 重命名变量与维度
# ============================================================
print("\n--- B4. 重命名 ---")
ds2 = ds.rename({"t2m": "temperature"})      # 重命名变量
print("重命名后变量:", list(ds2.data_vars))
ds3 = ds.rename_dims({"lat": "latitude"})    # 重命名维度
print("重命名维度后 dims:", ds3.dims)

# ============================================================
# B5. 变量间运算（Dataset 内跨变量）
# ============================================================
print("\n--- B5. 变量间运算 ---")
# 计算温差 / 距平（t2m - 全场平均）
anomaly = ds["t2m"] - ds["t2m"].mean(dim="time")
print("距平变量 dims:", anomaly.dims)
# 两个变量的合成
combined =  ds["t2m"] *0.01 + ds["msl"] * 0.001
print("合成变量 dims:", combined.dims)

# ============================================================
# B6. 对 Dataset 整体做索引/规约
# ============================================================
print("\n--- B6. 整体索引/规约 ---")
sub = ds.isel(time=slice(0, 6), lat=slice(0, 2))   # 同时裁时间+纬度
print("isel 后 time 长度:", len(sub.time), "lat 长度:", len(sub.lat))
# 对所有变量做时间平均
ds_mean = ds.mean(dim="time")
print("mean 后各变量 shape:")
for v in ds_mean.data_vars:
    print("  ", v, ds_mean[v].shape)

# ============================================================
# B7. 合并：merge（按坐标拼接变量） / concat（沿维度堆叠）
# ============================================================
print("\n--- B7. 合并 ---")
ds_a = xr.Dataset({"a": (("time",), np.arange(4))}, coords={"time": time[:4]})
ds_b = xr.Dataset({"b": (("time",), np.arange(4) * 2)}, coords={"time": time[:4]})
merged = xr.merge([ds_a, ds_b])
print("merge 后变量:", list(merged.data_vars))

ds_c = xr.Dataset({"a": (("time",), np.arange(4))}, coords={"time": time[:4]})
ds_d = xr.Dataset({"a": (("time",), np.arange(4) + 10)}, coords={"time": time[4:8]})
concat = xr.concat([ds_c, ds_d], dim="time")
print("concat 后 time 长度:", len(concat.time), "值:", concat["a"].values)

# ============================================================
# B8. 写出：转 DataFrame / 存 NetCDF
# ============================================================
print("\n--- B8. 写出结果 ---")
# DataArray → DataFrame（适合存 CSV）
df = ts.to_dataframe()
print("时间序列转 DataFrame 前几行:")
print(df.head())
# Dataset → NetCDF（需要 netcdf4）
# ds.to_netcdf("demo.nc")

print("\n全部示例运行完成。")
