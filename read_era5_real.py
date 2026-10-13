# -*- coding: utf-8 -*-
"""
ERA5 真实数据读取脚本（NetCDF / GRIB 通用）
使用方法：
  1. 把下面的 DATA_FILE 改成你下载的 ERA5 文件的完整路径
  2. 在 Spyder 里按 F5 运行
依赖：xarray, netcdf4（读 nc 需要）；cfgrib + eccodes（读 grib 需要）
  缺什么装什么：pip install netcdf4
                pip install cfgrib  （GRIB 还需系统装 eccodes，见末尾说明）
"""

import os
import xarray as xr

# ============================================================
# ★ 第一步：改成你自己的 ERA5 文件路径 ★
# ============================================================
# 单个文件示例（NetCDF 或 GRIB 都可以）：
DATA_FILE = r"C:\Users\86137\Downloads\dbda254db6bf7a56ffa3e6a8797f34c.nc"

# 多个分片文件示例（ERA5 常按年/月分成多个文件，用通配符 *）：
# 把下面这行取消注释，并改成你的路径：
# DATA_FILES ="C:\Users\86137\Downloads\dbda254db6bf7a56ffa3e6a8797f34c.nc"

# ============================================================
# 第二步：读取文件
# ============================================================
print("=" * 60)
print("读取文件:", DATA_FILE)
print("文件存在:", os.path.exists(DATA_FILE))
print("=" * 60)

# 根据后缀自动选择读取引擎
ext = os.path.splitext(DATA_FILE)[1].lower()

if ext in (".grib", ".grb", ".grib2"):
    # ---- 读 GRIB 格式 ----
    # 注意：一个 GRIB 文件可能含多种层次（地面/气压层），
    # 如果报错 "multiple values for key"，需要用 filter_by_keys 筛选，
    # 例如只取地面层：
    #   ds = xr.open_dataset(DATA_FILE, engine="cfgrib",
    #       backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}})
    print("检测到 GRIB 格式，使用 cfgrib 引擎读取...")
    ds = xr.open_dataset(DATA_FILE, engine="cfgrib")
else:
    # ---- 读 NetCDF 格式（最常用）----
    print("检测到 NetCDF 格式，使用默认引擎读取...")
    ds = xr.open_dataset(DATA_FILE)

print("\n读取成功！数据总览：")
print(ds)

# ============================================================
# 第三步：查看数据结构（读任何真实数据的第一步）
# ============================================================
print("\n" + "=" * 60)
print("数据结构详情")
print("=" * 60)

print("\n【所有变量名】:", list(ds.data_vars))
print("【所有坐标名】:", list(ds.coords))
print("【维度与长度】:", dict(ds.sizes))

print("\n【每个变量的单位】:")
for var in ds.data_vars:
    unit = ds[var].attrs.get("units", "（未标注）")
    long_name = ds[var].attrs.get("long_name", "（未标注）")
    print(f"  {var:12s} | 单位: {unit:10s} | 含义: {long_name}")

print("\n【坐标实际值（前几个）】:")
for coord in ds.coords:
    vals = ds[coord].values
    if hasattr(vals, "ndim") and vals.ndim > 0 and len(vals) > 5:
        print(f"  {coord}: {vals[:3]} ... {vals[-2:]}  (共{len(vals)}个)")
    else:
        print(f"  {coord}: {vals}")

# ============================================================
# 第四步：读取后取数（结合你学过的 sel / isel / mean）
# ============================================================
print("\n" + "=" * 60)
print("取数示例（把变量名改成你数据里真实存在的）")
print("=" * 60)

# ERA5 常见变量名：t2m(2米温度), msl(海平面气压), u10/v10(10米风),
#                   tp(总降水), z(位势高度), t(温度), u/v(风)
# 先看看你的数据里有什么变量，再取消对应注释

# --- 示例1：取某个变量 ---
# var_name = "t2m"   # ← 改成你数据里的变量名
# t2m = ds[var_name]
# print(f"\n变量 {var_name} 的形状:", t2m.shape)
# print(f"变量 {var_name} 的单位:", t2m.attrs.get("units"))

# --- 示例2：按时间和经纬度取一个点 ---
# point = t2m.sel(time="2020-01-15", latitude=30, longitude=110, method="nearest")
# print(f"\n2020-01-15 附近 (30N,110E) 的值: {float(point):.2f}")

# --- 示例3：区域平均（消掉经纬度，得到时间序列）---
# region_ts = t2m.sel(latitude=slice(25, 40), longitude=slice(105, 120)).mean(dim=["latitude", "longitude"])
# print(f"\n区域平均后形状: {region_ts.shape}")
# print(f"前5个值: {region_ts.values[:5]}")

# --- 示例4：时间平均（消掉时间，得到空间分布图）---
# time_mean = t2m.mean(dim="time")
# print(f"\n时间平均后形状: {time_mean.shape}")

# ============================================================
# 第五步：关闭文件（好习惯）
# ============================================================
ds.close()
print("\n文件已关闭。读取完成。")

# ============================================================
# 附：常见报错排查
# ============================================================
"""
1. ImportError: No module named 'netCDF4'
   → 解决：pip install netcdf4

2. ValueError: found the following matches with multiple values for key: 'typeOfLevel'
   → 原因：GRIB 文件含多种层次，cfgrib 不知道读哪个
   → 解决：加 filter_by_keys，例如：
        ds = xr.open_dataset("file.grib", engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "surface"}})

3. ModuleNotFoundError: No module named 'cfgrib'
   → 解决：pip install cfgrib
   → 另外 cfgrib 依赖系统级的 eccodes 库：
        Windows: 推荐用 conda install -c conda-forge eccodes cfgrib
        （纯 pip 在 Windows 上装 eccodes 比较麻烦，conda 更省心）

4. KeyError: 't2m'  /  找不到变量
   → 原因：你数据里的变量名可能不是 t2m
   → 解决：先 print(list(ds.data_vars)) 看真实变量名，ERA5 变量名参考：
        t2m=2m温度, msl=海平面气压, u10=10m东风, v10=10m北风,
        tp=总降水, z=位势高度, t=气压层温度, u=气压层东风, v=气压层北风

5. 内存不够 / 文件太大
   → 解决：用 open_mfdataset + chunks 分块读：
        ds = xr.open_mfdataset("era5_*.nc", chunks={"time": 100})
"""
