# xarray 学习笔记（ERA5 数据处理 + Cartopy 绘图）

> 来源：主对话 2026-10-04 关于 xarray 的学习内容，已完整迁移到本对话。
> 配套可运行脚本见同目录 
>
> `xarray_practice.py`
>
> 。



***

## Part 1｜xarray 核心对象：DataArray 与 Dataset



```
import numpy as np

import xarray as xr

\# ---- 1.1 创建 DataArray ----

\# 模拟一个 3D 变量：时间(time) × 纬度(lat) × 经度(lon)

time = np.arange(0, 24)          # 24 小时

lat  = np.linspace(20, 50, 7)    # 7 个纬度

lon  = np.linspace(100, 130, 9)  # 9 个经度

data = np.random.randn(len(time), len(lat), len(lon))

da = xr.DataArray(

&#x20;   data,

&#x20;   dims=("time", "lat", "lon"),

&#x20;   coords={"time": time, "lat": lat, "lon": lon},

&#x20;   name="temperature",

&#x20;   attrs={"units": "K", "long\_name": "Air Temperature"},

)

print(da)

\# ---- 1.2 Dataset：装多个变量 ----

ds = xr.Dataset({"temperature": da,

&#x20;                "pressure": da \* 0.5 + 100})  # 随便造一个派生变量

print(ds)

\# ---- 1.3 快速查看 ----

print(da.dims, da.shape)        # 维度

print(da.coords)                # 坐标

print(da.attrs)                 # 属性

print(da.name)                  # 变量名
```



***

## Part 2｜读取与查看 ERA5 数据



```
import xarray as xr

\# 打开 NetCDF 文件（按你的实际路径改）

ds = xr.open\_dataset("/your/path/era5\_2020.nc")

\# 常用读取方式

ds  = xr.open\_dataset("era5.nc")            # 直接读

ds  = xr.open\_mfdataset("era5\_\*.nc")        # 读取多个分片文件自动合并

ds  = xr.open\_dataset("era5.grib", engine="cfgrib")  # 读 GRIB 格式

print(ds)                          # 总览：变量、维度、坐标

print(ds\["t2m"])                   # 查看某个变量（2m 温度）

print(ds.variables)                # 所有变量名
```



***

## Part 3｜索引与切片（最常用）



```
\# ---- 3.1 位置索引 .isel() ----

da0 = ds\["t2m"].isel(time=0)          # 取第 1 个时次

da0 = ds\["t2m"].isel(time=slice(0,6)) # 取前 6 个时次

\# ---- 3.2 标签索引 .sel() ----

da1 = ds\["t2m"].sel(time="2020-07-01")          # 按时间标签取

da2 = ds\["t2m"].sel(lat=30)                     # 最近纬度

da3 = ds\["t2m"].sel(lat=slice(20,40),           # 纬度范围

&#x20;                   lon=slice(100,120))

da4 = ds\["t2m"].sel(time="2020-07-01",          # 多点选取

&#x20;                   lat=\[30, 35], lon=\[105, 110])

\# ---- 3.3 最近邻 vs 插值 ----

da5 = ds\["t2m"].sel(lat=30, method="nearest")   # 找最近格点

da6 = ds\["t2m"].interp(lat=30)                  # 线性插值到 30°N
```



***

## Part 4｜维度规约（平均、最大、求和）



```
\# 时间平均 → 得到气候态/日平均

clim = ds\["t2m"].mean(dim="time")

\# 区域平均

region = ds\["t2m"].sel(lat=slice(20,40), lon=slice(100,120)).mean(dim=\["lat","lon"])

\# 常用统计

da.mean(dim="time")    # 平均

da.std(dim="time")     # 标准差

da.max(dim="time")     # 最大

da.min(dim="time")     # 最小

da.sum(dim="lat")      # 求和

da.median(dim="time")  # 中位数

\# 时间序列（对空间做平均，得到一条时间曲线）

ts = ds\["t2m"].mean(dim=\["lat","lon"])   # 全球/区域平均时间序列
```



***

## Part 5｜groupby 分组计算（气候学最爱）



```
\# 计算逐日气候态（多年同一天平均）

\# 假设有多年逐日数据

clim\_day = ds\["t2m"].groupby("time.dayofyear").mean("time")

\# 计算距平（原数据 - 气候态）

anomaly = ds\["t2m"].groupby("time.dayofyear") - clim\_day

\# 按月平均（季节或月气候态）

monthly = ds\["t2m"].groupby("time.month").mean("time")

\# 按季节平均

seasonal = ds\["t2m"].groupby("time.season").mean("time")

\# 日平均（把一天内的小时合成日）

daily = ds\["t2m"].resample(time="1D").mean()
```



***

## Part 6｜滚动、差分与派生量计算



```
\# ---- 6.1 滚动平均（9 天滑动平均） ----

smooth = ds\["t2m"].rolling(time=9, center=True).mean()

\# ---- 6.2 差分（水平梯度 / 时间趋势） ----

dTdx = ds\["t2m"].differentiate("lon")    # dT/dlon

dTdt = ds\["t2m"].differentiate("time")   # dT/dt

\# ---- 6.3 用 xr 的 ufunc 直接做物理量运算 ----

\# 位温 θ = T \* (1000/p)^(R/cp)

theta = ds\["t"] \* (1000.0 / ds\["level"]) \*\* (287.0/1004.0)

\# ---- 6.4 xarray 自动对齐 + 广播 ----

\# 两个不同维度变量直接相乘，xarray 自动按坐标对齐

wind\_speed = (ds\["u10"]\*\*2 + ds\["v10"]\*\*2) \*\* 0.5
```



***

## Part 7｜用 Cartopy 绘制 ERA5 空间场



```
import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import cartopy.feature as cfeature

\# 取一个时次、做区域切片（如中国区域）

field = ds\["t2m"].sel(time="2020-07-01", method="nearest").sel(

&#x20;   lat=slice(15, 55), lon=slice(70, 140))

fig = plt.figure(figsize=(10, 6))

ax = plt.axes(projection=ccrs.PlateCarree())

ax.set\_extent(\[70, 140, 15, 55], crs=ccrs.PlateCarree())

\# 画等值线填充

cf = ax.contourf(field\["lon"], field\["lat"], field,

&#x20;                transform=ccrs.PlateCarree(),

&#x20;                cmap="RdBu\_r", levels=20)

\# 画海岸线、国家、省界

ax.coastlines(linewidth=0.8)

ax.add\_feature(cfeature.BORDERS, linewidth=0.5)

ax.add\_feature(cfeature.COASTLINE, linewidth=0.8)

\# 标题 + 色标

ax.set\_title("ERA5 2m Temperature 2020-07-01")

plt.colorbar(cf, orientation="horizontal", shrink=0.8, label="K")

plt.savefig("t2m\_plot.png", dpi=200, bbox\_inches="tight")

plt.show()
```



***

## Part 8｜站点插值提取（把格点数据取到站点）



```
import xarray as xr

\# 站点经纬度

stations = xr.Dataset(

&#x20;   {"lat": ("station", \[30.5, 34.0, 39.9]),

&#x20;    "lon": ("station", \[104.0, 108.9, 116.4])},

&#x20;   coords={"station": \["成都", "西安", "北京"]},

)

\# 双线性插值到站点

station\_ts = ds\["t2m"].interp(

&#x20;   lat=stations.lat, lon=stations.lon, method="linear")

print(station\_ts)
```



***

## Part 9｜写出结果



```
\# 保存为 NetCDF

ds\_result = xr.Dataset({"t2m\_mean": clim})

ds\_result.to\_netcdf("t2m\_climatology.nc")

\# 保存为 CSV（转成 pandas）

df = ts.to\_dataframe()          # DataArray → DataFrame

df.to\_csv("time\_series.csv")

\# 保存为 GRIB / 其它格式需要对应 engine
```



***

## 综合练习（可整段跑通）

把前面学的串起来：读数据 → 算区域平均时间序列 → 算气候态距平 → 画两张图。完整可运行版本见 `xarray_practice.py`。



```
import xarray as xr

import matplotlib.pyplot as plt

import cartopy.crs as ccrs

import cartopy.feature as cfeature

\# 1. 读数据（替换为你的文件）

ds = xr.open\_dataset("era5\_2t\_2020\_2024.nc")

t2m = ds\["t2m"] - 273.15            # K → ℃

\# 2. 中国区域平均时间序列

cn = t2m.sel(lat=slice(15,55), lon=slice(70,140)).mean(dim=\["lat","lon"])

\# 3. 逐日气候态 + 距平

clim\_day = cn.groupby("time.dayofyear").mean("time")

anom = cn.groupby("time.dayofyear") - clim\_day

\# 4. 画时间序列图

fig, axes = plt.subplots(2, 1, figsize=(11, 7))

cn.plot(ax=axes\[0], color="tab:red")

axes\[0].set\_title("China area-mean T2m (℃)")

anom.plot(ax=axes\[1], color="tab:blue")

axes\[1].set\_title("Anomaly (℃)")

axes\[1].axhline(0, color="k", lw=0.8)

plt.tight\_layout(); plt.savefig("series.png", dpi=200); plt.show()

\# 5. 画 2020-07-01 空间场

field = t2m.sel(time="2020-07-01", method="nearest")

ax = plt.axes(projection=ccrs.PlateCarree())

ax.set\_extent(\[70,140,15,55], crs=ccrs.PlateCarree())

ax.contourf(field.lon, field.lat, field, transform=ccrs.PlateCarree(),

&#x20;           cmap="RdBu\_r", levels=20)

ax.coastlines(linewidth=0.8); ax.add\_feature(cfeature.BORDERS, linewidth=0.5)

ax.set\_title("T2m 2020-07-01 (℃)")

plt.colorbar(ax.contourf(field.lon, field.lat, field,

&#x20;                        transform=ccrs.PlateCarree(), cmap="RdBu\_r", levels=20),

&#x20;            orientation="horizontal", shrink=0.8)

plt.savefig("map.png", dpi=200); plt.show()
```



***

## 实用小提示



* **读 GRIB** 需要装 `cfgrib` + `eccodes`；读 netCDF 需要 `netcdf4`。

* `.sel(..., method="nearest")` 是取最近格点，`interp()` 是插值，两者别混用。

* ERA5 常用变量里 `t2m` 是 2m 温度（单位 K，记得减 273.15 转 ℃），`u10/v10` 是 10m 风，`msl` 是海平面气压，`tp` 是降水。