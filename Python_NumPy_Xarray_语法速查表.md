# Python / NumPy / Xarray 常用语法速查表（基于之前所有脚本汇总）

> 分类整理：Python 自带 → NumPy 函数 → Xarray 方法 / 属性 → 运算符与语法糖



***

## 一、Python 自带语法与函数

### 1.1 导入模块



```
import numpy as np        # 导入 numpy，简称 np
import xarray as xr       # 导入 xarray，简称 xr
import pandas as pd       # 导入 pandas，简称 pd
```

### 1.2 打印与查看



```
print("文字")              # 打印内容
print("a =", a, "b =", b) # 打印多个值，自动空格分隔
print("%.2f" % 3.14159)   # 格式化输出：保留2位小数 → 3.14
```

### 1.3 类型转换与查询



```
len([0,1,2,3])           # → 4（求长度）
type(arr)                # 查看变量类型 → <class 'numpy.ndarray'>
float("3.14")            # 字符串转浮点数 → 3.14
list(x)                  # 其他类型转列表
abs(-5)                  # 绝对值 → 5
```

### 1.4 数据容器



```
[0, 1, 2, 3]             # 列表 list
{"a": 1, "b": 2}         # 字典 dict（键值对）
("time", "lat", "lon")   # 元组 tuple（不可改）
```

### 1.5 切片与索引



```
arr[0]                   # 取第 0 个
arr[:4]                  # 取前 4 个（含0，不含4）
arr[1:3]                 # 取第 1、2 个
arr[::-1]                # 倒序
```

### 1.6 条件与逻辑



```
if x > 0:
    ...
else:
    ...

True / False             # 布尔值
None                     # 空值
```

### 1.7 注释



```
# 这是单行注释
"""
这是多行注释
"""
```



***

## 二、NumPy 函数

### 2.1 生成数组



```
np.arange(4)             # → [0, 1, 2, 3]      整数序列（含头不含尾）
np.arange(1, 5)          # → [1, 2, 3, 4]
np.arange(0, 10, 2)      # → [0, 2, 4, 6, 8]   步长为2
np.linspace(20, 50, 4)   # → [20, 30, 40, 50]  等距4个点（含终点）
np.zeros(4)              # → [0, 0, 0, 0]      全0数组
np.ones(4)               # → [1, 1, 1, 1]      全1数组
```

### 2.2 随机数



```
np.random.randn(3, 4)    # 标准正态分布 N(0,1)，形状(3,4)
np.random.rand(3, 4)     # 均匀分布 [0,1)，形状(3,4)

# 新版推荐写法（可复现）
rng = np.random.default_rng(42)   # 固定种子42
rng.normal(0, 2, size=(3,4))      # 均值0、标准差2的正态分布
rng.normal(5, 2, (3,4))           # 同上，size参数可省略关键字
```

### 2.3 统计函数



```
np.mean(arr)             # 平均值
np.std(arr)              # 标准差
np.min(arr)              # 最小值
np.max(arr)              # 最大值
np.median(arr)           # 中位数
np.sum(arr)              # 求和
```

### 2.4 数组操作



```
np.round(arr, 2)         # 保留2位小数
np.allclose(a, b)        # 判断两个数组是否近似相等（容差内）
arr * 2                  # 每个元素乘2（逐元素运算）
arr ** 2                 # 每个元素平方
arr ** 0.5               # 每个元素开平方
arr + 5                  # 每个元素加5
```



***

## 三、Xarray 方法与属性

### 3.1 构造对象



```
# 构造 DataArray
da = xr.DataArray(
    data,                          # 数值数组
    dims=("time", "lat", "lon"),   # 维度名
    coords={"time": time, "lat": lat, "lon": lon},  # 坐标
    name="t2m",                    # 变量名
    attrs={"units": "K"}           # 属性（单位、说明等）
)

# 构造 Dataset（多变量容器）
ds = xr.Dataset(
    {"t2m": (("time","lat","lon"), data1),
     "msl": (("time","lat","lon"), data2)},
    coords={"time": time, "lat": lat, "lon": lon}
)
```

### 3.2 读取数据



```
ds = xr.open_dataset("era5.nc")           # 读单个 NetCDF 文件
ds = xr.open_mfdataset("era5_*.nc")       # 读多个分片文件自动合并
ds = xr.open_dataset("era5.grib", engine="cfgrib")  # 读 GRIB 格式
```

### 3.3 查看结构（属性）



```
da.dims                  # 维度名 → ('time', 'lat', 'lon')
da.shape                 # 形状 → (24, 4, 5)
da.coords                # 坐标信息
da.attrs                 # 属性字典
da.name                  # 变量名
da.values                # 取出纯 numpy 数组（丢坐标）
da.data_vars             # Dataset 的变量列表（只 Dataset 有）
```

### 3.4 索引与切片



```
# 按位置取（第几个）
da.isel(time=0)                  # 取第1个时次
da.isel(time=slice(0, 6))        # 取前6个时次

# 按坐标值取
da.sel(lat=30)                   # lat=30 的点
da.sel(lat=30, method="nearest") # 找最接近30的点
da.sel(lat=slice(20, 40))        # lat 在 20~40 之间
da.sel(time="2020-07-01")        # 按时间标签取
```

### 3.5 维度规约（沿某轴统计）



```
da.mean(dim="time")              # 沿时间平均 → 空间场
da.std(dim="time")                # 沿时间求标准差
da.max(dim=["lat", "lon"])       # 沿空间求最大值 → 时间序列
da.min(dim="time")               # 沿时间求最小值
da.median(dim="time")            # 中位数
da.sum(dim="lat")                # 求和
```

### 3.6 分组与重采样



```
da.groupby("time.month").mean("time")     # 按月平均
da.groupby("time.season").mean("time")    # 按季节平均
da.groupby("time.dayofyear").mean("time")  # 按天平均（气候态）

da.resample(time="1D").mean()             # 重采样为日平均
da.resample(time="1ME").mean()           # 重采样为月平均
da.resample(time="1YE").mean()            # 重采样为年平均
```

### 3.7 滚动与差分



```
da.rolling(time=31, center=True).mean()   # 31天滑动平均
da.differentiate("time")                   # dT/dt 时间变化率
da.differentiate("lon")                   # dT/dlon 水平梯度
```

### 3.8 坐标与属性操作



```
da.rename({"lat": "latitude"})             # 重命名维度
da.assign_coords(source="ERA5")            # 添加辅助坐标
da.copy()                                  # 复制对象
da.interp(lat=30, lon=105)                 # 插值到指定经纬度
```

### 3.9 合并与拼接



```
xr.merge([ds1, ds2])                       # 并排合并变量（坐标需对齐）
xr.concat([ds1, ds2], dim="time")          # 沿 time 维度首尾拼接
```

### 3.10 转换与写出



```
da.to_pandas()                             # 转 pandas（1维→Series，2维+→DataFrame）
da.to_dataframe()                          # 转 DataFrame
da.to_netcdf("result.nc")                  # 保存为 NetCDF
```



***

## 四、运算符速查



| 运算符               | 含义      | 例子            |
| ----------------- | ------- | ------------- |
| `**2`             | 平方      | `x**2` = x²   |
| `**0.5`           | 开平方     | `x**0.5` = √x |
| `* 2`             | 逐元素乘 2  | `arr * 2`     |
| `+ 5`             | 逐元素加 5  | `arr + 5`     |
| `/ 2`             | 逐元素除以 2 | `arr / 2`     |
| `==`              | 等于比较    | `if a == b`   |
| `!=`              | 不等于     | `if a != b`   |
| `>` `<` `>=` `<=` | 大小比较    |               |



***

## 五、常用物理公式（气象场景）



```
# 风速合成：wind = √(u² + v²)
wind = (u10**2 + v10**2) ** 0.5

# 温度开尔文转摄氏度
t2m_c = t2m_k - 273.15

# 距平：实际值 − 气候态
anomaly = actual - climatology

# 区域平均时间序列
cn_ts = t2m.mean(dim=["lat", "lon"])
```



***

## 六、记忆口诀



1. **dims 是骨架，shape 是尺寸，coords 是刻度，attrs 是便签**

2. **isel 按 "第几个"，sel 按 "值是多少"**

3. **mean (dim=...) 是 "压扁哪根轴"**

4. **xarray 方法大多返回新对象，要接住返回值**（`da = da.rename(...)`）

5. `randn(24,4,5)`**&#x20;维度分开写；**`normal(0,2,(24,4,5))`**&#x20;形状用元组**