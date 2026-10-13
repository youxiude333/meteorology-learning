# 河南特大暴雨研究（2021年7月20日）

## 研究对象
2021年7月17-22日河南特大暴雨过程（郑州7·20特大暴雨）

## 需要下载的 ERA5 数据

### 一、气压层数据（reanalysis-era5-pressure-levels）
- **变量**：
  - Temperature（温度 t）
  - Specific humidity（比湿 q）
  - U component of wind（u风）
  - V component of wind（v风）
  - Vertical velocity（垂直速度 w）
  - Relative humidity（相对湿度 r）
  - Geopotential（位势高度 z）
- **气压层**：1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 200, 100 hPa
- **时间**：2021年7月17日 - 7月22日，逐小时
- **区域**：北40°，西110°，南30°，东120°
- **格式**：NetCDF4

### 二、单层数据（reanalysis-era5-single-levels）
- **变量**：
  - Total precipitation（总降水 tp）
  - Convective precipitation（对流降水 cp）
  - Large scale precipitation（大尺度降水 lsp）
  - 2m temperature（2米温度 t2m）
  - 10m u-component of wind（10米u风 u10）
  - 10m v-component of wind（10米v风 v10）
  - Mean sea level pressure（海平面气压 msl）
  - Surface pressure（地表气压 sp）
- **时间**：同上，2021年7月17-22日，逐小时
- **区域**：同上
- **格式**：NetCDF4

## 数据下载地址
- 气压层：https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels
- 单层：https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

## 分析要点
1. 环流背景：500hPa 副高、低槽、台风"烟花"远程输送
2. 水汽条件：850hPa 水汽通量、比湿大值区
3. 动力条件：垂直速度、散度、涡度
4. 不稳定条件：假相当位温 θse 垂直分布、对流有效位能 CAPE
5. 地形作用：太行山、伏牛山地形抬升
