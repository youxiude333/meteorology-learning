# -*- coding: utf-8 -*-
"""
气象物理量计算教学（可直接运行）
覆盖：位温、假相当位温、风速风向、水汽通量、水平散度、相对涡度
依赖：numpy, xarray
运行：python met_physical_quantities.py
"""

import numpy as np
import xarray as xr

# 物理常数
Rd = 287.0          # 干空气气体常数 J/(kg·K)
Cp = 1004.0         # 干空气定压比热 J/(kg·K)
Lv = 2.5e6          # 水汽凝结潜热 J/kg
Rv = 461.5          # 水汽气体常数 J/(kg·K)
epsilon = 0.622     # Rd/Rv
g = 9.81             # 重力加速度 m/s²
P0 = 1000.0          # 标准气压 hPa
kappa = Rd / Cp      # ≈ 0.286

print("=" * 65)
print("一、位温 θ (Potential Temperature)")
print("=" * 65)
print("公式：θ = T × (P0/P)^κ ，其中 κ = Rd/Cp ≈ 0.286")
print("物理意义：气块从当前气压 P 绝热膨胀/压缩到 1000hPa 时的温度")
print("          位温在绝热过程中守恒，是判断大气稳定度的核心量")
print()

# 造一个三维数据：温度 T(K)、气压 P(hPa)
lat = np.linspace(20, 50, 5)
lon = np.linspace(100, 130, 5)
lev = np.array([1000, 850, 700, 500, 300])  # 气压层 hPa
T_data = 280 - 0.06 * (1000 - lev[:, None, None]) + np.random.randn(5, 5, 5) * 2
P_data = np.broadcast_to(lev[:, None, None], (5, 5, 5)).astype(float)

T = xr.DataArray(T_data, dims=("lev", "lat", "lon"),
                  coords={"lev": lev, "lat": lat, "lon": lon}, name="T")
P = xr.DataArray(P_data, dims=("lev", "lat", "lon"),
                  coords={"lev": lev, "lat": lat, "lon": lon}, name="P")

# 计算位温
theta = T * (P0 / P) ** kappa

print("输入：T 形状", T.shape, "，P 形状", P.shape)
print("输出：θ 形状", theta.shape)
print("\n各层平均位温（验证：高度越高位温越大）：")
for i, l in enumerate(lev):
    print("  %4d hPa: θ = %.2f K" % (l, float(theta.isel(lev=i).mean())))

print("\n单点验证：T=280K, P=850hPa")
print("  θ = 280 × (1000/850)^0.286 = %.2f K" % float(280 * (1000/850)**kappa))

print()
print("=" * 65)
print("二、假相当位温 θse (Equivalent Potential Temperature)")
print("=" * 65)
print("简化公式：θse ≈ θ × exp(Lv × r / (Cp × T))")
print("  其中 r = 混合比 (kg/kg)，θ = 位温")
print("物理意义：气块绝热上升到水汽全部凝结、潜热全部释放后，")
print("          再绝热下降到1000hPa的温度。比位温多考虑了潜热释放。")
print("          θse 随高度增加 = 对流稳定；θse 随高度减小 = 对流不稳定")
print("          （精确计算需要 Bolton 公式，涉及凝结温度，这里用教学简化版）")
print()

# 造数据：温度 T、气压 P、比湿 q(kg/kg)
T_simple = np.array([290, 280, 270, 260, 250], dtype=float)  # K
P_simple = np.array([1000, 850, 700, 500, 300], dtype=float)   # hPa
q_simple = np.array([0.015, 0.010, 0.006, 0.002, 0.0005])      # kg/kg

# 混合比 r ≈ q / (1 - q)（q 是比湿，r 是混合比，低空近似 r≈q）
r = q_simple / (1 - q_simple)

# 位温
theta_simple = T_simple * (P0 / P_simple) ** kappa

# 假相当位温（简化版）
theta_se = theta_simple * np.exp(Lv * r / (Cp * T_simple))

print("层次    P(hPa)  T(K)   q(g/kg)   θ(K)    θse(K)")
print("-" * 60)
for i in range(5):
    print("第%d层  %6.0f  %5.1f  %6.2f   %6.2f  %6.2f" %
          (i, P_simple[i], T_simple[i], q_simple[i]*1000,
           theta_simple[i], theta_se[i]))

print("\n观察：θse 从地面到 700hPa 增加（稳定），但如果某层 θse 随高度减小，")
print("      说明该层对流不稳定，容易产生对流天气。")
print("      （注：精确 θse 用 Bolton(1980) 公式，metpy 库有现成函数）")

print()
print("=" * 65)
print("三、风速和风向 (Wind Speed & Direction)")
print("=" * 65)
print("风速：wind = √(u² + v²)")
print("风向（气象定义）：风从哪个方向吹来，北风=0°，东风=90°，南风=180°，西风=270°")
print("公式：wd = 180 + arctan2(u, v) × 180/π，再对 360 取模")
print("  注意：u 是东风分量（向东为正），v 是北风分量（向北为正）")
print()

# 造 8 个风向的 u/v 对（验证用）
# 北风：u=0, v=-5（向南吹，从北来）
# 东风：u=5, v=0（向西吹，从东来）—— 不对，u>0 是向东吹，风从西来
# 重新理：u>0 风向东吹（西风），v>0 风向北吹（南风）
test_cases = [
    ("北风(0°)",   0,  -5),   # 从北向南吹
    ("东北风(45°)", -3.5, -3.5),
    ("东风(90°)",  5,   0),   # 从东向西吹 —— 不对，u>0是向东
    ("东南风(135°)", 3.5, 3.5),
    ("南风(180°)", 0,   5),   # 从南向北吹
    ("西南风(225°)", -3.5, 3.5),
    ("西风(270°)", -5,  0),   # 从西向东吹
    ("西北风(315°)", -3.5, -3.5),
]

print("方向       u(m/s)  v(m/s)  风速(m/s)  风向(°)")
print("-" * 55)
for name, u, v in test_cases:
    speed = np.sqrt(u**2 + v**2)
    wd = (180 + np.degrees(np.arctan2(u, v))) % 360
    print("%-8s  %6.2f  %6.2f  %8.2f  %8.1f" % (name, u, v, speed, wd))

print("\n三维数组批量计算示例：")
u3d = xr.DataArray(np.random.randn(3, 4, 5), dims=("time", "lat", "lon"))
v3d = xr.DataArray(np.random.randn(3, 4, 5), dims=("time", "lat", "lon"))
wind3d = np.sqrt(u3d**2 + v3d**2)
wd3d = (180 + np.degrees(np.arctan2(u3d, v3d))) % 360
print("  u 形状:", u3d.shape, "→ 风速形状:", wind3d.shape, "风向形状:", wd3d.shape)
print("  平均风速: %.2f m/s" % float(wind3d.mean()))

print()
print("=" * 65)
print("四、水汽通量 (Water Vapor Flux)")
print("=" * 65)
print("单层水汽通量：Q = q × V / g")
print("  q = 比湿(kg/kg)，V = 风速(m/s)，g = 9.81 m/s²")
print("  单位：kg/(m·s) 或 g/(cm·hPa·s)")
print("整层水汽通量：Q = (1/g) × ∫(q × V) dp （从地表积分到高层）")
print("物理意义：单位时间内通过单位长度的水汽质量，是水汽输送的度量")
print("          水汽通量大值区 = 水汽通道，对应强降水的水汽来源")
print()

# 造数据：比湿 q、u、v、气压层
lev_q = np.array([1000, 925, 850, 700, 600, 500])  # hPa
q_data = np.array([0.018, 0.014, 0.010, 0.005, 0.002, 0.0008])  # kg/kg
u_data = np.array([3, 5, 8, 12, 15, 18])   # m/s
v_data = np.array([-2, 0, 3, 5, 4, 2])      # m/s

# 单层水汽通量（以 850hPa 为例）
idx_850 = 2
q_850 = q_data[idx_850]
u_850 = u_data[idx_850]
v_850 = v_data[idx_850]
wind_850 = np.sqrt(u_850**2 + v_850**2)
Qx_850 = q_850 * u_850 / g   # 纬向水汽通量
Qy_850 = q_850 * v_850 / g   # 经向水汽通量
Q_850 = q_850 * wind_850 / g # 水汽通量大小

print("850hPa 单层水汽通量计算：")
print("  q = %.4f kg/kg = %.1f g/kg" % (q_850, q_850*1000))
print("  u = %.1f m/s, v = %.1f m/s, 风速 = %.2f m/s" % (u_850, v_850, wind_850))
print("  Qx(纬向) = q×u/g = %.4f kg/(m·s)" % Qx_850)
print("  Qy(经向) = q×v/g = %.4f kg/(m·s)" % Qy_850)
print("  Q(大小)  = q×V/g = %.4f kg/(m·s)" % Q_850)

# 整层水汽通量（梯形积分）
print("\n整层水汽通量（从1000hPa积分到500hPa，梯形法）：")
Qx_all = q_data * u_data / g
Qy_all = q_data * v_data / g
# 梯形积分：∫Q dp ≈ Σ (Q_i + Q_{i+1})/2 × (P_{i+1} - P_i)
# 注意气压从大到小，dp 为负，取绝对值
Qx_int = 0
Qy_int = 0
for i in range(len(lev_q)-1):
    dp = abs(lev_q[i+1] - lev_q[i]) * 100  # hPa → Pa
    Qx_int += (Qx_all[i] + Qx_all[i+1]) / 2 * dp
    Qy_int += (Qy_all[i] + Qy_all[i+1]) / 2 * dp
Qx_int /= g  # 整层通量已经含 1/g，这里修正
Qy_int /= g
print("  整层 Qx = %.2f kg/(m·s)" % Qx_int)
print("  整层 Qy = %.2f kg/(m·s)" % Qy_int)
print("  整层 Q大小 = %.2f kg/(m·s)" % np.sqrt(Qx_int**2 + Qy_int**2))

print()
print("=" * 65)
print("五、水汽通量散度 (Moisture Flux Divergence)")
print("=" * 65)
print("公式：∇·Q = ∂Qx/∂x + ∂Qy/∂y")
print("物理意义：")
print("  散度 > 0（辐散）：水汽向外流出，该区域水汽减少，不利于降水")
print("  散度 < 0（辐合）：水汽向内汇聚，该区域水汽增加，有利于降水")
print("  强降水区通常对应水汽通量辐合中心（负值中心）")
print()
print("注意：在经纬度网格上计算散度，需要考虑网格间距随纬度变化。")
print("      教学中用 xarray.differentiate 近似（假设等间距），")
print("      业务中推荐用 metpy.calc.divergence（自动处理投影）。")
print()

# 造一个二维场：Qx, Qy，带一个辐合中心
lat2d = np.linspace(20, 40, 21)
lon2d = np.linspace(100, 120, 21)
lon_grid, lat_grid = np.meshgrid(lon2d, lat2d)
# 造一个辐合中心在 (30°N, 110°E)：Qx 向西汇聚，Qy 向南汇聚
center_lat, center_lon = 30, 110
Qx_2d = -0.01 * (lon_grid - center_lon)   # 东侧向西流，西侧向东流 → 辐合
Qy_2d = -0.01 * (lat_grid - center_lat)   # 北侧向南流，南侧向北流 → 辐合

Qx_da = xr.DataArray(Qx_2d, dims=("lat", "lon"), coords={"lat": lat2d, "lon": lon2d})
Qy_da = xr.DataArray(Qy_2d, dims=("lat", "lon"), coords={"lat": lat2d, "lon": lon2d})

# 计算散度：∂Qx/∂x + ∂Qy/∂y
# xarray.differentiate 计算导数，coord 指定对哪个坐标求导
div_Q = Qx_da.differentiate("lon") + Qy_da.differentiate("lat")

print("输入：Qx, Qy 形状", Qx_da.shape)
print("输出：散度形状", div_Q.shape)
print("\n辐合中心 (30°N, 110°E) 附近的散度（应为负值）：")
center_div = div_Q.sel(lat=30, lon=110, method="nearest")
print("  中心点散度 = %.6f (负值=辐合)" % float(center_div))
print("\n四个角的散度（应为正值，辐散）：")
for la, lo in [(20, 100), (20, 120), (40, 100), (40, 120)]:
    val = float(div_Q.sel(lat=la, lon=lo, method="nearest"))
    print("  (%d°N, %d°E): %.6f (正值=辐散)" % (la, lo, val))

print()
print("=" * 65)
print("六、水平散度 (Horizontal Divergence)")
print("=" * 65)
print("公式：D = ∂u/∂x + ∂v/∂y")
print("物理意义：")
print("  D > 0（辐散）：空气向外流出，高空辐散对应地面低压发展")
print("  D < 0（辐合）：空气向内汇聚，低空辐合对应上升运动和降水")
print("  量级：通常 10^-5 s^-1")
print()

# 造一个二维风场，带一个辐合中心（类似气旋）
u_div = -0.005 * (lon_grid - center_lon) + 0.003 * (lat_grid - center_lat)
v_div = -0.005 * (lat_grid - center_lat) - 0.003 * (lon_grid - center_lon)

u_div_da = xr.DataArray(u_div, dims=("lat", "lon"), coords={"lat": lat2d, "lon": lon2d})
v_div_da = xr.DataArray(v_div, dims=("lat", "lon"), coords={"lat": lat2d, "lon": lon2d})

# 水平散度
D = u_div_da.differentiate("lon") + v_div_da.differentiate("lat")

print("输入：u, v 形状", u_div_da.shape)
print("输出：散度 D 形状", D.shape)
print("\n中心 (30°N, 110°E) 散度：")
print("  D = %.2e s^-1" % float(D.sel(lat=30, lon=110, method="nearest")))
print("  （负值=辐合，正值=辐散）")
print("\n全场散度范围：%.2e ~ %.2e s^-1" % (float(D.min()), float(D.max())))

print()
print("=" * 65)
print("七、相对涡度 (Relative Vorticity)")
print("=" * 65)
print("公式：ζ = ∂v/∂x - ∂u/∂y")
print("物理意义：空气的旋转程度")
print("  ζ > 0（正涡度）：气旋式旋转（北半球逆时针），对应低压系统")
print("  ζ < 0（负涡度）：反气旋式旋转（北半球顺时针），对应高压系统")
print("  量级：通常 10^-5 s^-1，台风中心可达 10^-3 s^-1")
print("  涡度平流（正涡度平流）是天气系统发展的重要机制")
print()

# 用上面的风场计算相对涡度
vort = v_div_da.differentiate("lon") - u_div_da.differentiate("lat")

print("输入：u, v 形状", u_div_da.shape)
print("输出：相对涡度 ζ 形状", vort.shape)
print("\n中心 (30°N, 110°E) 相对涡度：")
print("  ζ = %.2e s^-1" % float(vort.sel(lat=30, lon=110, method="nearest")))
print("  （正值=气旋式，负值=反气旋式）")
print("\n全场涡度范围：%.2e ~ %.2e s^-1" % (float(vort.min()), float(vort.max())))

print()
print("=" * 65)
print("八、散度 vs 涡度：公式对比（容易记混）")
print("=" * 65)
print("水平散度：  D = ∂u/∂x + ∂v/∂y   （加号，u对x + v对y）")
print("相对涡度：  ζ = ∂v/∂x - ∂u/∂y   （减号，v对x - u对y）")
print()
print("记忆口诀：")
print("  散度 = 自己对自己求导相加（u对x，v对y）")
print("  涡度 = 交叉求导相减（v对x，u对y）")
print()
print("物理意义对比：")
print("  散度：空气'聚还是散'（辐合→上升→降水；辐散→下沉→晴）")
print("  涡度：空气'转不转'（正涡度→气旋→低压；负涡度→反气旋→高压）")

print()
print("=" * 65)
print("九、经纬度网格的注意事项 + metpy 库")
print("=" * 65)
print("上面用 xarray.differentiate 计算散度/涡度是近似方法，因为：")
print("  1. 经纬度网格不是等间距的：经度方向 1° 的实际距离 = 111km × cos(lat)")
print("     纬度越高，经度方向网格越密，直接 differentiate 会有误差")
print("  2. 地球是球面，严格计算需要用球坐标公式")
print()
print("业务/科研中推荐用 metpy 库（气象专用，自动处理投影）：")
print("  pip install metpy")
print("  import metpy.calc as mpcalc")
print("  dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)  # 计算实际网格间距")
print("  div = mpcalc.divergence(u, v, dx, dy)           # 水平散度")
print("  vort = mpcalc.vorticity(u, v, dx, dy)            # 相对涡度")
print("  theta = mpcalc.potential_temperature(p, T)        # 位温")
print("  theta_e = mpcalc.equivalent_potential_temperature(p, T, Td)  # 假相当位温")
print()
print("metpy 还包含：风速风向、湿球温度、相对湿度、稳定度指数等几十种气象计算。")

print()
print("全部物理量计算教学完成。")
