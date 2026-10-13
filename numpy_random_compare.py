# -*- coding: utf-8 -*-
"""
numpy 三种随机函数对比（可直接运行）
randn / rand / normal 的写法区别与输出特征
运行：python numpy_random_compare.py
"""

import numpy as np

print("=" * 60)
print("1. np.random.randn(d0, d1, ...)  —— 标准正态分布 N(0,1)")
print("=" * 60)
# 括号里直接列维度，写几个就几维
a0 = np.random.randn()                 # 不写参数 → 标量
a1 = np.random.randn(5)               # 一个参数 → 一维
a2 = np.random.randn(4, 5)           # 两个参数 → 二维
a3 = np.random.randn(3, 4, 5)         # 三个参数 → 三维
print("标量   :", a0)
print("一维   : shape", a1.shape, "值:", np.round(a1, 2))
print("二维   : shape", a2.shape)
print("三维   : shape", a3.shape)
# 特征：均值≈0，标准差≈1
print("randn 均值=%.3f 标准差=%.3f 范围=[%.2f, %.2f]" %
      (a3.mean(), a3.std(), a3.min(), a3.max()))

print()
print("=" * 60)
print("2. np.random.rand(d0, d1, ...)  —— 均匀分布 [0, 1)")
print("=" * 60)
# 同样是括号里列维度，但值在 0~1 之间均匀分布
b0 = np.random.rand()
b1 = np.random.rand(5)
b2 = np.random.rand(4, 5)
b3 = np.random.rand(3, 4, 5)
print("标量   :", b0)
print("一维   : shape", b1.shape, "值:", np.round(b1, 2))
print("二维   : shape", b2.shape)
print("三维   : shape", b3.shape)
# 特征：均值≈0.5，范围在 0~1 之间
print("rand 均值=%.3f 标准差=%.3f 范围=[%.2f, %.2f]" %
      (b3.mean(), b3.std(), b3.min(), b3.max()))

print()
print("=" * 60)
print("3. rng.normal(mean, std, size)  —— 指定均值和标准差的正态分布")
print("=" * 60)
rng = np.random.default_rng(42)   # 固定种子可复现
# 注意：第三个参数是"形状元组"，要用 () 包起来
c0 = rng.normal(0, 1)                    # 不写 size → 标量
c1 = rng.normal(5, 2, size=5)           # 一维，均值5标准差2
c2 = rng.normal(5, 2, size=(4, 5))      # 二维
c3 = rng.normal(5, 2, size=(3, 4, 5))   # 三维
print("标量   :", c0)
print("一维   : shape", c1.shape, "值:", np.round(c1, 2))
print("二维   : shape", c2.shape)
print("三维   : shape", c3.shape)
# 特征：均值≈5，标准差≈2
print("normal(5,2) 均值=%.3f 标准差=%.3f 范围=[%.2f, %.2f]" %
      (c3.mean(), c3.std(), c3.min(), c3.max()))

print()
print("=" * 60)
print("4. 关键区别：形状参数怎么传")
print("=" * 60)
print("randn(24, 4, 5)        ✅ 维度分开写")
print("randn((24, 4, 5))      ❌ 不能传元组，会报错")
print("normal(0, 2, (24,4,5)) ✅ 形状用元组")
print()
print("randn(24,4,5)*2 + 5    =  normal(5, 2, (24,4,5))  （两种写法等价）")

print()
print("=" * 60)
print("5. 小验证：randn*2+5 是否真的等于 normal(5,2,...)")
print("=" * 60)
rng2 = np.random.default_rng(1)
x = rng2.standard_normal((10000,)) * 2 + 5
rng3 = np.random.default_rng(1)
y = rng3.normal(5, 2, size=10000)
print("randn*2+5  → 均值=%.3f 标准差=%.3f" % (x.mean(), x.std()))
print("normal(5,2) → 均值=%.3f 标准差=%.3f" % (y.mean(), y.std()))
print("两者等价？", np.allclose(x, y))

print("\n运行完成。")
