import numpy as np
import pandas as pd
import json

# ---------- 物理常数 ----------
kB_eV_K = 8.617333262e-5      # 玻尔兹曼常数 eV/K
e_C = 1.602176634e-19         # 电子电荷 C

# ---------- 加载数据 ----------
params = json.load(open('E:/西工大/NPU课程/高分子器件QXU6028/Coursework/QXU6028_G12_VirtualLab/regular_G12/data/L1_params.json'))
data = pd.read_csv('E:/西工大/NPU课程/高分子器件QXU6028/Coursework/QXU6028_G12_VirtualLab/regular_G12/data/L1_data.csv')

c = params['c']               # 线性色散系数 eV·m
T = params['T']               # 温度 K
EF_op = params['EF_op_eV']    # 工作点费米能级 eV
rho = params['rho_ohm_sq']    # 方块电阻 Ω/sq

# ---------- 函数定义 ----------

def n_band(EF, T, c, E_max=2.0, npts=10000):
    hbar = 6.582119569e-16
    E_grid = np.linspace(0, E_max, npts)
    g = 2 * E_grid / (np.pi * (hbar * c)**2)
    f = 1 / (np.exp((E_grid - EF) / (kB_eV_K * T)) + 1)
    return np.trapezoid(g * f, E_grid)
    
    return np.trapezoid(g * f, E_grid)

def n_fermi(EF, T, c, E_max=2.0, npts=20000):
    """
    费米参考计数：从 EF 积分到 E_max
    """
    hbar = 6.582119569e-16
    E_grid = np.linspace(EF, E_max, npts)
    dE = E_grid[1] - E_grid[0]
    
    # 态密度 g(E) = 2|E| / (π * (hbar*c)**2)
    g = 2 * np.abs(E_grid) / (np.pi * (hbar * c)**2)
    
    f = 1 / (np.exp((E_grid - EF) / (kB_eV_K * T)) + 1)
    
    return np.trapezoid(g * f, E_grid)

# ---------- 验证参考比值 ----------
test_EFs = [0.0, 0.073, 0.10, 0.20]
for ef in test_EFs:
    nb = n_band(ef, T, c)
    nf = n_fermi(ef, T, c)
    ratio = nb / nf
    print(f"EF={ef:.3f} eV: n_band={nb:.4e}, n_fermi={nf:.4e}, C={ratio:.4f}")

# ---------- 计算工作点的修正因子 ----------
EF_op = params['EF_op_eV']
rho = params['rho_ohm_sq']

# 计算工作点的 n_band 和 n_fermi
n_band_op = n_band(EF_op, T, c)
n_fermi_op = n_fermi(EF_op, T, c)

# 修正因子 C
C = n_band_op / n_fermi_op
print(f"\n工作点 EF_op = {EF_op:.6f} eV")
print(f"n_band = {n_band_op:.6e} cm^-2")
print(f"n_fermi = {n_fermi_op:.6e} cm^-2")
print(f"修正因子 C = {C:.6f}")

# ---------- 修正载流子密度 ----------
# Hall 测得的 n_band 需要除以 C 才能得到真实的 n_fermi
# 使用数据中的第一个 n_band 值作为示例
n_band_first = data['n_band_cm2'].iloc[0]
n_fermi_corrected = n_band_first / C
print(f"\n修正后载流子密度 n_fermi = {n_fermi_corrected:.6e} cm^-2")

# ---------- 计算修正后的迁移率 ----------
# 电导迁移率公式：μ = 1 / (e * n * ρ)
# 其中 e 是电子电荷，ρ 是方块电阻
e_charge = 1.602176634e-19  # C
mu_corrected = 1 / (e_charge * n_fermi_corrected * rho)
print(f"修正后迁移率 μ = {mu_corrected:.6e} cm^2/(V·s)")

import numpy as np

# 常数
hbar_eVs = 6.582119569e-16    # eV·s
c_eV_m = 0.046329815372874394 # eV/m（来自 params.json）

# 在 E = 0.1 eV 处的态密度
E = 0.1
g = 2 * E / (np.pi * (hbar_eVs * c_eV_m)**2)
print(f"g(E=0.1 eV) = {g:.6e} states/(eV·m²)")
print(f"若换算为 cm²: g = {g * 1e-4:.6e} states/(eV·cm²)")
