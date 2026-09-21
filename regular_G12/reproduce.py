import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 物理常数 ----------
kB_eV_K = 8.617333262e-5
hbar_eVs = 6.582119569e-16
e_charge = 1.602176634e-19

# ---------- 加载数据 ----------
with open('data/L1_params.json') as f:
    params = json.load(f)

data = pd.read_csv('data/L1_data.csv')

c = params['c']
T = params['T']
EF_op = params['EF_op_eV']
rho = params['rho_ohm_sq']

# ---------- 核心函数 ----------
def n_band(EF, T, c, E_max=2.0, npts=10000):
    E_grid = np.linspace(0, E_max, npts)
    g = 2 * E_grid / (np.pi * (hbar_eVs * c)**2)
    f = 1 / (np.exp((E_grid - EF) / (kB_eV_K * T)) + 1)
    return np.trapezoid(g * f, E_grid)

def n_fermi(EF, T, c, E_max=2.0, npts=20000):
    E_grid = np.linspace(EF, E_max, npts)
    g = 2 * np.abs(E_grid) / (np.pi * (hbar_eVs * c)**2)
    f = 1 / (np.exp((E_grid - EF) / (kB_eV_K * T)) + 1)
    return np.trapezoid(g * f, E_grid)

# ---------- 门禁验证 ----------
print("=== 门禁验证 ===")
for ef in [0.0, 0.073, 0.10, 0.20]:
    nb = n_band(ef, T, c)
    nf = n_fermi(ef, T, c)
    print(f"EF={ef:.3f} eV: C={nb/nf:.4f}")

# ---------- 计算工作点 ----------
C = n_band(EF_op, T, c) / n_fermi(EF_op, T, c)
n_band_at_EF_op = data[data['EF_eV'] == 0.11]['n_band_cm2'].values[0]
n_fermi_op = n_band_at_EF_op / C
mu_corrected = 1 / (e_charge * n_fermi_op * rho)

print(f"\n=== L1 结果 ===")
print(f"修正因子 C = {C:.6f}")
print(f"n_fermi_op = {n_fermi_op:.6e} cm^-2")
print(f"修正后迁移率 μ = {mu_corrected:.6e} cm^2/(V·s)")

# ---------- 绘图 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
EF_values = data['EF_eV'].values
C_values = []
for ef in EF_values:
    nb = n_band(ef, T, c)
    nf = n_fermi(ef, T, c)
    C_values.append(nb / nf)

plt.figure(figsize=(8, 5))
plt.plot(EF_values, C_values, 'b-o', markersize=4, label='C(EF)')
plt.axhline(y=2, color='gray', linestyle='--', alpha=0.7, label='C=2')
plt.axhline(y=4, color='gray', linestyle='--', alpha=0.7, label='C=4')
plt.axvline(x=EF_op, color='red', linestyle=':', alpha=0.7, label=f'EF_op = {EF_op:.4f} eV')
plt.scatter([EF_op], [C], color='red', s=80, zorder=5, label=f'工作点 C={C:.3f}')
plt.xlabel('EF (eV)')
plt.ylabel('Correction Factor C')
plt.title('L1: Correction Factor vs Fermi Level')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(r'E:\西工大\NPU课程\高分子器件QXU6028\Coursework\QXU6028_G12_VirtualLab\regular_G12\L1_C_vs_EF.png', dpi=150, bbox_inches='tight')
plt.close()
print("L1_C_vs_EF.png 已保存")

# ---------- 保存 JSON ----------
results = {
    "schema_version": "QXU6028-results-v1",
    "group": "G12",
    "dataset_set": "regular",
    "pipeline_commit": "REPLACE_WITH_GIT_COMMIT_HASH",
    "L1": {
        "correction_factor": float(C),
        "n_fermi_op_cm2": float(n_fermi_op),
        "mobility_corrected_cm2_Vs": float(mu_corrected)
    },
    "L2": {
        "model": None,
        "Eg_kT": None,
        "Eg_uncertainty_kT": None,
        "EF_trajectory_eV": []
    },
    "L3": {
        "map": [],
        "features": [],
        "resolution_probe_pitch": None,
        "detection_limit_fraction": None,
        "pass": None,
        "confidence": None
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nresults.json 已生成")