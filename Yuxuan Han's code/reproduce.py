#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QXU6028 Polymer Devices 2026-27 - Group G12 one-command pipeline.

Computes L1 (forward carrier counting), L2 (zero-D inverse), L3 (two-D inverse)
from the untouched group data, fills results_template.json exactly, and writes
all checkpoint figures/notes/logs.

Run in VSCode terminal (from workspace root):
    python reproduce.py --only l1                       # today's task
    python reproduce.py --spec <threshold>              # full run once L3 spec known
Blind test (frozen code, only change the data path):
    python reproduce.py --data-dir data/blind_GNN --out-dir outputs_blind \
        --results-name blind_results.json --spec <threshold>

Method follows QXU6028_Virtual_Lab_Practical_Methods_Guide_2026-27 (pp.3-14).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive, safe in VSCode
import matplotlib.pyplot as plt

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ---------------------------------------------------------------- constants
KB = 8.617333e-5                # eV K^-1          (guide p.3)
NA = 3.816e15                   # atoms cm^-2      (guide p.3)
E_CHARGE = 1.602176634e-19      # C                (guide p.3)
EMAX = 2.5                      # eV               (guide p.3)

SEED_L2 = 20262712              # recorded random seeds (guide p.9 / p.12)
SEED_L3 = 20262713
N_BOOT = 200
N_NOISE = 500


# ------------------------------------------------------------------ helpers
class Tee:
    """Mirror stdout into run_log.txt (needed for the blind-test upload)."""
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")
        self.so = sys.stdout

    def write(self, s):
        self.so.write(s)
        self.f.write(s)

    def flush(self):
        self.so.flush()
        self.f.flush()


def fermi(E, EF, T):
    x = np.clip((E - EF) / (KB * T), -60.0, 60.0)     # clip, guide p.3
    return 1.0 / (np.exp(x) + 1.0)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        print("[WARN] not a git repo -> pipeline_commit will be a placeholder. "
              "Run 'git init && git commit' before the real run.")
        return "NO_COMMIT_FOUND"


def load_template(data_dir: Path) -> dict:
    for p in (data_dir / "results_template.json",
              data_dir.parent / "results_template.json",
              Path("results_template.json")):
        if p.exists():
            print(f"[template] using {p}")
            return json.loads(p.read_text(encoding="utf-8"))
    print("[WARN] results_template.json not found - building minimal skeleton")
    return {
        "schema_version": "QXU6028-results-v1", "group": "G12",
        "dataset_set": "regular", "pipeline_commit": "REPLACE_WITH_GIT_COMMIT_HASH",
        "L1": {"correction_factor": None, "n_fermi_op_cm2": None,
               "mobility_corrected_cm2_Vs": None},
        "L2": {"model": None, "Eg_kT": None, "Eg_uncertainty_kT": None,
               "EF_trajectory_eV": []},
        "L3": {"map": [], "features": [], "resolution_probe_pitch": None,
               "detection_limit_fraction": None, "pass": None, "confidence": None},
    }


def fill_section(results: dict, section: str, values: dict) -> None:
    """Write computed values into the template's L1/L2/L3 sub-dict,
    preserving the supplied keys/nesting (guide p.14)."""
    tgt = results.setdefault(section, {})
    for k, v in values.items():
        tgt[k] = v


def fill_commit(results: dict, commit: str) -> None:
    if results.get("pipeline_commit", "REPLACE_WITH_GIT_COMMIT_HASH") \
            == "REPLACE_WITH_GIT_COMMIT_HASH" or "pipeline_commit" not in results:
        results["pipeline_commit"] = commit


def report_nulls(results: dict) -> None:
    """Scan recursively for unfilled (null) fields."""
    nulls: list[str] = []

    def scan(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            if not obj and path.endswith((".map", ".EF_trajectory_eV")):
                nulls.append(path + " (empty)")
            for i, v in enumerate(obj):
                scan(v, f"{path}[{i}]")
        elif obj is None:
            nulls.append(path)

    scan(results, "results")
    if nulls:
        print(f"[WARN] results.json still has unfilled fields:\n       "
              + "\n       ".join(nulls)
              + "\n       -> supply --spec (Session 3 threshold) and rerun.")
    else:
        print("[ok   ] results fully filled - no nulls remain.")


# ----------------------------------------------------------------------- L1
def n_band(EF, T, c):
    E = np.linspace(0.0, EMAX, 20001)
    return NA * trapezoid(c * E * fermi(E, EF, T), E)


def n_fermi(EF, T, c):
    E = np.linspace(EF, EMAX, 20001)
    return NA * trapezoid(c * E * fermi(E, EF, T), E)


def n_fermi_closed(EF, T, c):
    kT = KB * T
    return NA * c * kT * (EF * np.log(2.0) + (np.pi ** 2 / 12.0) * kT)


def mobility(n_cm2, rho):
    """1/(e n rho) in cm^2 V^-1 s^-1. NO extra 1e4 factor (guide p.4)."""
    return 1.0 / (E_CHARGE * n_cm2 * rho)


def l1_run(data_dir: Path, out_dir: Path) -> dict:
    print("\n================ L1 forward carrier counting ================")
    params = json.loads((data_dir / "L1_params.json").read_text(encoding="utf-8"))
    c, T = float(params["c"]), float(params["T"])
    EF_op, rho_op = float(params["EF_op_eV"]), float(params["rho_ohm_sq"])
    cols = params.get("columns", ["EF_eV", "n_band_cm2", "rho_ohm_sq"])
    df = pd.read_csv(data_dir / "L1_data.csv")
    for cc in cols:
        assert cc in df.columns, f"L1_data.csv missing column {cc!r}"
    EF_col, nband_col, rho_col = cols
    df = df.sort_values(EF_col).reset_index(drop=True)
    print(f"[params] c={c}  T={T} K  EF_op={EF_op:.6f} eV  rho_op={rho_op:.4f} ohm/sq")

    # --- gate: reference ratios must pass first (guide p.5) ---
    for ef, expect in [(0.0, 1.00), (0.073, 2.00), (0.100, 2.60), (0.200, 5.10)]:
        got = n_band(ef, T, c) / n_fermi(ef, T, c)
        stat = "PASS" if abs(got - expect) / expect <= 0.06 else "FAIL"
        print(f"[gate ] E_F={ef:5.3f} eV  C={got:7.4f} (expect ~{expect:.2f})  [{stat}]")
        assert stat == "PASS", "L1 gate failed - fix integration limits first!"

    EF = df[EF_col].to_numpy(float)
    nband_sup = df[nband_col].to_numpy(float)
    rho_rows = df[rho_col].to_numpy(float)
    nF = np.array([n_fermi(e, T, c) for e in EF])
    C = nband_sup / nF
    nF_closed = np.array([n_fermi_closed(e, T, c) for e in EF])
    print(f"[check] n_F numerical vs closed form: max rel diff "
          f"{np.max(np.abs(nF - nF_closed) / nF_closed):.2e}")

    def crossing(level):
        for i in range(len(EF) - 1):
            if C[i] != C[i + 1] and (C[i] - level) * (C[i + 1] - level) <= 0 \
                    and C[i + 1] > C[i]:
                return float(EF[i] + (level - C[i]) * (EF[i + 1] - EF[i])
                             / (C[i + 1] - C[i]))
        return None

    EF_C2, EF_C4 = crossing(2.0), crossing(4.0)
    print(f"[result] C=2 at E_F={EF_C2} eV ; C=4 at E_F={EF_C4} eV")

    # exact operating point - NEVER the nearest CSV row (guide p.4 step 4)
    nb_op, nF_op = n_band(EF_op, T, c), n_fermi(EF_op, T, c)
    C_op = nb_op / nF_op
    mu_band_op, mu_corr_op = mobility(nb_op, rho_op), mobility(nF_op, rho_op)
    i_near = int(np.argmin(np.abs(EF - EF_op)))
    print(f"[info ] nearest CSV row E_F={EF[i_near]:.4f} (C={C[i_near]:.3f}); "
          f"EXACT point gives C={C_op:.4f}")
    print(f"[result] n_F(op)={nF_op:.6e} cm^-2   C(op)={C_op:.6f}")
    print(f"[result] mu_band={mu_band_op:.4e}  mu_corrected={mu_corr_op:.4e} "
          f"cm^2 V^-1 s^-1  (= {C_op:.4f} x mu_band)")

        # ---------------------------------------------------------------- figure
    # four panels: (a) n_band, (b) n_F, (c) C, (d) corrected mobility
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(11.5, 8.6),
                                              sharex=True)
    ef_cf = np.linspace(min(0.0, EF.min()), max(EF.max(), EF_op) + 0.02, 400)
    nb_cf = np.array([n_band(e, T, c) for e in ef_cf])
    nf_cf = np.array([n_fermi(e, T, c) for e in ef_cf])
    nf_cf_closed = np.array([n_fermi_closed(e, T, c) for e in ef_cf])
    c_cf = nb_cf / nf_cf
    rho_cf = np.interp(ef_cf, EF, rho_rows)      # measured rho, interpolated
    mu_band_cf = mobility(nb_cf, rho_cf)
    mu_corr_cf = mobility(nf_cf, rho_cf)

    # (a) band-edge carrier density
    a1.plot(ef_cf, nb_cf, "-", lw=1.4, color="tab:blue", alpha=0.7,
            label="model: count from $E=0$")
    a1.plot(EF, nband_sup, "o", ms=5, mfc="none", color="tab:blue",
            label="data (datasheet)")
    a1.plot([EF_op], [nb_op], "*", ms=14, color="tab:red",
            label=f"operating point, $E_F$={EF_op:.3f} eV")
    a1.set_yscale("log")
    a1.set(ylabel="$n_{band}$ [cm$^{-2}$]",
           title="(a) Band-edge carrier density")
    a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=8)

    # (b) Fermi-referenced carrier density
    a2.plot(ef_cf, nf_cf, "-", lw=1.4, color="tab:green", alpha=0.7,
            label="model: count from $E=E_F$")
    a2.plot(ef_cf, nf_cf_closed, "--", lw=1.0, color="k", alpha=0.5,
            label="closed form (numerical check)")
    a2.plot(EF, nF, "o", ms=5, mfc="none", color="tab:green",
            label="model at data $E_F$")
    a2.plot([EF_op], [nF_op], "*", ms=14, color="tab:red",
            label=f"$n_F(op)$ = {nF_op:.3e} cm$^{{-2}}$")
    a2.set_yscale("log")
    a2.set(ylabel="$n_F$ [cm$^{-2}$]",
           title="(b) Fermi-referenced carrier density")
    a2.grid(alpha=0.3, which="both"); a2.legend(fontsize=8)

    # (c) carrier-count correction
    a3.plot(ef_cf, c_cf, "-", lw=1.4, color="tab:purple", alpha=0.7,
            label="model curve")
    a3.plot(EF, C, "o", ms=5, mfc="none", color="tab:purple",
            label="data C = $n_{band}/n_F$")
    for lv, ex in ((2.0, EF_C2), (4.0, EF_C4)):
        a3.axhline(lv, color="grey", ls=":", lw=0.8)
        if ex is not None:
            a3.plot([ex], [lv], "s", color="tab:red")
            a3.annotate(f"C={lv:.0f}: {ex:.3f} eV", (ex, lv), fontsize=8,
                        color="tab:red", xytext=(8, -12),
                        textcoords="offset points")
    a3.plot([EF_op], [C_op], "*", ms=14, color="tab:red",
            label=f"exact operating point, C={C_op:.3f}")
    a3.axhline(1.0, color="k", ls="--", lw=0.6, alpha=0.5)
    a3.set(xlabel="$E_F$ [eV]", ylabel="C = $n_{band}/n_F$",
           title="(c) Carrier-count correction")
    a3.grid(alpha=0.3); a3.legend(fontsize=8)

    # (d) corrected mobility (measured rho held fixed)
    a4.plot(ef_cf, mu_corr_cf, "-", lw=1.4, color="tab:green", alpha=0.8,
            label="corrected (Fermi-referenced)")
    a4.plot(EF, mobility(nF, rho_rows), "o", ms=5, mfc="none",
            color="tab:green", label="corrected at data $E_F$")
    a4.plot(ef_cf, mu_band_cf, "--", lw=1.0, color="tab:orange", alpha=0.6,
            label="band-edge (datasheet, for reference)")
    a4.plot([EF_op], [mu_corr_op], "*", ms=14, color="tab:red",
            label=f"$\\mu_{{corr}}(op)$ = {mu_corr_op:.3e}")
    a4.set_yscale("log")
    a4.set(xlabel="$E_F$ [eV]",
           ylabel=r"$\mu$ [cm$^2$ V$^{-1}$ s$^{-1}$]",
           title="(d) Corrected mobility (measured $\\rho$)")
    a4.grid(alpha=0.3, which="both"); a4.legend(fontsize=8)

    fig.suptitle("L1 forward carrier counting - G12", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "L1_figure.png", dpi=200)
    plt.close(fig)


    (out_dir / "L1_note.md").write_text(f"""# L1 brief note (scaffold - personalise before submitting)
Method: both counts use the same linear DoS g(E)=cE and finite-temperature Fermi
function; only the lower integration limit differs (band-edge from E=0,
Fermi-referenced from E=E_F). {20001} points up to {EMAX} eV, np.trapezoid.
Why C grows with doping: n_band contains all states in [0,E_F] (~E_F^2/2) whereas
n_F = NA c kT [E_F ln2 + (pi^2/12) kT] only sees the thermal window (linear in E_F).
Crossings: C=2 at {EF_C2} eV, C=4 at {EF_C4} eV.
Exact operating point (EF_op={EF_op:.6f} eV, rho={rho_op:.4f} ohm/sq):
  n_band={nb_op:.6e}  n_F={nF_op:.6e} cm^-2  C={C_op:.6f}
  mu_band={mu_band_op:.6e} -> mu_corrected={mu_corr_op:.6e} cm^2 V^-1 s^-1 (= C x mu_band)
Sensor advice: Hall sensitivity scales as 1/n at fixed current, so the datasheet
under-states it by the same factor C={C_op:.3f} (~{(C_op - 1) * 100:.1f}% too low).
[TODO: group's own commentary/caveats.]
""", encoding="utf-8")

    (out_dir / "l1_diagnostics.json").write_text(json.dumps({
        "c": c, "T": T, "EF_op_eV": EF_op, "rho_ohm_sq": rho_op,
        "n_band_op_cm2": float(nb_op), "n_fermi_op_cm2": float(nF_op),
        "correction_factor": float(C_op),
        "mobility_band_cm2_Vs": float(mu_band_op),
        "mobility_corrected_cm2_Vs": float(mu_corr_op),
        "EF_C_eq_2_eV": EF_C2, "EF_C_eq_4_eV": EF_C4}, indent=2),
        encoding="utf-8")

    return {"correction_factor": float(C_op),
            "n_fermi_op_cm2": float(nF_op),
            "mobility_corrected_cm2_Vs": float(mu_corr_op)}


# ----------------------------------------------------------------------- L2
def dos(E, model, c, Eg):
    if model == "cone":
        return c * np.abs(E)
    D = Eg / 2
    return np.where(np.abs(E) > D, c * (np.abs(E) - D), 0.0)


def build_model_library(c: float, T: float) -> list[dict]:
    """Precompute normalised model curves once; reuse in bootstrap (guide p.7)."""
    kBT = KB * T
    EF_grid = np.linspace(-8 * kBT, 9 * kBT, 400)
    E = np.linspace(-EMAX, EMAX, 4000)
    f = fermi(E[:, None], EF_grid[None, :], T)
    gaps = np.round(np.arange(0.5, 8.0 + 1e-9, 0.25), 2)
    lib = []
    for model, Eg in [("cone", 0.0)] + [("hard_gap", g * kBT) for g in gaps]:
        g = dos(E, model, c, Eg)[:, None]
        above = E[:, None] >= EF_grid[None, :]
        n = NA * trapezoid(np.where(above, g * f, 0.0), E, axis=0)
        p = NA * trapezoid(np.where(~above, g * (1 - f), 0.0), E, axis=0)
        RH = (p - n) / (p + n) ** 2                 # e = mu = 1, scale cancels
        RS = 1.0 / (p + n)
        lib.append({"model": model, "Eg_kT": float(Eg / kBT), "EF": EF_grid,
                    "xy": np.column_stack([RH / np.max(np.abs(RH)),
                                           RS / np.max(RS)])})
    return lib


def fit_curve(data_xy, model_xy):
    d2 = ((data_xy[:, None, :] - model_xy[None, :, :]) ** 2).sum(axis=2)
    j = np.argmin(d2, axis=1)
    return float(np.sqrt(np.mean(np.min(d2, axis=1)))), j


def select_model(data_xy, lib):
    """Cone vs best hard gap with the 15% rule (guide p.8)."""
    r_cone, j_cone = fit_curve(data_xy, lib[0]["xy"])
    fits = [(fit_curve(data_xy, e["xy"])[0], e, fit_curve(data_xy, e["xy"])[1])
            for e in lib[1:]]
    r_gap, gap_entry, j_gap = min(fits, key=lambda t: t[0])
    if r_gap < 0.85 * r_cone and gap_entry["Eg_kT"] >= 1.0:
        return gap_entry, j_gap, r_gap, r_cone, gap_entry
    return lib[0], j_cone, r_cone, r_cone, min(fits, key=lambda t: t[0])[1]


def l2_run(data_dir: Path, out_dir: Path, c: float) -> dict:
    print("\n========== L2 zero-dimensional inverse problem ==========")
    meta = json.loads((data_dir / "L2_meta.json").read_text(encoding="utf-8"))
    T = float(meta["T_K"]); kBT = KB * T
    idx_c, rep_c, rh_c, rxx_c = meta["columns"]
    df = pd.read_csv(data_dir / "L2_data.csv")
    RH = df.pivot_table(index=idx_c, columns=rep_c, values=rh_c,
                        aggfunc="mean").sort_index().to_numpy(float)
    RXX = df.pivot_table(index=idx_c, columns=rep_c, values=rxx_c,
                         aggfunc="mean").sort_index().to_numpy(float)
    order = np.array(sorted(df[idx_c].unique()))
    n_idx, n_rep = RH.shape
    print(f"[data ] {n_idx} measurement indices x {n_rep} repeats, T={T} K")
    if n_rep != 3:
        print(f"[WARN ] expected 3 repeats, found {n_rep}")

    RH_m, RXX_m = RH.mean(axis=1), RXX.mean(axis=1)
    bracket = None
    for i in range(n_idx - 1):                    # zero-crossing BEFORE normalising
        if RH_m[i] * RH_m[i + 1] < 0:
            bracket = (int(order[i]), int(order[i + 1])); break
    print(f"[gate ] R_H sign change bracketed by indices {bracket}")

    def norm(rh, rs):
        return rh / np.max(np.abs(rh)), rs / np.max(rs)   # separate normalisation

    rhz, rsz = norm(RH_m, RXX_m)
    assert np.min(rhz) < 0 < np.max(rhz), "normalised R_H must still cross zero"
    data_xy = np.column_stack([rhz, rsz])

    lib = build_model_library(c, T)
    sel, jn, rms_sel, r_cone, best_gap_entry = select_model(data_xy, lib)
    best_gap_r = fit_curve(data_xy, best_gap_entry["xy"])[0]
    print(f"[fit  ] r_cone={r_cone:.5f}  best hard-gap r={best_gap_r:.5f} "
          f"(Eg={best_gap_entry['Eg_kT']:.2f} kBT)")
    print(f"[fit  ] 15% rule -> selected {sel['model']}, "
          f"Eg={sel['Eg_kT']:.2f} kBT, rms={rms_sel:.5f}")
    EF_traj = sel["EF"][jn]
    assert len(EF_traj) == n_idx, "trajectory length must equal #unique indices"

    # bootstrap uncertainty (guide p.9)
    rng = np.random.default_rng(SEED_L2)
    eg_samples, gap_flags = [], []
    EF_boot = np.empty((N_BOOT, n_idx))
    for b in range(N_BOOT):
        ch = rng.integers(0, n_rep, size=n_idx)
        xy_b = np.column_stack(norm(RH[np.arange(n_idx), ch],
                                    RXX[np.arange(n_idx), ch]))
        s_b, j_b, *_ = select_model(xy_b, lib)
        gap_flags.append(s_b["model"] == "hard_gap")
        eg_samples.append(s_b["Eg_kT"] if s_b["model"] == "hard_gap" else 0.0)
        EF_boot[b] = s_b["EF"][j_b]
    eg_samples = np.asarray(eg_samples)
    p16, p50, p84 = np.percentile(eg_samples, [16, 50, 84])
    frac_gap = float(np.mean(gap_flags))
    flag = " -> UNCERTAIN model decision" if 0.35 < frac_gap < 0.65 else ""
    print(f"[boot ] Eg/kBT median={p50:.2f}  16-84%=({p16:.2f},{p84:.2f})  "
          f"hard-gap fraction={frac_gap:.2f}{flag}")
    Eg_out = 0.0 if sel["model"] == "cone" else float(sel["Eg_kT"])

    # figures
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    r_c, _ = fit_curve(data_xy, lib[0]["xy"])
    a1.plot(lib[0]["xy"][:, 0], lib[0]["xy"][:, 1], "--", lw=1,
            label=f"cone (r={r_c:.4f})")
    a1.plot(best_gap_entry["xy"][:, 0], best_gap_entry["xy"][:, 1], ":", lw=1,
            label=f"hard gap {best_gap_entry['Eg_kT']:.2f} kBT (r={best_gap_r:.4f})")
    a1.plot(rhz, rsz, "o", ms=5, color="tab:red", label="mean data")
    a1.set(xlabel="normalised R_H", ylabel="normalised R_s",
           title=f"L2 fit: {sel['model']}, Eg={Eg_out:.2f} kBT")
    a1.grid(alpha=0.3); a1.legend(fontsize=8)
    for r in range(n_rep):
        a2.plot(order, RH[:, r], "o-", ms=3, lw=0.8, label=f"repeat {r + 1}")
    if bracket:
        a2.axvspan(bracket[0], bracket[1], color="tab:red", alpha=0.15,
                   label="sign-change bracket")
    a2.axhline(0, color="k", lw=0.6)
    a2.set(xlabel="measurement_index", ylabel="raw R_H",
           title="raw repeats, sign preserved")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "L2_model_figure.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    lo, hi = np.percentile(EF_boot, [16, 84], axis=0)
    ax.fill_between(order, lo, hi, alpha=0.25, label="bootstrap 16-84%")
    ax.plot(order, EF_traj, "o-", color="tab:blue", label="assigned E_F")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set(xlabel="measurement_index", ylabel="assigned E_F [eV]",
           title=f"L2 Fermi-level trajectory ({sel['model']}, Eg={Eg_out:.2f} kBT)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "L2_trajectory_figure.png", dpi=200); plt.close(fig)

    (out_dir / "L2_note.md").write_text(f"""# L2 brief note (scaffold - personalise)
Selected: {sel['model']}, Eg={Eg_out:.2f} kBT (bootstrap median {p50:.2f},
16-84% = ({p16:.2f},{p84:.2f}), hard-gap fraction {frac_gap:.2f} over {N_BOOT}
replicates, seed {SEED_L2}). r_cone={r_cone:.5f} vs best-gap r={best_gap_r:.5f};
15% rule applied. Neutrality bracket: indices {bracket}.
Trajectory length = {n_idx} = number of unique measurement indices.
[TODO: device recommendation (sensor/interconnect vs switching); state honestly
if a sub-thermal gap (Eg < 1 kBT) cannot be distinguished from gapless.]
""", encoding="utf-8")

    (out_dir / "l2_diagnostics.json").write_text(json.dumps({
        "model": sel["model"], "Eg_kT": Eg_out, "Eg_median_kT": float(p50),
        "Eg_p16": float(p16), "Eg_p84": float(p84),
        "hard_gap_fraction": frac_gap, "r_cone": r_cone,
        "r_best_gap": best_gap_r, "rms_selected": rms_sel,
        "sign_change_bracket": bracket, "seed": SEED_L2, "n_boot": N_BOOT},
        indent=2), encoding="utf-8")

    return {"model": sel["model"], "Eg_kT": Eg_out,
            "Eg_uncertainty_kT": float((p84 - p16) / 2.0),
            "EF_trajectory_eV": [float(x) for x in EF_traj]}


# ----------------------------------------------------------------------- L3
def gradient_matrix(npix: int = 10) -> np.ndarray:
    rows = []
    N = npix * npix
    for i in range(npix):
        for j in range(npix):
            k = i * npix + j
            if j < npix - 1:
                r = np.zeros(N); r[k] = -1; r[k + 1] = 1; rows.append(r)
            if i < npix - 1:
                r = np.zeros(N); r[k] = -1; r[k + npix] = 1; rows.append(r)
    return np.asarray(rows)


def l3_run(data_dir: Path, out_dir: Path, spec: float | None) -> dict:
    print("\n=============== L3 two-dimensional inverse ===============")
    A = np.load(data_dir / "L3_A.npy")
    d_reps = np.loadtxt(data_dir / "L3_d.csv", delimiter=",")
    assert A.shape == (128, 100), f"A shape {A.shape} != (128, 100)"
    assert d_reps.shape == (3, 128), f"d shape {d_reps.shape} != (3, 128)"
    rowsum = np.max(np.abs(A.sum(axis=1)))
    assert rowsum < 1e-11, "rows of A must sum to ~0 (L3 gate)"
    print(f"[gate ] shapes OK; max |row sum of A| = {rowsum:.2e} -> "
          f"zero-mean constraint required")
    dbar = d_reps.mean(axis=0)

    N = A.shape[1]
    L = gradient_matrix(10)
    assert L.shape == (180, 100)
    one = np.ones((N, 1)) / np.sqrt(N)
    P = np.eye(N) - one @ one.T
    eta0 = 1e-2 * np.linalg.norm(A, 2) ** 2 / np.linalg.norm(L, 2) ** 2
    print(f"[eta  ] eta0 = {eta0:.4e}")

    def reconstruct(dvec, eta=eta0):
        M = P @ (A.T @ A + eta * (L.T @ L)) @ P + 1e-9 * np.eye(N)
        b = P @ (A.T @ dvec)
        m = np.linalg.lstsq(M, b, rcond=None)[0]
        return m - m.mean()                       # impose mean(m)=0 (guide p.11)

    mhat = reconstruct(dbar).reshape(10, 10)
    assert np.all(np.isfinite(mhat)) and abs(mhat.mean()) < 1e-12
    a_amp = float(np.max(np.abs(mhat)))
    print(f"[map  ] max|m| = {a_amp:.4f}   mean(m) = {mhat.mean():.2e}")

    # regularisation sensitivity (guide p.11)
    sens = []
    for factor in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = reconstruct(dbar, eta=factor * eta0)
        sens.append((factor, float(np.linalg.norm(A @ m - dbar)),
                     float(np.linalg.norm(L @ m))))
    print("[sens ] factor  residual||Am-d||  roughness||Lm||")
    for factor, r, ro in sens:
        print(f"        {factor:6.2f}   {r:10.4f}   {ro:10.4f}")

    # resolution test (guide p.12)
    centres = np.linspace(-2.7, 2.7, 10)

    def fwhm_at(i0, j0):
        q = np.zeros(100); q[i0 * 10 + j0] = 1.0; q -= q.mean()
        rec = reconstruct(A @ q).reshape(10, 10)
        w = np.maximum(rec, 0.0)
        ii, jj = np.mgrid[0:10, 0:10]
        r2 = (centres[jj] - centres[j0]) ** 2 + (centres[ii] - centres[i0]) ** 2
        sigma2 = float(np.sum(w * r2) / (2 * np.sum(w)))
        return 2.355 * np.sqrt(sigma2), rec

    fw_c, rec_c = fwhm_at(4, 4)          # central pixel
    fw_o, rec_o = fwhm_at(1, 8)          # off-centre pixel
    resolution = max(fw_c, fw_o)         # conservative, guide p.12
    print(f"[res  ] FWHM central={fw_c:.3f}  off-centre={fw_o:.3f}  -> "
          f"resolution = {resolution:.3f} probe pitches")

    # noise-linked detection limit (guide p.12)
    sigma_j = d_reps.std(axis=0, ddof=1)
    rng = np.random.default_rng(SEED_L3)
    noise = rng.normal(0.0, sigma_j, size=(N_NOISE, 128))
    maxs = np.array([np.max(np.abs(reconstruct(noise[k])))
                     for k in range(N_NOISE)])
    delta95 = float(np.percentile(maxs, 95))
    print(f"[det  ] median sigma_j={np.median(sigma_j):.4f}  "
          f"delta_95={delta95:.4f}  (seed {SEED_L3})")

    # features: merge neighbouring pixels, |amp| >= delta95 (guide p.11)
    def components(mask):
        lab = np.zeros(mask.shape, int); cur = 0
        for i in range(10):
            for j in range(10):
                if mask[i, j] and lab[i, j] == 0:
                    cur += 1; q = deque([(i, j)]); lab[i, j] = cur
                    while q:
                        a, b = q.popleft()
                        for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            x, y = a + da, b + db
                            if 0 <= x < 10 and 0 <= y < 10 \
                                    and mask[x, y] and lab[x, y] == 0:
                                lab[x, y] = cur; q.append((x, y))
        return lab, cur

    features = []
    for sign, mask in (("positive", mhat >= delta95),
                       ("negative", mhat <= -delta95)):
        if not mask.any():
            continue
        lab, n = components(mask)
        for k in range(1, n + 1):
            selm = lab == k
            pix = np.argwhere(selm)[np.argmax(np.abs(mhat[selm]))]
            i, j = int(pix[0]), int(pix[1])
            features.append({"sign": sign, "row": i, "col": j,
                             "x_probe_pitch": float(centres[j]),
                             "y_probe_pitch": float(centres[i]),
                             "amplitude": float(mhat[i, j]),
                             "pixels_merged": int(selm.sum())})
    features.sort(key=lambda f: -abs(f["amplitude"]))
    for ft in features:
        print(f"[feat ] {ft['sign']:8s} amp={ft['amplitude']:+.4f} at "
              f"(x={ft['x_probe_pitch']:+.2f}, y={ft['y_probe_pitch']:+.2f}) "
              f"pixel({ft['row']},{ft['col']}), {ft['pixels_merged']} px")

    # pass / fail with confidence (guide p.12)
    if spec is None:
        passed, conf = None, ("PENDING - rerun with --spec <threshold announced "
                              "during Session 3>")
        print("[dec  ] no specification threshold supplied - pass left unfilled")
    else:
        if a_amp + delta95 < spec:
            passed, conf = True, "clear pass"
        elif a_amp - delta95 > spec:
            passed, conf = False, "clear fail"
        else:
            passed, conf = False, "borderline"
        print(f"[dec  ] a={a_amp:.4f}  delta_95={delta95:.4f}  spec={spec}  -> {conf}")

    # figures
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    vmax = max(a_amp, delta95)
    im = ax.imshow(mhat, origin="lower",
                   extent=[centres[0] - 0.3, centres[-1] + 0.3,
                           centres[0] - 0.3, centres[-1] + 0.3],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for ft in features:
        ax.plot(ft["x_probe_pitch"], ft["y_probe_pitch"], "k*",
                ms=12 if abs(ft["amplitude"]) == a_amp else 8)
        ax.annotate(f"{ft['amplitude']:+.2f}",
                    (ft["x_probe_pitch"], ft["y_probe_pitch"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="x [probe pitch]", ylabel="y [probe pitch]",
           title="L3 zero-mean non-uniformity map (eta = eta0)")
    fig.colorbar(im, ax=ax, label="map value")
    fig.tight_layout()
    fig.savefig(out_dir / "L3_map_figure.png", dpi=200); plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.loglog([s[2] for s in sens], [s[1] for s in sens], "o-")
    for factor, r, ro in sens:
        a1.annotate(f"x{factor:g}", (ro, r), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    a1.set(xlabel="roughness ||Lm||", ylabel="residual ||Am-d||",
           title="eta sensitivity (reference: eta0)")
    a1.grid(alpha=0.3, which="both")
    a2.plot(centres, rec_c[4, :], "o-", ms=3, label=f"central pixel FWHM={fw_c:.2f}")
    a2.plot(centres, rec_o[1, :], "s-", ms=3, label=f"off-centre FWHM={fw_o:.2f}")
    a2.axhline(0, color="k", lw=0.6)
    a2.set(xlabel="position [probe pitch]", ylabel="reconstructed amplitude",
           title=f"point-spread test; resolution = {resolution:.2f} pitches")
    a2.grid(alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "L3_method_figure.png", dpi=200); plt.close(fig)

    (out_dir / "L3_note.md").write_text(f"""# L3 brief note (scaffold - personalise)
Noise: channel-wise repeat std median = {np.median(sigma_j):.4f}.
Regularisation eta0 = {eta0:.4e} (guide's reproducible formula; sensitivity
x0.01..x100 shown in the method figure).
Resolution = {resolution:.3f} probe pitches (max of central {fw_c:.2f} and
off-centre {fw_o:.2f} FWHM).
Detection limit delta_95 = {delta95:.4f} ({N_NOISE} noise reconstructions,
seed {SEED_L3}). max|m_hat| = {a_amp:.4f}.
Decision: {conf} (spec threshold = {spec}).
[TODO: physical interpretation of the features; comment on how the decision
degrades with noise.]
""", encoding="utf-8")

    (out_dir / "l3_diagnostics.json").write_text(json.dumps({
        "eta0": float(eta0), "sensitivity": sens,
        "fwhm_central": float(fw_c), "fwhm_offcentre": float(fw_o),
        "resolution_probe_pitch": float(resolution), "delta_95": delta95,
        "max_abs_map": a_amp, "seed": SEED_L3}, indent=2), encoding="utf-8")

    return {"map": [float(v) for v in mhat.reshape(-1)],   # row-major, 100 values
            "features": features,
            "resolution_probe_pitch": float(resolution),
            "detection_limit_fraction": delta95,
            "pass": passed, "confidence": conf}


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="QXU6028 G12 pipeline (L1+L2+L3)")
    ap.add_argument("--data-dir", default="data/regular_GNN",
                    help="group data folder (change only this for the blind test)")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--results-name", default="results.json")
    ap.add_argument("--spec", type=float, default=None,
                    help="pass/fail threshold announced during Session 3")
    ap.add_argument("--only", choices=["l1", "l2", "l3", "all"], default="all")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    if not data_dir.exists():
        sys.exit(f"[ERROR] data dir not found: {data_dir.resolve()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run_log.txt")

    print(f"[run  ] data={data_dir}  out={out_dir}  only={args.only}  spec={args.spec}")
    l1_params = json.loads((data_dir / "L1_params.json").read_text(encoding="utf-8"))
    c = float(l1_params["c"])          # same sample -> same DoS slope in L2

    sections: dict[str, dict] = {}
    if args.only in ("l1", "all"):
        sections["L1"] = l1_run(data_dir, out_dir)
    if args.only in ("l2", "all"):
        sections["L2"] = l2_run(data_dir, out_dir, c)
    if args.only in ("l3", "all"):
        sections["L3"] = l3_run(data_dir, out_dir, args.spec)

    results = load_template(data_dir)
    commit = git_hash()
    fill_commit(results, commit)
    for sec, vals in sections.items():
        fill_section(results, sec, vals)

    (out_dir / args.results_name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "commit_hash.txt").write_text(commit + "\n", encoding="utf-8")

    report_nulls(results)
    print(f"[done ] commit {commit}")
    print(f"[done ] outputs in {out_dir}: "
          f"{sorted(p.name for p in out_dir.iterdir())}")


if __name__ == "__main__":
    main()
