QXU6028 VIRTUAL-LAB DATA - G12

This pack is unique to G12. Do not copy numerical results from another group.

L1: L1_data.csv and L1_params.json
L2: L2_data.csv and L2_meta.json
L3: L3_A.npy, L3_d.csv and L3_meta.json

Use results_template.json as the machine-readable output schema.

Verified L1 implementation checks (using the supplied model at 300 K):
- n_band/n_Fermi is 1.00 at EF = 0 (neutrality).
- n_band/n_Fermi is approximately 2.60 at EF = 0.10 eV.
- n_band/n_Fermi is approximately 5.10 at EF = 0.20 eV.
- The ratio is monotonic over the supplied positive-EF sweep.

The exact equations and fitting procedure for L1-L3 are in
QXU6028_Virtual_Lab_Practical_Methods_Guide_2026-27.pdf in the shared student pack.

Verified L3 implementation check:
- Each row of the supplied projected forward matrix sums to numerical zero.
- Your reconstructed map must be constrained to have zero mean.

Deadline for the final group submission: Tuesday 24 November 2026, 22:00 UK time.
