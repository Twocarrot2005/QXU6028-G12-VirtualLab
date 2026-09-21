# L1 brief note (scaffold - personalise before submitting)
Method: both counts use the same linear DoS g(E)=cE and finite-temperature Fermi
function; only the lower integration limit differs (band-edge from E=0,
Fermi-referenced from E=E_F). 20001 points up to 2.5 eV, np.trapezoid.
Why C grows with doping: n_band contains all states in [0,E_F] (~E_F^2/2) whereas
n_F = NA c kT [E_F ln2 + (pi^2/12) kT] only sees the thermal window (linear in E_F).
Crossings: C=2 at 0.07265174803587815 eV, C=4 at 0.15730158118131016 eV.
Exact operating point (EF_op=0.110610 eV, rho=324.2386 ohm/sq):
  n_band=1.274237e+12  n_F=4.475962e+11 cm^-2  C=2.846845
  mu_band=1.510687e+04 -> mu_corrected=4.300693e+04 cm^2 V^-1 s^-1 (= C x mu_band)
Sensor advice: Hall sensitivity scales as 1/n at fixed current, so the datasheet
under-states it by the same factor C=2.847 (~184.7% too low).
[TODO: group's own commentary/caveats.]
