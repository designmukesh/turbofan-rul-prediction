# Data Dictionary — NASA C-MAPSS FD001

**Dataset:** C-MAPSS Turbofan Engine Degradation Simulation Data Set
**Publisher:** NASA Ames Prognostics Center of Excellence (Saxena & Goebel, 2008)
**Source:** PHM Society data repository mirror (S3)
**License:** Public / research use (cite Saxena et al., 2008)
**Nature:** *Simulated* run-to-failure trajectories from the C-MAPSS
thermodynamic engine model — not field/operational data.

## File layout

| File | Contents |
|---|---|
| `train_FD001.txt` | Full run-to-failure trajectories, 100 engines, 20,631 cycles |
| `test_FD001.txt` | Truncated trajectories, 100 engines (failure point hidden) |
| `RUL_FD001.txt` | True remaining-useful-life (cycles) for each test engine at truncation |

Each row = one flight cycle of one engine. Columns are space-separated, no header.

## Columns

| Column | Meaning | Units / notes |
|---|---|---|
| `unit` | Engine identifier (1–100) | Grouping key — **never split randomly across it** |
| `cycle` | Operational cycle number within that engine's life | Time axis |
| `op1` | Operational setting 1 — altitude | FD001: constant (single operating condition) |
| `op2` | Operational setting 2 — Mach number | FD001: constant |
| `op3` | Operational setting 3 — throttle resolver angle | FD001: constant |
| `s1` | T2 — total temperature at fan inlet | Constant in FD001 → excluded |
| `s2` | T24 — total temperature at LPC outlet | **Informative** |
| `s3` | T30 — total temperature at HPC outlet | **Informative** |
| `s4` | T50 — total temperature at LPT outlet | **Informative** |
| `s5` | P2 — pressure at fan inlet | Constant → excluded |
| `s6` | P15 — total pressure in bypass duct | Constant → excluded |
| `s7` | P30 — total pressure at HPC outlet | **Informative** |
| `s8` | Nf — physical fan speed | **Informative** |
| `s9` | Nc — physical core speed | **Informative** |
| `s10` | epr — engine pressure ratio (P50/P2) | Constant → excluded |
| `s11` | Ps30 — static pressure at HPC outlet | **Informative** |
| `s12` | phi — fuel flow / Ps30 ratio | **Informative** |
| `s13` | NRf — corrected fan speed | **Informative** |
| `s14` | NRc — corrected core speed | **Informative** |
| `s15` | BPR — bypass ratio | **Informative** |
| `s16` | farB — burner fuel-air ratio | Constant → excluded |
| `s17` | htBleed — bleed enthalpy | **Informative** |
| `s18` | Nf_dmd — demanded fan speed | Constant → excluded |
| `s19` | PCNfR_dmd — demanded corrected fan speed | Constant → excluded |
| `s20` | W31 — HPT coolant bleed flow | **Informative** |
| `s21` | W32 — LPT coolant bleed flow | **Informative** |

## What the sensors mean physically

The engine degrades via **high-pressure compressor (HPC) efficiency loss**
(the single fault mode in FD001). As the HPC degrades:

- **Temperatures (s2, s3, s4)** rise — the compressor does less useful work,
  so more fuel energy leaves as heat. s3 (HPC outlet) and s4 (LPT outlet)
  are the most direct witnesses.
- **Pressures (s7, s11)** and **speeds (s8, s9, s13, s14)** shift as the
  control system compensates to hold thrust.
- **Fuel-related (s12, s15, s17, s20, s21)** drift as more fuel/bleed is
  needed for the same output.

This is why the model's top features (s4/s3 temperature statistics, s8/s9
speed statistics) are *physically plausible*, not just correlated —
see `docs/methodology.md` ("Does the model rediscover the physics?").

## Target

**Remaining Useful Life (RUL)** in cycles, modeled as *piecewise-linear*:
`RUL = min(125, cycles_remaining)`. The 125-cycle cap is standard C-MAPSS
practice — sensor traces show no degradation trend in early life, so the
model is only asked to resolve the last 125 cycles. The cap is a **modeling
choice**, documented as an assumption, not a measured threshold.

## Known limitations of the data

1. **Simulated**, from a thermodynamic model — sensor noise and fault
   progression are cleaner than field data.
2. **FD001 only**: one operating condition, one fault mode. Results do not
   transfer to FD002–FD004 (six conditions, two fault modes) without work.
3. All engines start healthy and degrade monotonically to failure — no
   maintenance events, no mixed/compound faults, no sensor dropouts.
4. Test engines are truncated at random points; each contributes exactly
   one scored prediction.
