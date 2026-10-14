# Plotting implementation provenance

The user supplied these branches before this edition; they were inspected and
adapted while keeping the shared validated numerical model:

| Report figures | Earlier code used | Integration |
|---|---|---|
| 1–2 | feature-q1-overview-plots, `src/plotting1.py` | Four-panel drawing block adapted to current hourly data; same raw dual family ordering |
| 3–4 | q2b, `main_q2_linear.py` | Midpoint load profiles, original line/marker/grid conventions; Figure 4 uses report hours 7/9/11/18 |
| 5–6,26 | feature-q2c, `run_q2c.py` | Base and four-panel sweep plotting blocks adapted directly; Figure 26 follows benchmark comparison |
| 7–8 | Report JPG compositions of Q2(b/c) plots | Rebuilt as two-panel Matplotlib figures from current model results, using original component sampling |
| 9–11 | 2e, `src/q2e_nonseparable.py`, `run_q2e.py` | Base/comparison/timing separation; price input is read from the newly saved experiment design |
| 12–13 | feature-q3_new, `src/3e/runner.py`, plus final report panels | Restore signed deviations and grouped flow panels; final caption interval convention [t,t+1) |
| 14–17 | feature-q3_new, `src/3f/runner.py` | Same six scenarios, four/six-panel layouts, midpoint profiles, labels and units |
| 18–20 | feature-q3_new, `src/3g/runner.py`, plus final report panels | Actual charge/discharge/SOC, 2x2 price/flow overlays, two deviation panels |
| 21–25 | Final report definitions | New implementation of the mean-preserving capacity/power scans, matched line semantics, and computed plateau marks |

Only drawing conventions were adapted from previous branches. Obsolete pi=2 and
multiplicative price scaling were not restored, because the final report requires
calibrated pi and mean-preserving variability. All original input JSON is unchanged.

No output picture is created by copying a report picture. The package contains
no PNG/JPG in `report_reference/`; the renderer consumes numerical CSV/JSON only.
Original report images were used externally during visual review, never at runtime.
