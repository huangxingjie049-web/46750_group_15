# Report-aligned integration plan

Purpose: a complete branch-ready package reproducing Q1--Q3 and bonus Q2(e), with main.py as the sole root entry point. Uploaded baseline JSON bytes remain unchanged.

1. Implement one parameterized convex consumer model: utility, linear, quadratic, coupled quadratic; optional daily requirement and battery. Explicit bounds provide dual values.
2. Provide independent scalar procurement/reference solutions plus physical, accounting, KKT and complementarity validation for every model. Test degenerate ties by value, not a fixed schedule.
3. Implement each experiment separately in src/experiments.py: Q1 A/B; Q2 threshold and quadratic sweeps, cross-model comparison, temporal experiment; Q3 daily constraint and coefficient sweeps; calibrated battery base and all report sizing/variability scans.
4. Save hourly, summary, snapshots, validation, figures and tables from the same run. Add manifests and report differences without fabricating agreements.
5. Verify all tasks under Gurobi 13.0.1, run meaningful per-model and integration tests, provide Windows/Linux environment and branch integration instructions.

The report formula defines the Q2(e) sum of adjacent deviations; the old extraction bug omitting this penalty is corrected. Report numerical differences are disclosed, never forced by changing baseline inputs.
