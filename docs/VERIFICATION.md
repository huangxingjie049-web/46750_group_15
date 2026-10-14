# Fresh solve-and-draw verification — 2026-10-14

## Clean full run

Command: `python main.py --task all`, started with no `results/` directory.
Python 3.12; Gurobi 13.0.1 and dependencies from `requirements-lock.txt`.

- 1,091 model runs were independently solved and passed their physical,
  accounting, KKT, and applicable scalar-oracle validations.
- 426 report numeric checkpoints passed at the report's displayed precision.
- All 26 numbered report images were freshly rendered by Matplotlib and saved
  at the report's PNG/JPG paths, with vector PDF companions.
- Report source PNG/JPG/PDF files were deliberately absent from the package.
- The image-generation stage runs after all required numerical tasks finish,
  so an empty-folder full run has no hidden dependency on prior outputs.

The final visual adjustments were rerendered from the same freshly solved,
report-checked outputs. Dimensions and numerical source hashes were rechecked.

## Tests

`python -m pytest -q`: **45 passed**.

The submission was retested after adding LF/CRLF compatibility
to the original-data hash check.

The seven drawing tests cover default CLI generation without original pictures,
a changed decision changing actual pixels and bar heights, failed numeric checks
stopping the pipeline, all 26 figure definitions, a standalone Q2(e) run, the
explicit no-plots flag, and report response hours versus solved CSV values.
Four tests failed against the predecessor before implementation, then passed.
The full model/data/experiment test suite is retained. Q2(d)'s 49-solve report
sampling replaces its previous 77-solve diagnostic sampling.

Formatting preserved the Python AST, including constants and docstrings.
Final plotting/CLI checks were repeated after formatting and headless-backend
configuration. See the log files in this folder.

## Figure review and traceability

All 26 images were visually paired with the uploaded report images. The original
Q1/Q2(c) drawing blocks and Q2/Q3(f) midpoint-display conventions were restored;
missing panels, Figure 4 hour-11 selection, shared flow limits, price overlays,
base/saturation marks, and interval-boundary displays were checked.

`figure_generation.json` records newly generated image hashes, each numerical
source file and its hash, and each `.plot_data.json` path. The latter captures
actual Matplotlib line/bar arrays and labels, allowing a reviewer to trace the
picture back to computed data rather than relying only on its appearance.

## Limits recorded honestly

Correspondence means report data, panels, selected hours, units, plot meaning,
and image dimensions. It is not a promise of historical image-byte equality:
nonunique dual vectors or equal-cost battery dispatch can differ, and historical
font/rendering details or manually assembled JPG spacing may differ. The actual
newly solved choices are plotted rather than fabricated to match screenshot pixels.
The three inconsistent report sentences remain documented in the numerical manifest.
