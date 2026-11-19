<#
Reproduces every Q3 number and figure of the report in the right order (about 9-10 h on a 16-thread laptop).
Run from the repository root (the folder that contains Q1_models, Q2_final and Q3_final):

    powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1 -DryRun     # check paths and print the plan
    powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1             # run everything
    powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1 -From 12    # resume at step 12
    powershell -ExecutionPolicy Bypass -File Q3_final\run_all.ps1 -From 22    # only the N study (same start and limits
                                                                              # for every N), then summary and figures

Every step writes its own log to Q3_final\logs\run_all\. The script stops at the first failing step and tells you
which -From value resumes it. Warm-start files that do not exist yet are skipped by the solver (multi-start), so the
script also works on a fresh checkout; existing price vectors from earlier runs are only used as start candidates.
#>
param(
    [switch]$DryRun,
    [int]$From = 1,
    [string]$Python = "$env:USERPROFILE\miniforge3-dtu\envs\gurobi_env\python.exe"
)

$ErrorActionPreference = "Continue"
$envDir = Split-Path $Python -Parent
$env:PATH = "$envDir;$envDir\Library\bin;$env:PATH"      # numpy/matplotlib DLLs of the conda env (Windows)

$R    = "Q3_final/results/sdp"
$A5   = "$R/SDP_CVaR_a0.9_w0.5_in5_bb-expectation/prices.json"   # N = 5 price vector A (validation-selected)
$E5   = "$R/SDP_E_in5_bb-expectation/prices.json"
$C5   = "$R/SDP_CC_eps0_in5_bb-expectation/prices.json"
$DPBB = "$R/SDP_E_expected_bb-expectation/prices.json"
$CC10 = "$R/SDP_CC_eps0_in10_bb-expectation/prices.json"
$E10  = "$R/SDP_E_in10_bb-expectation/prices.json"
$CV10 = "$R/SDP_CVaR_a0.9_w0.5_in10_bb-expectation/prices.json"
$E5P3 = "$R/SDP_E_pen3_in5_bb-expectation/prices.json"
$CAP5 = "$R/SDP_E_in5_bb-none/prices.json"
$N5A  = "Q3_final/data/n5_vectors/A.json"                     # archived N = 5 solutions (see README)
$N5D  = "Q3_final/data/n5_vectors/D.json"
$q3   = "Q3_final\run_q3.py"
# "pool:<set>" = every price vector already found for that scenario set (results/sdp and, for in5, data/n5_vectors),
# evaluated at fixed prices in phase 1; the best complete solution starts the MILP (pooled multi-start). Running the
# script twice therefore never makes a model worse, and a variant can never end above a solution feasible for it.

# id, name (log file), arguments; experiments as in the README table (E0-E7)
$steps = @(
    @(1,  "tests",          @("-m", "pytest", "Q3_final\tests", "-q")),
    @(2,  "scenarios",      @($q3, "scenarios")),
    @(3,  "validate",       @($q3, "validate")),
    @(4,  "sdp_validate",   @($q3, "sdp-validate")),
    @(5,  "baseline_Q1Q2",  @($q3, "baseline")),
    @(6,  "DP_cap_only",    @($q3, "sdp", "--model", "E", "--set", "expected", "--bb", "none")),
    @(7,  "DP_BB",          @($q3, "sdp", "--model", "E", "--set", "expected", "--time-limit", "600")),
    @(8,  "E_in5_cap_only", @($q3, "sdp", "--model", "E", "--set", "in5", "--bb", "none", "--warm", "pool:in5")),
    @(9,  "E_in10_cap_only",@($q3, "sdp", "--model", "E", "--set", "in10", "--bb", "none", "--warm", "pool:in10,$CAP5")),
    @(10, "E_in5",          @($q3, "sdp", "--model", "E", "--set", "in5", "--time-limit", "1800", "--warm", "pool:in5")),
    @(11, "CVaR_in5",       @($q3, "sdp", "--model", "CVaR", "--set", "in5", "--time-limit", "1800", "--warm", "pool:in5")),
    @(12, "CVaR_weight",    @($q3, "sweep", "--model", "CVaR", "--param", "weight", "--values", "0.01,0.25,0.75,1", "--time-limit", "600", "--warm", "pool:in5")),
    @(13, "CVaR_alpha",     @($q3, "sweep", "--model", "CVaR", "--param", "alpha", "--values", "0.4,0.6,0.8", "--weight", "0.5", "--time-limit", "600", "--warm", "pool:in5")),
    @(14, "CC_in5",         @($q3, "sweep", "--model", "CC", "--param", "epsilon", "--values", "0,0.2,0.4,0.6", "--time-limit", "600", "--warm", "pool:in5")),
    @(15, "CC_in10",        @($q3, "sweep", "--model", "CC", "--param", "epsilon", "--values", "0,0.3,0.6", "--set", "in10", "--time-limit", "1800", "--warm", "pool:in10,$N5D,$N5A")),
    @(16, "E_in10",         @($q3, "sdp", "--model", "E", "--set", "in10", "--time-limit", "900", "--warm", "pool:in10,$N5D")),
    @(17, "CVaR_in10",      @($q3, "sdp", "--model", "CVaR", "--set", "in10", "--time-limit", "900", "--warm", "pool:in10,$N5D")),
    @(18, "select_in5",     @($q3, "select", "--oos")),
    @(19, "E_pen3",         @($q3, "sdp", "--model", "E", "--penalty", "3", "--time-limit", "900", "--rules", "optimistic", "--warm", "pool:in5")),
    @(20, "CVaR_pen3",      @($q3, "sdp", "--model", "CVaR", "--penalty", "3", "--time-limit", "900", "--rules", "optimistic", "--warm", "pool:in5")),
    @(21, "CC_pen3",        @($q3, "sweep", "--model", "CC", "--param", "epsilon", "--values", "0,0.2,0.4,0.6", "--penalty", "3", "--time-limit", "900", "--warm", "pool:in5")),
    @(22, "N_study",        @($q3, "nstudy", "--values", "1,2,5,10,20", "--time-limit", "1800")),
    @(23, "summary",        @($q3, "summary")),
    @(24, "figures",        @($q3, "figures"))
)

# ---- checks -----------------------------------------------------------------------------------------------------------
$ok = $true
if (-not (Test-Path $Python)) { Write-Host "Python not found: $Python  (use -Python <path to python.exe>)" -ForegroundColor Red; $ok = $false }
foreach ($d in @("Q1_models", "Q2_final", "Q3_final")) {
    if (-not (Test-Path $d)) { Write-Host "Folder $d not found: run the script from the repository root" -ForegroundColor Red; $ok = $false }
}
if ($ok -and -not $DryRun) {
    & $Python -c "import gurobipy, numpy, scipy, pandas, matplotlib; print('packages ok, gurobipy', gurobipy.gurobi.version())"
    if ($LASTEXITCODE -ne 0) { Write-Host "Package check failed" -ForegroundColor Red; $ok = $false }
}
if (-not $ok) { exit 1 }

$logDir = "Q3_final\logs\run_all"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# ---- plan / run -------------------------------------------------------------------------------------------------------
$t0 = Get-Date
foreach ($s in $steps) {
    $id, $name, $argList = $s
    if ($id -lt $From) { continue }
    $log = Join-Path $logDir ("{0:D2}_{1}.txt" -f $id, $name)
    $line = "[{0:D2}] {1}" -f $id, $name
    if ($DryRun) {
        Write-Host $line -ForegroundColor Cyan
        Write-Host ("     python " + ($argList -join " "))
        $w = [array]::IndexOf($argList, "--warm")
        if ($w -ge 0) {
            foreach ($f in $argList[$w + 1].Split(",")) {
                $state = if ($f -like "pool:*") { "pool of all runs so far" } elseif (Test-Path $f) { "found" } else { "missing (skipped, fine on a fresh run)" }
                Write-Host ("       warm: {0}  [{1}]" -f $f, $state) -ForegroundColor DarkGray
            }
        }
        continue
    }
    $start = Get-Date
    Write-Host ("{0}  started {1:HH:mm}  -> {2}" -f $line, $start, $log) -ForegroundColor Cyan
    & $Python @argList *> $log
    $code = $LASTEXITCODE
    $min = ((Get-Date) - $start).TotalMinutes
    if ($code -ne 0) {
        Write-Host ("{0}  FAILED (exit code {1}) after {2:N1} min. See {3}. Resume with: -From {4}" -f $line, $code, $min, $log, $id) -ForegroundColor Red
        exit $code
    }
    Write-Host ("{0}  done in {1:N1} min" -f $line, $min) -ForegroundColor Green
}
if (-not $DryRun) {
    Write-Host ("All steps done in {0:N1} h. Summary: Q3_final\results\q3_summary.csv, figures: Q3_final\figures" -f ((Get-Date) - $t0).TotalHours) -ForegroundColor Green
}
