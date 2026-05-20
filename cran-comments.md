# cran-comments.md

## Submission

First CRAN submission of `gbrs` 0.1.0.

GBRS (Gradient Boosted Risk Scores) fits interpretable points-based risk
scores via gradient boosting for regression, binary classification, and
survival outcomes. The package wraps a C++17 core (also exposed via Python
bindings outside of CRAN scope) through Rcpp + RcppEigen.

## R CMD check results

Local checks (`R CMD check --as-cran` with R 4.5):

* Windows 11, R 4.5.2: **0 errors | 0 warnings | 1 note**
* Debian trixie (via rocker/r-base:4.5.0), R 4.5.0: **0 errors | 0 warnings | 2 notes**

The Windows note and the first Debian note are both the standard
`New submission` flag. The second Debian note is "unable to verify
current time" — an artefact of running inside a sandboxed Docker
container with no NTP access. Neither indicates a package issue.

## Configure script

The package ships a `configure` script (~50 lines, POSIX shell) that
probes for OpenMP support and writes `src/Makevars` from
`src/Makevars.in`. On Windows, `configure.win` simply copies
`src/Makevars.win.in` (Rtools always has OpenMP). `cleanup` removes the
generated `Makevars` files. This is the same pattern used by `data.table`
and `xgboost`.

The configure script has been verified to:
* successfully probe `$(SHLIB_OPENMP_CXXFLAGS)` and substitute the
  resulting flag into `src/Makevars` (Linux check above).
* pass `checkbashisms` (no bashisms in configure or cleanup).

## Compilation flags

Per CRAN policy, the package does NOT set `-O3`, `-march=native`,
`-fPIC`, or any direct `-fopenmp`. R's default `CXXFLAGS` apply.
OpenMP is enabled via `$(SHLIB_OPENMP_CXXFLAGS)` only. Users who want
CPU-specific vectorisation can set `CXX17FLAGS = -O3 -march=native` in
`~/.R/Makevars` and reinstall — this is documented in the README.

## Test environments

Tested locally on:

* Windows 11, R 4.5.2 with Rtools45.
* Debian trixie, R 4.5.0, gcc 14.2.0 (in Docker via `rocker/r-base:4.5.0`).

I have not yet exercised macOS or R-devel. Pre-CRAN R-hub run is
planned before submission if any reviewer requests broader coverage.

## Downstream dependencies

None. This is a first submission.
