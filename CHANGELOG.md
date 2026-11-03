# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-03

### Added

- initial module skeleton generated from ModuleTemplate
- `HQPBlock`, the hierarchical (lexicographic) QP as one model in the
  original variables, with the methods that move it from a level to the next
  one and change its data by Modification, a weight for each variable in the
  regularization, whose constant is in the `Objective`, a dynamic group of
  rows `link` that an ancestor fills, and its netCDF format
- `HQPLevelBlock`, the interface of the levels of a hierarchical QP that
  `HQPSolver` uses, which `HQPBlock` and `HQPDistBlock` implement, with the
  scale of the whole `Objective` (`set_objective_scale()`), which changes no
  minimizer
- `HQPDistBlock`, the hierarchical QP split among owners of its columns (a
  sub-Block `HQPBlock` for each owner, with copies of the variables of the
  others in its rows, and the consensus rows in the root, with the
  structure `kConsensus`; the coupling rows in the root, with a copy in the
  root of each variable of an equality coupling row and the parts of the
  other variables in the sub-Blocks bounded by its variables, with
  `kBenders`, chosen by `set_structure()`), exactly the centralized
  hierarchy, so that it can be solved by decomposition, and its netCDF
  format; with `kBenders` the `Objective` of the root has all its
  Variable, and the root starts each level at the solution of the previous
  one
- `HQPSolver`, which solves the levels of an `HQPBlock` one after the other,
  each with an inner Solver of its own taken from the same
  `BlockSolverConfig`, so that an inner Solver that re-optimizes (e.g.,
  DAQP or qpOASES) starts each level from its solution of the same level in
  the previous `compute()`, e.g., the previous step of a Model Predictive
  Control; `get_elapsed_iterations()` sums their iterations, the inner
  Solvers get its log and, after an NBModification of the Block, are
  created anew; with `intLevelScale` the `Objective` of each level is
  multiplied by the inverse of its value, so that the inner Solver sees a
  level of value about 1
- `tools/npz2nc4.py`, from `.npz` to the netCDF format of `HQPBlock`
- `tools/ho_mpc_radial.py`, which writes as `.npz` the hierarchical QPs of
  the centralized Model Predictive Control of the radial switching scenario
  of the HO-MPC package, i.e., the instances of `data/nc4`
- the unit test of `HQPBlock` and `HQPDistBlock`, which needs the core
  library only: the methods of the Blocks and the Modification they issue,
  the netCDF format and its degenerate cases, checked without solving
- the stage structure of an optimal control problem in `HQPBlock`
  (`HQPBlock::StageData`, an optional argument of `load()`): the number of
  stages, the stage of each column and whether it is a state or an input,
  optionally the stage of each row, and the level of the dynamics, with
  their getters, in `print()` and in the netCDF format (`NumStages`,
  `VarStage`, `VarState`, `EqStage`, `IneqStage`, `DynamicsLevel`); a file
  without them is read as before; `get_objective_scale()`
- `HQPStageSolver`, a Solver of one level of an `HQPBlock` with the stage
  structure, e.g., the inner Solver of `HQPSolver`: a primal active-set
  method started at the current point of the `HQPBlock` (the solution of
  the previous level), whose subproblems are solved by a Riccati recursion
  with stage equalities and constraints-to-go, rank-revealing so that
  dependent rows are allowed, with the multipliers of the rows from a
  forward sweep; it keeps its working set from a `compute()` to the next
  one (`intWarmStart`), computes the dynamics and the rows on the stages
  anew only when the coefficients of some row change, and has the
  tolerances `dblRankTol`, `dblFeasTol` and `dblDualTol`
- `tools/ho_mpc_radial.py` takes the control horizon and the avoidance of
  the collisions at every stage, and writes the stage structure, which
  `tools/npz2nc4.py` converts; `data/nc4` (version 2026-10-31) has also
  steps of the control horizons 10, 20 and 50 with the stage structure
- the unit test checks the stage structure of `HQPBlock`, the recursion of
  `HQPStageSolver` against the dense KKT system on random problems with
  dependent rows, constraints-to-go, a free or empty initial state and
  stages with no input, the cascade of `HQPStageSolver` on hierarchies
  whose values are known, with dependent rows and on steps of Model
  Predictive Control, what it refuses, and, as the test
  `HQPBlock_unit_test_radial`, the instances of `data/nc4` with the stage
  structure against their solution

### Changed

- the README points to the installation guide of the wiki
  (Installing-SMS++) for the channels that distribute SMS++

### Fixed

- on Windows, the executables that copy their DLLs next to themselves
  are linked one at a time (a job pool of Ninja), since two copies of the
  same DLL into one directory collided (e.g., with the clang++ of MSYS2)

- with the compilers of MinGW (MSYS2) the target asks whoever links it
  for the symbol that forces the module in, as with MSVC and on macOS:
  the linker imports a DLL only for the symbols asked for, so that a
  program that used none had the module left out and its classes
  missing from the factory
- `HQPSolver::compute()` is left unlocked when an inner Solver throws

[Unreleased]: https://gitlab.com/smspp/hqpblock/-/compare/0.1.0...develop
[0.1.0]: https://gitlab.com/smspp/hqpblock/-/tags/0.1.0
