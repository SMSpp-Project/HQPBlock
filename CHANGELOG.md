# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-16

### Added

- the hard rows of `HQPBlock`, `A_h x = b_h` and `C_h x <= d_h`, outside
  the levels, with no slack and no weight, which every level keeps
  (`HardRows` in `load()`, the groups of rows `hard_eq` and `hard_ineq`,
  `set_hard_A()`, `set_hard_b()`, `set_hard_C()`, `set_hard_d()` by
  Modification, and the variables `HardEq*` and `HardIneq*` of the netCDF
  format); `HQPDistBlock` and `HQPStageSolver` refuse an `HQPBlock` with
  hard rows
- quadratic hard rows of `HQPBlock`, `1/2 x' Q_i x + c_i' x <= e_i` with
  `Q_i` positive semidefinite, which make each level a convex QCQP
  (`Q`, `Cq` and `eq` of `HardRows`, the group of rows `hard_quad` with a
  `QuadFunction` each, `set_hard_eq()` by Modification, and the variables
  `HardQuad*` of the netCDF format); `tools/trots2nc4.py` converts the
  convex quadratic cost-functions of TROTS (Type 2) into them, and
  `write_hqp()` of `tools/npz2nc4.py` writes them
- sum-exp and p-norm hard rows of `HQPBlock`, `sum_k c_ik exp( a_ik' x +
  b_ik ) + l_i' x <= u_i` and `s_i || A_i x + b_i ||_p_i + l_i' x <= u_i`
  (`SE`, `SEb`, `SEc`, `SEl`, `se`, `PN`, `PNb`, `PNp`, `PNs`, `PNl` and
  `pn` of `HardRows`, `set_hard_se()` and `set_hard_pn()`, and the
  variables `HardSumExp*` and `HardPNorm*` of the netCDF format), imposed
  by cuts only; `tools/trots2nc4.py` converts the LTCP (Type 4) and the
  convex gEUD (Type 3, a >= 1, minimised or bounded above) cost-functions
  of TROTS into them, and `write_hqp()` of `tools/npz2nc4.py` writes them
- the linear cuts of an outer approximation of the nonlinear hard rows of
  `HQPBlock`, which `generate_dynamic_constraints()` separates into the
  dynamic group `hard_cuts`: the linearization at the current `x`, or at
  the point where the segment from the last point that strictly satisfied
  the row crosses its boundary, optionally in the logarithmic form for the
  sum-exp rows, with the tolerance and the method from the
  `dynamic_constraints` of the `BlockConfig`; bit 1 of its
  `static_constraints` makes the quadratic hard rows cuts too, for the
  inner Solvers that take no quadratic rows, and `set_hard_eq()`,
  `set_hard_se()` and `set_hard_pn()` shift the cuts of the rows they
  change; a `:MILPSolver` with `intRelaxIntVars` 2 solves and separates
  in a loop
- `is_feasible()` of `HQPBlock`, on the hard rows of the data, the
  nonlinear ones computed by the Block whatever their abstract
  representation, or on the abstract representation, and
  `hard_row_value()`, the value and the gradient of a nonlinear hard row
- `intOneInner` of `HQPSolver`: a single inner Solver for all the levels,
  each one starting from the solution of the previous level, which is
  feasible for it, rather than one for each level starting from its
  solution of the same level in the previous `compute()`
- `tools/trots2nc4.py`, which converts the lexicographic problem of a case
  of TROTS, the Radiotherapy Optimisation Test Set, with linear
  cost-functions (all the proton cases) into the netCDF format of
  `HQPBlock`, the constraints and `x >= 0` as hard rows (as a level 0 with
  `--soft`), and its solution as the group `HQPPoint`; with `--weighted`,
  the weighted sum of the objectives of the case (structure `problem`), of
  which that solution is the optimum, as the Linear Program of an
  `AbstractBlock`
- `write_hqp()` of `tools/npz2nc4.py`, which writes an `HQPBlock` out of
  its matrices, dense or CSR, with its hard rows
- `set_hard_Cq()`, `set_hard_SEl()` and `set_hard_PNl()` of `HQPBlock`,
  which change the linear terms of the nonlinear hard rows, those of the
  `QuadFunction` of `hard_quad` and the coefficients of the cuts by
  Modification (the cuts of the logarithmic form of a sum-exp row are
  removed), and `set_hard_SEb()` and `set_hard_PNb()`, which change their
  vectors `b`, shifting the right-hand side of the cuts of a p-norm row
  and removing those of a sum-exp row
- `tools/speedcone.py`, which adds to an `HQPBlock` of the radial switching
  scenario a limit on the speed of each robot at each stage, as quadratic
  hard rows on new columns tied to the inputs by hard equality rows

### Fixed

- the slacks fixed by `HQPBlock::next_level()` at `w + tol`, which
  `set_slack_values()` overwrote with `w`, so that
  `HQPSolver::get_var_solution()` threw with `dblSlackTol > 0`: a fixed
  slack is now kept, also in `HQPDistBlock`
- `set_hard_eq()`, `set_hard_se()` and `set_hard_pn()` of `HQPBlock` given
  the vector of the `HardRows` of the `HQPBlock` itself

- the installed module `smspp.hqpblock` could not be imported, since its
  package gave the interface of the module the target of the library by its
  name in the build tree, which an installed package does not have

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

[Unreleased]: https://gitlab.com/smspp/hqpblock/-/compare/0.2.0...develop
[0.2.0]: https://gitlab.com/smspp/hqpblock/-/compare/0.1.0...0.2.0
[0.1.0]: https://gitlab.com/smspp/hqpblock/-/tags/0.1.0
