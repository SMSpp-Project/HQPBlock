# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- initial module skeleton generated from ModuleTemplate
- `HQPBlock`, the hierarchical (lexicographic) QP as one model in the
  original variables, with the methods that move it from a level to the next
  one and change its data by Modification, a weight for each variable in the
  regularization, and its netCDF format
- `HQPLevelBlock`, the interface of the levels of a hierarchical QP that
  `HQPSolver` uses, which `HQPBlock` and `HQPDistBlock` implement
- `HQPDistBlock`, the hierarchical QP split among owners of its columns (a
  sub-Block `HQPBlock` for each owner, with copies of the variables of the
  others in its rows, and the consensus rows in the root), exactly the
  centralized hierarchy on the consensus, so that it can be solved by
  decomposition, and its netCDF format
- `HQPSolver`, which solves the levels of an `HQPBlock` one after the other,
  each with an inner Solver of its own taken from the same
  `BlockSolverConfig`, so that an inner Solver that re-optimizes (e.g.,
  DAQP or qpOASES) starts each level from its solution of the same level in
  the previous `compute()`, e.g., the previous step of a Model Predictive
  Control; `get_elapsed_iterations()` sums their iterations, the inner
  Solvers get its log and, after an NBModification of the Block, are
  created anew
- `tools/npz2nc4.py`, from `.npz` to the netCDF format of `HQPBlock`
- the unit test of `HQPBlock` and `HQPDistBlock`, which needs the core
  library only: the methods of the Blocks and the Modification they issue,
  the netCDF format and its degenerate cases, checked without solving
