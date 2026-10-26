# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- initial module skeleton generated from ModuleTemplate
- `HQPBlock`, the hierarchical (lexicographic) QP as one model in the
  original variables, with the methods that move it from a level to the next
  one and change its data by Modification, and its netCDF format
- `HQPSolver`, which solves the levels of an `HQPBlock` one after the other
  with an inner Solver taken from a `BlockSolverConfig`
- `tools/npz2nc4.py`, from `.npz` to the netCDF format of `HQPBlock`
- the unit test of `HQPBlock`, which needs the core library only: the
  methods of the Block and the Modification they issue, the netCDF format
  and its degenerate cases, checked without solving
