# HQPBlock

[![Pipeline](https://gitlab.com/smspp/hqpblock/badges/develop/pipeline.svg)](https://gitlab.com/smspp/hqpblock/-/pipelines?ref=develop)
[![Release](https://gitlab.com/smspp/hqpblock/-/badges/release.svg)](https://gitlab.com/smspp/hqpblock/-/releases)
[![Documentation](https://img.shields.io/badge/docs-Doxygen-blue)](https://smspp.gitlab.io/smspp-project/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

`HQPBlock` is a SMS++ :Block for the hierarchical (lexicographic)
Quadratic Programs of task-priority control, e.g., in the Model Predictive
Control of robots. Level p (p = 0 being the highest priority) has the tasks
`we_p (A_p x - b_p) = v_p` and `wi_p (C_p x - d_p) <= w_p`, with `w_p >= 0`,
and it minimizes `||v_p||^2 / 2 + ||w_p||^2 / 2` (plus, possibly, the
regularization `eps sum_j r_j (x_j - c_j)^2 / 2`, the weights `r_j` being 1
if not given) keeping the slacks of the higher levels
at their optimal values. Thus, an equality task of a higher level keeps its
optimal residual, and an inequality task is never violated more than at its
optimum.

The classical approach solves each level in the null space of the equality
tasks of the higher ones, and therefore it faces a different QP at each
level. In an `HQPBlock` all the levels are instead one model in the original
variables: the abstract representation has the `ColVariable` `x`, `v` and
`w`, a `FRowConstraint` with a `LinearFunction` for each row of each level,
and a `FRealObjective` with a `DQuadFunction`. Moving to the next level fixes
the slacks of the level just solved and changes some coefficients of the
`Objective`; a new step of the Model Predictive Control changes `b_p`, `d_p`,
`A_p` and `C_p` through the methods of the Block. Since all these are
Modification of the same Block, a Solver that can re-optimize (typically an
active-set one) starts each solve from the previous solution.

`HQPSolver` solves the levels of an `HQPBlock` one after the other, each
with an inner Solver of its own attached to the same Block, so that the
inner Solver of a level starts from its solution of the same level in the
previous step. The inner Solver is any Solver of convex QPs with linear
constraints (in practice a :MILPSolver, such as `DAQPMILPSolver` or
`MOSEKMILPSolver`), taken from the factory by its name in the
`BlockSolverConfig` given by the parameter `strInnerBSC`. With
`intProxCenter` the center of the regularization follows the solution of the
previous level, and each level is then the problem of the null-space
approach with a regularized step. With `intRelaxLower` (the default)
the rows of the levels below the current one are relaxed and their slacks
fixed to 0; this keeps the Hessian positive definite if `eps > 0`, while the
rows left in the model with free slacks make it singular, which the inner
Solvers handle rather less accurately. All that `HQPSolver` uses of the
Block is in the interface `HQPLevelBlock`, which `HQPBlock` implements and
so does `HQPDistBlock`.

`HQPDistBlock` is the same hierarchy with its variables split among owners,
e.g., the robots of a multi-robot Model Predictive Control, so that it can
be solved by decomposition, with one of two structures chosen by
`set_structure()`. With `kConsensus`, the one of a Lagrangian decomposition,
each row gets an owner among those of its columns, and each
owner is a sub-Block, an `HQPBlock` with its own variables, a copy of each
variable of another owner that appears in one of its rows, and its rows;
the root has no variables, an empty `Objective` and the consensus rows that
make each copy equal to the variable it copies. The regularization of a
column is split among its copies, so that on the consensus the problem of
each level is exactly that of the centralized hierarchy, which a Solver of
the whole tree solves and a Lagrangian decomposition relaxes. Moving to the
next level first gives the copies the values of their owners and fixes
again the slacks of the levels already fixed at these values, so that every
level is feasible whatever the accuracy of a decomposition. With `kBenders`,
the one of a Benders decomposition, each owner has only its own variables
and the rows all of whose entries are in them, while each coupling row is
in the root, with its slack and a variable `t` for each owner with entries
in it, which bounds the part of the row in the variables of that owner by
a row of the dynamic group `link` of its sub-Block; again, the problem of
each level is exactly that of the centralized hierarchy.

An `HQPBlock` is serialized into and deserialized out of netCDF, with the
rows in compressed sparse row form (cf. `HQPBlock::deserialize()`). The
script `tools/npz2nc4.py` converts a hierarchy given as a `.npz` file (keys
`A_<p>`, `b_<p>`, `C_<p>`, `d_<p>` and, optionally, `we_<p>`, `wi_<p>`) into
that format. An
`HQPDistBlock` is the owners of the columns (and, optionally, of the rows)
together with the centralized `HQPBlock` in a group of its own (cf.
`HQPDistBlock::deserialize()`). The instances of `data/nc4` are some steps
of the radial switching scenario of the centralized hierarchical Model
Predictive Control of a team of robots of the
[HO-MPC package](https://github.com/ddebenedittis/hierarchical_optimization_mpc)
(D. De Benedittis, M. Garabini, L. Pallottino, "Managing Conflicting Tasks
in Heterogeneous Multi-Robot Systems Through Hierarchical Optimization",
IEEE Robotics and Automation Letters 10(6):5305-5312, 2025), with the
solution of each level of that implementation, written by
`tools/ho_mpc_radial.py` with the package and converted by
`tools/npz2nc4.py`.

The unit test `test/test.cpp` needs the core library only and solves
nothing: it checks the methods of the Block (the sizes of the levels, the
abstract representation, the move from a level to the next one, the changes
of the data and the Modification they issue), the netCDF format and the
errors of the reader, on hierarchies with the degenerate parts of the format
(levels with no equalities or no inequalities, empty levels, rows with no
entries, weights different from 1), and those of `HQPDistBlock` (the owners
of the rows, the copies and the consensus rows, the slacks fixed again,
the changes of the data in and out of the pattern of a row). `HQPSolver`,
whose inner Solver comes
from another module, is tested by the suite
[`HQPBlock`](https://gitlab.com/smspp/tests/-/tree/develop/HQPBlock) of the
system tests of SMS++, which compares several inner Solvers and settings,
on the `HQPBlock` and on the `HQPDistBlock` of the same hierarchies, with an
independent implementation of the null-space approach, on random and
degenerate hierarchies and on the instances of `data/nc4`.


## Getting started

These instructions will let you build the `HQPBlock` module on
your system.

### Requirements

- The [SMS++ core library](https://gitlab.com/smspp/smspp) and its
  requirements.
- To use `HQPSolver`, a module providing its inner Solver, typically
  [MILPSolver](https://gitlab.com/smspp/milpsolver). The instances of
  `data/nc4` are downloaded from the GitLab Package Registry by the system
  tests that read them.
- For the tools, Python with numpy and netCDF4 (`tools/npz2nc4.py`), and
  the HO-MPC package with its requirements (`tools/ho_mpc_radial.py`).

### Build and install with CMake

Configure and build the library with:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```

The library has the same configuration options of
[SMS++](https://gitlab.com/smspp/smspp-project/-/wikis/Customize-the-configuration).

Optionally, install the library in the system with:

```sh
cmake --install .
```

### Usage with CMake

After the library is built, you can use it in your CMake project with:

```cmake
find_package(HQPBlock)
target_link_libraries(<my_target> SMS++::HQPBlock)
```

### Build and install with makefiles

Carefully hand-crafted makefiles have also been developed for those unwilling
to use CMake. Makefiles build the executable in-source (in the same directory
tree where the code is) as opposed to out-of-source (in the copy of the
directory tree constructed in the build/ folder) and therefore it is more
convenient when having to recompile often, such as when developing/debugging
a new module, as opposed to the compile-and-forget usage envisioned by CMake.

Each executable using `HQPBlock` has to include a "main makefile" of
the module, which typically is either [makefile-c](makefile-c) including all
necessary libraries comprised the "core SMS++" one, or
[makefile-s](makefile-s) including all necessary libraries but not the "core
SMS++" one (for the common case in which this is used together with other
modules that already include them). These in turn recursively include all the
required other makefiles, hence one should only need to edit the "main
makefile" for compilation type (C++ compiler and its options) and it all
should be good to go. In case some of the external libraries are not at their
default location, it should only be necessary to create the
`../extlib/makefile-paths` out of the `extlib/makefile-default-paths-*` for
your OS `*` and edit the relevant bits (commenting out all the rest).

Check the [SMS++ installation wiki](https://gitlab.com/smspp/smspp-project/-/wikis/Customize-the-configuration#location-of-required-libraries)
for further details.


## Getting help

If you need support, you want to submit bugs or propose a new feature, you
can [open a new issue](https://gitlab.com/smspp/hqpblock/-/issues/new).


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of
conduct, and the process for submitting merge requests to us.


## Authors

### Current Lead Authors

- **Antonio Frangioni**  
  Dipartimento di Informatica  
  Università di Pisa

- **Donato Meoli**  
  Dipartimento di Informatica  
  Università di Pisa

### Contributors


## License

This code is provided free of charge under the [GNU Lesser General Public
License version 3.0](https://opensource.org/licenses/lgpl-3.0.html) -
see the [LICENSE](LICENSE) file for details.


## Disclaimer

The code is currently provided free of charge under an open-source license.
As such, it is provided "*as is*", without any explicit or implicit warranty
that it will properly behave or it will suit your needs. The Authors of
the code cannot be considered liable, either directly or indirectly, for
any damage or loss that anybody could suffer for having used it. More
details about the non-warranty attached to this code are available in the
license description file.
