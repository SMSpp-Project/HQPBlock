#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#    ho_mpc_radial.py: the hierarchical QPs of a centralized Model            #
#    Predictive Control of N robots, as .npz for npz2nc4.py                   #
#                                                                             #
#    Runs the radial switching scenario of the HO-MPC package (each of the    #
#    R omnidirectional robots starts on a circle and has to reach the         #
#    antipodal point, so that all cross in the middle; control horizon N,     #
#    3 if not given, with no prediction horizon; levels: 0 the dynamics, 1    #
#    the limits of the inputs, 2 their smoothness, 3 the avoidance of the     #
#    collisions, at the first stage only or, with "all", at every stage, 4    #
#    the goals) and, at the given MPC steps, saves the tasks A_<p>, b_<p>,    #
#    C_<p>, d_<p> that the package hands to its hierarchical QP solver, with  #
#    the solution after each level p (xref_<p>), obtained by the same solver  #
#    on the hierarchy truncated at p; the one of the last level is the one    #
#    the MPC applies.                                                         #
#                                                                             #
#    The columns are the deviations of the inputs u_{j,k} and of the next     #
#    states s_{j,k+1} of each robot j from the point of the linearization,    #
#    the inputs first, each block by robot and then by stage, and level 0     #
#    holds the dynamics s_{j,k+1} = A_{j,k} s_{j,k} + B_{j,k} u_{j,k}, with   #
#    s_{j,0} = 0; the stage structure of HQPBlock is saved as well:           #
#    n_stages = N, var_stage and var_state, the stage of each column and 1    #
#    for a state, and dynamics_level = 0.                                     #
#                                                                             #
#    The package is not part of SMS++: the directory given as <ho-mpc> must   #
#    hold its packages hierarchical_optimization_mpc (in                      #
#    src/hierarchical_optimization_mpc of the repository                      #
#    github.com/ddebenedittis/hierarchical_optimization_mpc) and              #
#    hierarchical_qp (in hqp/, from the branch ros2 of                        #
#    github.com/ddebenedittis/hierarchical_qp); they need numpy, scipy,       #
#    casadi, qpsolvers and quadprog, not ROS.                                 #
#                                                                             #
#    The scenario and the HO-MPC package are those of                         #
#                                                                             #
#      D. De Benedittis, M. Garabini, L. Pallottino, "Managing Conflicting    #
#      Tasks in Heterogeneous Multi-Robot Systems Through Hierarchical        #
#      Optimization", IEEE Robotics and Automation Letters 10(6):5305-5312,   #
#      2025, doi 10.1109/LRA.2025.3559843                                     #
#                                                                             #
#    and of the repository above, of D. De Benedittis and F. Iadarola, whose  #
#    code is released by its authors under the MIT license; the instances it  #
#    generates are distributed with HQPBlock with their consent.              #
#                                                                             #
#    Usage: ho_mpc_radial.py <ho-mpc> <out dir> <step,step,...>               #
#                            [ <R> [ <N> [ all ] ] ]                          #
#                                                                             #
#    The files are <out dir>/radial<R>_s<step>.npz for N = 3 without "all",   #
#    <out dir>/radial<R>_N<N>[c]_s<step>.npz otherwise (N on three digits,    #
#    c for "all"), e.g., for the data of the tests (regularization 1e-6,      #
#    the one of the package):                                                 #
#                                                                             #
#        ho_mpc_radial.py ho-mpc npz 0,1,2,85,86,87,170,171                   #
#        ho_mpc_radial.py ho-mpc npz 0,1,2,40,41,85,86 8 10                   #
#        ho_mpc_radial.py ho-mpc npz 0,1,2 8 20                               #
#        ho_mpc_radial.py ho-mpc npz 0,1,2 8 20 all                           #
#        ho_mpc_radial.py ho-mpc npz 0 8 50                                   #
#        for f in npz/*.npz; do                                               #
#         npz2nc4.py --eps 1e-6 $f ../data/nc4/$(basename $f .npz).nc4        #
#        done                                                                 #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #

import copy
import os
import sys
import types

import numpy as np
import casadi as ca

root, out = sys.argv[1], sys.argv[2]
steps = sorted(int(k) for k in sys.argv[3].split(','))
n_robot = int(sys.argv[4]) if len(sys.argv) > 4 else 8
horizon = int(sys.argv[5]) if len(sys.argv) > 5 else 3
coll_all = len(sys.argv) > 6 and sys.argv[6] == 'all'

# the package asks ROS for a directory only when it writes its own output
ament = types.ModuleType('ament_index_python')
ament.packages = types.ModuleType('ament_index_python.packages')
ament.packages.get_package_share_directory = lambda name: out
sys.modules['ament_index_python'] = ament
sys.modules['ament_index_python.packages'] = ament.packages
sys.path[:0] = [os.path.join(root, 'src/hierarchical_optimization_mpc'),
                os.path.join(root, 'hqp')]

from hierarchical_optimization_mpc.ho_mpc_multi_robot import (  # noqa: E402
    HOMPCMultiRobot, TaskBiCoeff, TaskType)
from hierarchical_optimization_mpc.utils.robot_models import (  # noqa: E402
    RobCont, get_omnidirectional_model)
from hierarchical_qp.hierarchical_qp import HierarchicalQP  # noqa: E402

os.makedirs(out, exist_ok=True)

# the tasks and the solution of each call of the hierarchical QP solver
captured = {}
solve = HierarchicalQP.__call__


def recording_solve(self, A, b, C, d, *args, **kwargs):
    captured['args'] = [copy.deepcopy(v) for v in (A, b, C, d)]
    captured['x'] = solve(self, A, b, C, d, *args, **kwargs)
    return captured['x']


HierarchicalQP.__call__ = recording_solve


def evolve(s, u_star, dt):
    for j, _ in enumerate(s.omni):
        for _ in range(10):
            s.omni[j] = s.omni[j] + dt / 10 * np.array(u_star.omni[j][:2])
    return s


def dense(M, n):
    return M if M is not None and M.size else np.empty((0, n))


def flat(v, m):
    return (np.asarray(v).flatten() if v is not None else np.empty(0))[:m]


# ----- the scenario -------------------------------------------------------- #
np.random.seed(1)
dt = 0.05
v_max = 2
n_robots = RobCont(omni=n_robot)
s = RobCont(omni=None)
u = RobCont(omni=None)
s_kp1 = RobCont(omni=None)
s.omni, u.omni, s_kp1.omni = get_omnidirectional_model(5 * dt)

input_limits = RobCont(omni=ca.vertcat(u.omni[0] - v_max, -u.omni[0] - 2,
                                       u.omni[1] - 2, -u.omni[1] - 2))
input_smooth = RobCont(omni=ca.vertcat(u.omni[0], -u.omni[0],
                                       u.omni[1], -u.omni[1])).tolist()
input_smooth_coeffs = [[[np.array([0.95, 0.95, 0.9, 0.9])]
                        for j in range(n_robot)], [[]]]

radius = 5 / (2 * np.sin(np.pi / n_robot))
goals, s_init = [], []
for i in range(n_robot):
    th = 2 * np.pi * i / n_robot
    goals.append(np.array([radius * np.cos(th), radius * np.sin(th)]))
    s_init.append(np.array([radius * np.cos(th - np.pi),
                            radius * np.sin(th - np.pi)]))

aux = ca.SX.sym('aux', 2, 2)
mapping = RobCont(omni=ca.vertcat(s.omni[0], s.omni[1]))
avoid = ca.vertcat(-((aux[0, 0] - aux[1, 0]) ** 2)
                   - (aux[0, 1] - aux[1, 1]) ** 2)
avoid_coeff = [TaskBiCoeff(0, i, 0, j, k, -(0.6 ** 2))
               for k in (range(horizon) if coll_all else [0])
               for i in range(n_robot) for j in range(i + 1, n_robot)]

hompc = HOMPCMultiRobot(s.tolist(), u.tolist(), s_kp1.tolist(),
                        n_robots.tolist())
# a prediction horizon would repeat the last input, coupling the stages
hompc.n_control = horizon
hompc.n_pred = 0
hompc.create_task(name='input_limits', prio=1, type=TaskType.Same,
                  ineq_task_ls=input_limits.tolist())
hompc.create_task(name='input_smooth', prio=2, type=TaskType.SameTimeDiff,
                  ineq_task_ls=input_smooth,
                  ineq_task_coeff=input_smooth_coeffs)
hompc.create_task_bi(name='collision_avoidance', prio=3, type=TaskType.Bi,
                     aux=aux, mapping=mapping.tolist(), ineq_task_ls=avoid,
                     ineq_task_coeff=avoid_coeff)
for nn in range(n_robot):
    hompc.create_task(
        name=f'pos_ref_{nn}', prio=4, type=TaskType.Same,
        eq_task_ls=RobCont(omni=ca.vertcat(s_kp1.omni[0],
                                           s_kp1.omni[1])).tolist(),
        eq_task_coeff=RobCont(omni=[[goals[nn]]
                                    for _ in range(n_robot)]).tolist(),
        robot_index=[[nn]])

# ----- the stage structure ------------------------------------------------- #
n_s, n_u = 2, 2   # the states and the inputs of an omnidirectional robot


def stages(n):
    """the stage of each of the n columns and 1 for a state, 0 for an input:
    u_{j,k} in column j N n_u + k n_u + i, s_{j,k+1} in column R N n_u +
    j N n_s + k n_s + i"""
    if n != n_robot * horizon * (n_s + n_u):
        sys.exit(f'{n} columns, not those of {n_robot} robots over '
                 f'{horizon} stages')
    stage, state = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for j in range(n_robot):
        for k in range(horizon):
            for i in range(n_u):
                stage[j * horizon * n_u + k * n_u + i] = k
            for i in range(n_s):
                c = n_robot * horizon * n_u + j * horizon * n_s + k * n_s + i
                stage[c], state[c] = k + 1, 1
    return stage, state


if horizon == 3 and not coll_all:
    name = f'radial{n_robot}'
else:
    name = f'radial{n_robot}_N{horizon:03d}{"c" if coll_all else ""}'

# ----- the run ------------------------------------------------------------- #
s = RobCont(omni=s_init)
for k in range(steps[-1] + 1):
    if np.all(np.abs(np.array(s.omni)[:, :2] - np.array(goals)) < 1e-2):
        sys.exit(f'the goals are reached at step {k}, before {steps[-1]}')
    u_star = hompc(copy.deepcopy(s.tolist()))
    if k in steps:
        args = [copy.deepcopy(v) for v in captured['args']]
        P = len(args[0])
        n = args[0][0].shape[1]
        var_stage, var_state = stages(n)
        rec = {'n_levels': P, 'step': k, 'n_stages': horizon,
               'var_stage': var_stage, 'var_state': var_state,
               'dynamics_level': 0}
        for p in range(P):
            rec[f'A_{p}'] = dense(args[0][p], n)
            rec[f'b_{p}'] = flat(args[1][p], rec[f'A_{p}'].shape[0])
            rec[f'C_{p}'] = dense(args[2][p], n)
            rec[f'd_{p}'] = flat(args[3][p], rec[f'C_{p}'].shape[0])
            rec[f'xref_{p}'] = np.asarray(solve(
                HierarchicalQP(solver=hompc.solver),
                *[copy.deepcopy(v[:p + 1]) for v in args])).flatten()
        if np.abs(rec[f'xref_{P - 1}'] -
                  np.asarray(captured['x']).flatten()).max() != 0:
            sys.exit(f'step {k}: the last level is not what the MPC applied')
        np.savez(os.path.join(out, f'{name}_s{k:03d}.npz'), **rec)
    s = evolve(s, RobCont(omni=u_star[0]), dt)
