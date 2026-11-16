#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#    speedcone.py: adds to an HQPBlock of the radial switching scenario       #
#    (tools/ho_mpc_radial.py) a limit on the speed of each robot at each      #
#    stage, as convex quadratic hard rows                                     #
#                                                                             #
#    The columns are the deviations x of the inputs from the point u0 of the  #
#    linearization; the input u_{j,k} = u0_{j,k} + x_{j,k} of robot j at      #
#    stage k, a velocity in the plane, is a pair of columns of the rows of    #
#    --level (the limits of the inputs), each +-x_j <= --limit -+ u0_j,       #
#    whence u0. The file gets two new columns v_{j,k} for each pair, the      #
#    hard equality rows v_{j,k} - x_{j,k} = u0_{j,k} and the quadratic hard   #
#    rows ||v_{j,k}||_2^2 <= vmax^2, so that only the right-hand sides of     #
#    the equality rows change from one MPC step to the next. The levels have  #
#    zero on the new columns; the group HQPReference and the stage structure  #
#    are not copied (the reference does not satisfy the new rows, and         #
#    HQPStageSolver does not take hard rows).                                 #
#                                                                             #
#    Usage: speedcone.py [--vmax <v>] [--level <p>] [--limit <l>] <in.nc4>    #
#                        <out.nc4>                                            #
#                                                                             #
#    with vmax 2.5, level 1 and limit 2 by default, those of the scenario.    #
#                                                                             #
#    Requires numpy, scipy and netCDF4, and npz2nc4.py alongside.             #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #

import argparse

import numpy as np
import scipy.sparse as sp
from netCDF4 import Dataset

from npz2nc4 import write_hqp


def read_rows(g, pfx, n, P):
    """the matrices (CSR, n columns), right-hand sides and weights of the
    rows pfx ('Eq' or 'Ineq') of each of the P levels of the group g"""
    if f'Num{pfx}Rows' not in g.dimensions:
        return ([sp.csr_matrix((0, n))] * P, [np.zeros(0)] * P,
                [np.zeros(0)] * P)
    m = g.dimensions[f'Num{pfx}Rows'].size
    start = np.append(g[f'{pfx}Start'][:], 0).astype(int)
    if f'{pfx}Index' in g.variables:
        index, value = g[f'{pfx}Index'][:], g[f'{pfx}Value'][:]
        start[-1] = index.size
    else:
        index, value = np.zeros(0, dtype=int), np.zeros(0)
    M = sp.csr_matrix((value, index, start), shape=(m, n))
    rhs = np.asarray(g[f'{pfx}RHS'][:], dtype=float)
    w = (np.asarray(g[f'{pfx}Weight'][:], dtype=float)
         if f'{pfx}Weight' in g.variables else np.ones(m))
    cut = np.concatenate(([0], np.cumsum(g[f'{pfx}Rows'][:]))).astype(int)
    return ([M[cut[p]:cut[p + 1]] for p in range(P)],
            [rhs[cut[p]:cut[p + 1]] for p in range(P)],
            [w[cut[p]:cut[p + 1]] for p in range(P)])


def inputs(C, d, limit):
    """the pairs of columns of the inputs and their u0, from the rows C
    x <= d of the limits of the inputs"""
    u0 = {}
    for i in range(C.shape[0]):
        row = C.getrow(i)
        if row.nnz != 1 or abs(abs(row.data[0]) - 1) > 1e-12:
            raise ValueError(f'row {i} of the limits is not +-x_j <= d_i')
        j = int(row.indices[0])
        u = limit - d[i] if row.data[0] > 0 else d[i] - limit
        if j in u0 and abs(u0[j] - u) > 1e-9 * max(1, abs(u)):
            raise ValueError(f'the limits of column {j} disagree on u0')
        u0[j] = u
    cols = sorted(u0)
    if len(cols) % 2:
        raise ValueError('an odd number of columns of the inputs')
    return np.array(cols).reshape(-1, 2), np.array([u0[j] for j in cols])


def main():
    ap = argparse.ArgumentParser(description='speed limits of the robots '
                                 'as quadratic hard rows of an HQPBlock')
    ap.add_argument('--vmax', type=float, default=2.5)
    ap.add_argument('--level', type=int, default=1)
    ap.add_argument('--limit', type=float, default=2.0)
    ap.add_argument('inp')
    ap.add_argument('out')
    a = ap.parse_args()

    with Dataset(a.inp) as f:
        g = f['Block_0']
        if any(k.startswith('Hard') for k in g.variables):
            raise ValueError(f'{a.inp} has hard rows already')
        n = g.dimensions['NumVar'].size
        P = g.dimensions['NumLevels'].size
        A, b, we = read_rows(g, 'Eq', n, P)
        C, d, wi = read_rows(g, 'Ineq', n, P)
        eps = float(g['Regularization'][...]) \
            if 'Regularization' in g.variables else 0.0
        if 'RegularizationWeight' in g.variables:
            raise ValueError(f'{a.inp} has weights of the regularization')

    pairs, u0 = inputs(C[a.level], d[a.level], a.limit)
    m = u0.size                       # the new columns, v after x
    N = n + m
    grow = lambda M: sp.hstack([M, sp.csr_matrix((M.shape[0], m))],
                               format='csr')
    E = sp.hstack([-sp.csr_matrix((np.ones(m), (np.arange(m),
                                                pairs.flatten())),
                                  shape=(m, n)), sp.eye(m)], format='csr')
    Q = [sp.csr_matrix((2 * np.ones(2), (n + 2 * i + np.arange(2),
                                         n + 2 * i + np.arange(2))),
                       shape=(N, N)) for i in range(m // 2)]
    hard = (E, u0, sp.csr_matrix((0, N)), np.zeros(0), Q,
            sp.csr_matrix((m // 2, N)), np.full(m // 2, a.vmax ** 2))
    write_hqp(a.out, N, [grow(M) for M in A], b, we, [grow(M) for M in C],
              d, wi, eps, hard=hard)


if __name__ == '__main__':
    main()
