#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#    npz2nc4.py: converts a hierarchical QP from .npz to the SMS++ netCDF     #
#    format of HQPBlock                                                       #
#                                                                             #
#    The .npz holds, for each level p = 0, 1, ... (0 the highest priority),   #
#    the arrays A_<p>, b_<p>, C_<p>, d_<p> of the tasks                       #
#                                                                             #
#        we_p ( A_p x - b_p ) = v_p ,   wi_p ( C_p x - d_p ) <= w_p ,         #
#                                                                             #
#    dense or as 0-row matrices for a level with no task of that kind (a      #
#    missing key is taken as empty), and optionally we_<p>, wi_<p> (a scalar  #
#    or a vector, all ones if missing); the number of levels is n_levels if   #
#    there, the number of the A_<p> otherwise. The output is an eBlockFile    #
#    whose Block_0 is the HQPBlock [see HQPBlock::deserialize()], with the    #
#    regularization of --eps. If the .npz has xref_<p>, the solution after    #
#    level p of some other code, they go in the variable X (NumLevels x       #
#    NumVar) of the group HQPReference of the file, which HQPBlock ignores.   #
#                                                                             #
#    Usage: npz2nc4.py [--eps <e>] <in.npz> <out.nc4>                         #
#                                                                             #
#    Requires numpy and netCDF4.                                              #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #

import argparse

import numpy as np
from netCDF4 import Dataset


def level_rows(npz, key, p, n):
    """the matrix key_p as a dense 2D array with n columns (0 rows if not
    there or empty)"""
    k = f'{key}_{p}'
    if k not in npz or npz[k].size == 0:
        return np.zeros((0, n))
    M = np.atleast_2d(np.asarray(npz[k], dtype=float))
    if M.shape[1] != n:
        raise ValueError(f'{k} has {M.shape[1]} columns instead of {n}')
    return M


def level_vector(npz, key, p, m, default=None):
    """the vector key_p of size m (default if not there; a scalar is
    repeated)"""
    k = f'{key}_{p}'
    if k not in npz or npz[k].size == 0:
        if default is None and m:
            raise ValueError(f'{k} is missing')
        return np.full(m, 1.0 if default is None else default)
    v = np.asarray(npz[k], dtype=float).flatten()
    if v.size == 1 and m != 1:
        v = np.full(m, v[0])
    if v.size != m:
        raise ValueError(f'{k} has {v.size} entries instead of {m}')
    return v


def write_rows(g, pfx, mats, rhs, wgh, levels):
    """the rows of one kind of all the levels, as HQPBlock::deserialize()
    reads them"""
    m = sum(M.shape[0] for M in mats)
    if m == 0:
        return
    start, index, value = [], [], []
    for M in mats:
        for row in M:
            start.append(len(index))
            nz = np.nonzero(row)[0]
            index.extend(nz.tolist())
            value.extend(row[nz].tolist())
    g.createDimension(f'Num{pfx}Rows', m)
    g.createVariable(f'{pfx}Rows', 'u4', ('NumLevels',))[:] = \
        [M.shape[0] for M in mats]
    g.createVariable(f'{pfx}Start', 'u4', (f'Num{pfx}Rows',))[:] = start
    if index:
        g.createDimension(f'Num{pfx}Elements', len(index))
        g.createVariable(f'{pfx}Index', 'u4',
                         (f'Num{pfx}Elements',))[:] = index
        g.createVariable(f'{pfx}Value', 'f8',
                         (f'Num{pfx}Elements',))[:] = value
    g.createVariable(f'{pfx}RHS', 'f8', (f'Num{pfx}Rows',))[:] = \
        np.concatenate(rhs)
    w = np.concatenate(wgh)
    if np.any(w != 1):
        g.createVariable(f'{pfx}Weight', 'f8', (f'Num{pfx}Rows',))[:] = w


def main():
    ap = argparse.ArgumentParser(description='hierarchical QP from .npz '
                                 'to the netCDF format of HQPBlock')
    ap.add_argument('--eps', type=float, default=0.0,
                    help='weight of the regularization (default 0)')
    ap.add_argument('npz')
    ap.add_argument('nc4')
    a = ap.parse_args()

    npz = np.load(a.npz)
    if 'n_levels' in npz:
        P = int(npz['n_levels'])
    else:
        P = 0
        while f'A_{P}' in npz or f'C_{P}' in npz:
            P += 1
    if P == 0:
        raise ValueError('no levels')
    n = next(np.atleast_2d(npz[k]).shape[1] for p in range(P)
             for k in (f'A_{p}', f'C_{p}') if k in npz and npz[k].ndim == 2)

    A = [level_rows(npz, 'A', p, n) for p in range(P)]
    C = [level_rows(npz, 'C', p, n) for p in range(P)]
    b = [level_vector(npz, 'b', p, A[p].shape[0], 0.0 if not A[p].shape[0]
                      else None) for p in range(P)]
    d = [level_vector(npz, 'd', p, C[p].shape[0], 0.0 if not C[p].shape[0]
                      else None) for p in range(P)]
    we = [level_vector(npz, 'we', p, A[p].shape[0], 1.0) for p in range(P)]
    wi = [level_vector(npz, 'wi', p, C[p].shape[0], 1.0) for p in range(P)]

    with Dataset(a.nc4, 'w', format='NETCDF4') as f:
        f.setncattr('SMS++_file_type', np.int32(1))
        g = f.createGroup('Block_0')
        g.setncattr('type', 'HQPBlock')
        g.createDimension('NumVar', n)
        g.createDimension('NumLevels', P)
        write_rows(g, 'Eq', A, b, we, P)
        write_rows(g, 'Ineq', C, d, wi, P)
        if a.eps != 0:
            g.createVariable('Regularization', 'f8')[...] = a.eps
        if all(f'xref_{p}' in npz for p in range(P)):
            r = f.createGroup('HQPReference')
            r.createDimension('NumLevels', P)
            r.createDimension('NumVar', n)
            r.createVariable('X', 'f8', ('NumLevels', 'NumVar'))[:] = \
                np.vstack([np.asarray(npz[f'xref_{p}']).flatten()
                           for p in range(P)])


if __name__ == '__main__':
    main()
