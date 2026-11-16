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
#    there, the number of the A_<p> otherwise. With n_stages > 0, the .npz    #
#    also holds the stage structure: var_stage and var_state, the stage of    #
#    each column and 1 for a state, 0 for an input, optionally eq_stage and   #
#    ineq_stage, the stage of each row of each kind of all the levels, and    #
#    dynamics_level, the level of the dynamics (0 if missing) [see            #
#    HQPBlock::StageData]. The output is an eBlockFile                        #
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


def write_rows(g, pfx, mats, rhs, wgh, levels=True):
    """the rows of one kind of all the levels, as HQPBlock::deserialize()
    reads them; if not levels, the hard rows of one kind (mats and rhs of
    one entry), with neither the rows of each level nor the weights; if
    levels is the name of a dimension, it takes the place of NumLevels
    (e.g., the terms of the sum-exp hard rows, one entry of mats each)"""
    m = sum(M.shape[0] for M in mats)
    if m == 0:
        return
    start, index, value = [], [], []
    for M in mats:
        if hasattr(M, 'indptr'):  # a CSR matrix, e.g., of scipy.sparse
            start.extend((len(index) + M.indptr[:-1]).tolist())
            index.extend(M.indices.tolist())
            value.extend(M.data.tolist())
            continue
        for row in M:
            start.append(len(index))
            nz = np.nonzero(row)[0]
            index.extend(nz.tolist())
            value.extend(row[nz].tolist())
    g.createDimension(f'Num{pfx}Rows', m)
    if levels:
        dim = levels if isinstance(levels, str) else 'NumLevels'
        g.createVariable(f'{pfx}Rows', 'u4', (dim,))[:] = \
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
    if not levels:
        return
    w = np.concatenate(wgh)
    if np.any(w != 1):
        g.createVariable(f'{pfx}Weight', 'f8', (f'Num{pfx}Rows',))[:] = w


def write_stages(g, npz):
    """the stage structure, as HQPBlock::deserialize() reads it"""
    g.createDimension('NumStages', int(npz['n_stages']))
    for key, var, dim in (('var_stage', 'VarStage', 'NumVar'),
                          ('var_state', 'VarState', 'NumVar'),
                          ('eq_stage', 'EqStage', 'NumEqRows'),
                          ('ineq_stage', 'IneqStage', 'NumIneqRows')):
        if key not in npz:
            if key.startswith('var'):
                raise ValueError(f'{key} is missing')
            continue
        v = np.asarray(npz[key], dtype=np.uint32).flatten()
        if dim not in g.dimensions or v.size != g.dimensions[dim].size:
            raise ValueError(f'{key} has not one entry for each of {dim}')
        g.createVariable(var, 'u4', (dim,))[:] = v
    g.createVariable('DynamicsLevel', 'u4')[...] = \
        int(npz.get('dynamics_level', 0))


def write_quad_rows(g, Q, Cq, eq):
    """the quadratic hard rows 1/2 x' Q[i] x + Cq[i] x <= eq[i], as
    HQPBlock::deserialize() reads them: the linear part as the hard rows
    HardQuad, the lower triangle of each Q[i] as triplets (needs scipy)"""
    import scipy.sparse as sp
    write_rows(g, 'HardQuad', [sp.csr_matrix(Cq)], [np.asarray(eq)], None,
               False)
    qr, qi, qj, qv = [], [], [], []
    for i, Qi in enumerate(Q):
        L = sp.tril(sp.coo_matrix(Qi)).tocoo()
        L.sum_duplicates()
        qr += [i] * L.nnz
        qi += L.row.tolist()
        qj += L.col.tolist()
        qv += L.data.tolist()
    if qv:
        g.createDimension('NumHardQuadQElements', len(qv))
        for name, v, t in (('HardQuadQRow', qr, 'u4'),
                           ('HardQuadQI', qi, 'u4'),
                           ('HardQuadQJ', qj, 'u4'),
                           ('HardQuadQValue', qv, 'f8')):
            g.createVariable(name, t, ('NumHardQuadQElements',))[:] = v


def write_sumexp_rows(g, S, b, c, L, u):
    """the sum-exp hard rows sum_k c[i][k] exp(S[i][k] x + b[i][k]) + L[i] x
    <= u[i], as HQPBlock::deserialize() reads them: L and u as the hard rows
    HardSumExp, the terms as HardSumExpTerm, the hard rows in place of the
    levels (S[i] dense or CSR, needs scipy)"""
    import scipy.sparse as sp
    if len(u) == 0:
        return
    write_rows(g, 'HardSumExp', [sp.csr_matrix(L)],
               [np.asarray(u, dtype=float)], None, False)
    write_rows(g, 'HardSumExpTerm', [sp.csr_matrix(M) for M in S],
               [np.asarray(v, dtype=float) for v in b],
               [np.asarray(v, dtype=float) for v in c], 'NumHardSumExpRows')


def write_pnorm_rows(g, A, b, p, s, L, u):
    """the p-norm hard rows s[i] ||A[i] x + b[i]||_p[i] + L[i] x <= u[i], as
    HQPBlock::deserialize() reads them, as write_sumexp_rows() with HardPNorm
    and the exponents p and the scales s (needs scipy)"""
    import scipy.sparse as sp
    if len(u) == 0:
        return
    write_rows(g, 'HardPNorm', [sp.csr_matrix(L)],
               [np.asarray(u, dtype=float)], None, False)
    bs = [np.asarray(v, dtype=float) for v in b]
    write_rows(g, 'HardPNormTerm', [sp.csr_matrix(M) for M in A], bs,
               [np.ones(v.size) for v in bs], 'NumHardPNormRows')
    g.createVariable('HardPNormP', 'f8', ('NumHardPNormRows',))[:] = p
    if np.any(np.asarray(s) != 1):
        g.createVariable('HardPNormScale', 'f8',
                         ('NumHardPNormRows',))[:] = s


def write_hqp(path, n, A, b, we, C, d, wi, eps=0.0, stages=None,
              xref=None, hard=None, sumexp=None, pnorm=None):
    """the eBlockFile path whose Block_0 is the HQPBlock of n columns with
    the levels A, b, we, C, d, wi (a list each, one entry per level, the
    matrices dense or CSR), the regularization eps, the stage structure in
    stages (a dict with the keys of the .npz), if given, the solution
    after each level in xref and the hard rows in hard, the tuple (A_h,
    b_h, C_h, d_h) or (A_h, b_h, C_h, d_h, Q, C_q, e_q) with the quadratic
    hard rows 1/2 x' Q[i] x + C_q[i] x <= e_q[i], each Q[i] a symmetric
    sparse n x n matrix of which the lower triangle is written, the sum-exp
    hard rows in sumexp, the tuple (S, b, c, L, u) of write_sumexp_rows(),
    and the p-norm ones in pnorm, the tuple (A, b, p, s, L, u) of
    write_pnorm_rows()"""
    P = len(A)
    with Dataset(path, 'w', format='NETCDF4') as f:
        f.setncattr('SMS++_file_type', np.int32(1))
        g = f.createGroup('Block_0')
        g.setncattr('type', 'HQPBlock')
        g.createDimension('NumVar', n)
        g.createDimension('NumLevels', P)
        write_rows(g, 'Eq', A, b, we, P)
        write_rows(g, 'Ineq', C, d, wi, P)
        if hard is not None:
            write_rows(g, 'HardEq', [hard[0]], [hard[1]], None, False)
            write_rows(g, 'HardIneq', [hard[2]], [hard[3]], None, False)
            if len(hard) > 4 and len(hard[6]):
                write_quad_rows(g, hard[4], hard[5], hard[6])
        if sumexp is not None:
            write_sumexp_rows(g, *sumexp)
        if pnorm is not None:
            write_pnorm_rows(g, *pnorm)
        if eps != 0:
            g.createVariable('Regularization', 'f8')[...] = eps
        if stages is not None:
            write_stages(g, stages)
        if xref is not None:
            r = f.createGroup('HQPReference')
            r.createDimension('NumLevels', P)
            r.createDimension('NumVar', n)
            r.createVariable('X', 'f8', ('NumLevels', 'NumVar'))[:] = \
                np.vstack(xref)


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

    xref = None
    if all(f'xref_{p}' in npz for p in range(P)):
        xref = [np.asarray(npz[f'xref_{p}']).flatten() for p in range(P)]
    write_hqp(a.nc4, n, A, b, we, C, d, wi, a.eps,
              npz if int(npz.get('n_stages', 0)) > 0 else None, xref)


if __name__ == '__main__':
    main()
