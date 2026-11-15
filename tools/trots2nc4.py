#!/usr/bin/env python3
# --------------------------------------------------------------------------- #
#    trots2nc4.py: converts the lexicographic problem of a case of TROTS,     #
#    the Radiotherapy Optimisation Test Set, into the SMS++ netCDF format     #
#    of HQPBlock                                                              #
#                                                                             #
#    A case of TROTS (S. Breedveld, B. Heijmen, Data in Brief 12:143-149,     #
#    2017, doi:10.1016/j.dib.2017.03.037) is a MAT 7.3 file whose structure   #
#    problem_lex lists constraints and objectives on the doses d = A x + b of #
#    the structures of the patient, x >= 0 being the pencil-beam weights,     #
#    each objective with a priority and, possibly, a sufficient value. Only   #
#    the linear cost-functions (Type 1: max(d) if Minimise, min(d) if not, a  #
#    mean being a matrix of one row), which are all those of the proton       #
#    cases, the convex quadratic ones (Type 2, 1/2 x' A x + b' x + c), the    #
#    gEUD ones with a >= 1 (Type 3, Parameters a, m^(-1/a) ||d||_a on the m   #
#    nonnegative doses) and the LTCP ones (Type 4, Parameters (d_p, alpha),   #
#    (1/m) sum_i exp(-alpha (d_i - d_p))), the last three minimised or        #
#    bounded from above, are converted; a case with others is refused. A      #
#    quadratic constraint is a quadratic hard row, a gEUD one a p-norm hard   #
#    row and an LTCP one a sum-exp hard row, and such an objective f gets a   #
#    column t with the hard row f - t <= 0, which needs a sufficient value.   #
#    Each linear objective of more than one row gets a column t, bounding it  #
#    by hard rows, which also are the constraints and x >= 0, and the         #
#    objectives of the k-th priority are the inequality tasks of level k - 1, #
#    t <= s (t >= s for a maximized minimum) with s the sufficient value (0   #
#    if there is none for a minimized maximum of nonnegative doses, the task  #
#    then weighted by the inverse of the largest 1-norm of its rows, e.g.,    #
#    the mean of the weights rather than their sum, whose square is out of    #
#    scale), so that each level minimizes the squared excess over s of its    #
#    objectives and keeps those of the higher ones at their optimal values;   #
#    with --soft, the hard rows are rather the inequality tasks of a level 0, #
#    and the objectives of the k-th priority those of level k. The solution   #
#    of TROTS (solutionX), with each t at its least value, is written in the  #
#    group HQPPoint (variable X, NumVar), which HQPBlock ignores. With        #
#    --weighted, the weighted sum of the objectives of the case (structure    #
#    problem, whose optimum solutionX is) is rather written as the Linear     #
#    Program of an AbstractBlock in the MPS format, its value at the solution #
#    of TROTS in HQPPoint (variable Value).                                   #
#                                                                             #
#    Usage: trots2nc4.py [--eps <e>] [--soft | --weighted] <case.mat>         #
#                        <out.nc4>                                            #
#                                                                             #
#    Requires numpy, scipy, h5py and netCDF4, and npz2nc4.py alongside.       #
#                                                                             #
#                                Donato Meoli                                 #
#                         Dipartimento di Informatica                         #
#                             Universita' di Pisa                             #
# --------------------------------------------------------------------------- #

import argparse

import h5py
import numpy as np
import scipy.sparse as sp
from netCDF4 import Dataset

from npz2nc4 import write_hqp


def field(f, entry, i):
    """the entry i of a field of a structure array of f, as a flat array
    (None if empty)"""
    o = f[np.ravel(entry[()])[i]]
    if o.attrs.get('MATLAB_empty', 0):
        return None
    return np.ravel(o[()])


def matrix(f, i):
    """the matrix i (from 0) of data.matrix, as CSR, and its offset b"""
    o = f[np.ravel(f['data/matrix/A'][()])[i]]
    if isinstance(o, h5py.Group):
        m = int(o.attrs['MATLAB_sparse'])
        jc = o['jc'][()]
        A = sp.csc_matrix((o['data'][()], o['ir'][()], jc),
                          shape=(m, len(jc) - 1)).tocsr()
    else:  # dense, stored transposed
        a = np.asarray(o[()], dtype=float)
        A = sp.csr_matrix(a.T if a.ndim == 2 else a.reshape(1, -1))
    b = field(f, f['data/matrix/b'], i)
    b = np.zeros(A.shape[0]) if b is None else np.broadcast_to(
        np.asarray(b, dtype=float), (A.shape[0],)).copy()
    return A, b


def pad(Q, N):
    """the symmetric part (Q + Q') / 2 of the square matrix Q, which gives
    the same 1/2 x' Q x, widened to N x N with zeros"""
    Q = sp.coo_matrix((Q + Q.T) / 2)
    return sp.csr_matrix((Q.data, (Q.row, Q.col)), shape=(N, N))


def quad_const(f, i):
    """the constant c of the quadratic cost-function of the matrix i (0 if
    empty)"""
    c = field(f, f['data/matrix/c'], i)
    return 0.0 if c is None else float(c[0])


def quad_value(f, mats, i, x):
    """1/2 x' A x + b' x + c for the quadratic cost-function of the matrix
    i at x"""
    A, b = mats[i]
    return float(0.5 * x @ (A @ x) + b @ x + quad_const(f, i))


def conic_value(e, mats, x):
    """the value at x of the LTCP or of the gEUD cost-function of e"""
    A, b = mats[e['data']]
    dose = A @ x + b
    m = dose.size
    if e['kind'] == 'ltcp':
        dp, alpha = e['par']
        return float(np.exp(-alpha * (dose - dp)).mean())
    a = e['par'][0]
    return float(m ** (-1.0 / a) * np.linalg.norm(dose, ord=a))


def write_weighted(a, f, n, nt, cons, objs, mats, rows, X):
    """the weighted sum of the objectives as the Linear Program of an
    AbstractBlock in a.nc4, min sum_k w_k t_k (an objective of one row
    being its row itself) under the rows of the constraints and of the t,
    and its value at X, if any"""
    N = n + nt
    c = np.zeros(N)
    const = 0.0
    M, lo, up = [], [], []
    for e in cons:
        A, b = mats[e['data']]
        M.append(rows(A))
        inf = np.full(A.shape[0], np.inf)
        lo.append(-inf if e['max'] else e['rhs'] - b)
        up.append(e['rhs'] - b if e['max'] else inf)
    for e in objs:
        A, b = mats[e['data']]
        sg = 1.0 if e['max'] else -1.0  # min max(d), max min(d)
        if e['t'] is None:
            c[:n] += sg * e['w'] * A.toarray().ravel()
            const += sg * e['w'] * b[0]
            continue
        c[e['t']] += sg * e['w']
        M.append(rows(A, e['t'], -1.0))
        inf = np.full(A.shape[0], np.inf)
        lo.append(-inf if e['max'] else -b)
        up.append(-b if e['max'] else inf)
    M = sp.vstack(M, format='csc')
    lo = np.concatenate(lo)
    up = np.concatenate(up)
    kind = np.where(np.isinf(lo), 'L', 'G')
    rhs = np.where(np.isinf(lo), up, lo)

    out = ['NAME TROTS', 'ROWS', ' N OBJ']
    out += [f' {k} R{i}' for i, k in enumerate(kind)]
    out.append('COLUMNS')
    for j in range(N):
        cj = f'    C{j} '
        if c[j] != 0:
            out.append(f'{cj}OBJ {float(c[j])!r}')
        s, t = M.indptr[j], M.indptr[j + 1]
        out += [f'{cj}R{i} {float(v)!r}'
                for i, v in zip(M.indices[s:t], M.data[s:t])]
    out.append('RHS')
    if const != 0:
        out.append(f'    RHS OBJ {-const!r}')
    out += [f'    RHS R{i} {float(v)!r}' for i, v in enumerate(rhs) if v]
    out.append('BOUNDS')
    out += [f' FR BND C{j}' for j in range(n, N)]
    out.append('ENDATA')

    with Dataset(a.nc4, 'w', format='NETCDF4') as g:
        g.setncattr('SMS++_file_type', np.int32(1))
        b0 = g.createGroup('Block_0')
        b0.setncattr('type', 'AbstractBlock')
        v = b0.createVariable('Model', str)
        v[0] = '\n'.join(out) + '\n'
        v.setncattr('ModelType', 'M')
        if X is not None:
            r = g.createGroup('HQPPoint')
            r.createDimension('NumVar', N)
            r.createVariable('X', 'f8', ('NumVar',))[:] = X
            r.createVariable('Value', 'f8')[...] = c @ X + const

    print(f'{a.mat}: {N} columns ({n} beamlets, {nt} t), weighted sum, '
          f'{M.shape[0]} rows, {M.nnz} non-zeros' +
          ('' if X is None else f', value {c @ X + const} at the solution '
           'of TROTS'))


def main():
    ap = argparse.ArgumentParser(description='lexicographic problem of a '
                                 'case of TROTS to the netCDF format of '
                                 'HQPBlock')
    ap.add_argument('--eps', type=float, default=0.0,
                    help='weight of the regularization (default 0)')
    kind = ap.add_mutually_exclusive_group()
    kind.add_argument('--soft', action='store_true',
                      help='the hard rows as a level 0 rather than as hard '
                      'rows')
    kind.add_argument('--weighted', action='store_true',
                      help='the weighted sum of the objectives, as the '
                      'Linear Program of an AbstractBlock')
    ap.add_argument('mat')
    ap.add_argument('nc4')
    a = ap.parse_args()

    f = h5py.File(a.mat, 'r')
    st = 'problem' if a.weighted else 'problem_lex'
    if st not in f:
        raise ValueError(f'{a.mat} has no {st}')
    L = f[st]
    misc = f['data/misc']
    n = int(np.ravel(misc['size'][()])[0])
    if int(np.ravel(misc['real'][()])[0]) != n:
        raise ValueError(f'{a.mat} has auxiliary decision variables')

    cons, objs = [], []
    for i in range(L['Type'].shape[0]):
        get = lambda k: field(f, L[k], i)
        if get('Active') is not None and not get('Active')[0]:
            continue
        name = ''.join(map(chr, get('Name').astype(int)))
        tp = int(get('Type')[0])
        if tp not in (1, 2, 3, 4):
            raise ValueError(f'{a.mat}: "{name}" has the cost-function '
                             f'type {tp}, only the linear (1), quadratic '
                             f'(2), gEUD (3) and LTCP (4) ones are '
                             f'converted')
        kind = {1: 'lin', 2: 'quad', 3: 'geud', 4: 'ltcp'}[tp]
        e = dict(name=name, data=int(get('dataID')[0]) - 1,
                 max=bool(get('Minimise')[0]), quad=(tp == 2),
                 conic=(tp in (3, 4)), kind=kind)
        if (e['quad'] or e['conic']) and (a.soft or a.weighted):
            raise ValueError(f'{a.mat}: "{name}" is {kind}, which only '
                             f'the hard rows take (neither --soft nor '
                             f'--weighted)')
        if (e['quad'] or e['conic']) and not e['max']:
            raise ValueError(f'{a.mat}: "{name}" is {kind} maximised or '
                             f'bounded from below, which is not convex')
        if e['conic']:
            par = get('Parameters')
            e['par'] = [] if par is None else [float(v) for v in par]
            if kind == 'geud' and (len(e['par']) < 1 or
                                   not 1 <= e['par'][0] < np.inf):
                raise ValueError(f'{a.mat}: "{name}" is a gEUD with '
                                 f'parameters {e["par"]}, only a >= 1 '
                                 f'(minimised or bounded from above) is '
                                 f'convex and converted')
            if kind == 'ltcp' and (len(e['par']) < 2 or
                                   not np.all(np.isfinite(e['par'][:2]))):
                raise ValueError(f'{a.mat}: "{name}" is an LTCP with '
                                 f'parameters {e["par"]} rather than the '
                                 f'finite (d_p, alpha)')
        if get('IsConstraint')[0]:
            e['rhs'] = float(get('Objective')[0])
            cons.append(e)
        elif a.weighted:
            w = get('Weight')
            e['w'] = 1.0 if w is None else float(w[0])
            objs.append(e)
        else:
            s = get('Sufficient')
            e['prio'] = int(get('Priority')[0])
            e['rhs'] = None if s is None else float(s[0])
            objs.append(e)

    # the columns: x, then a t for each objective of more than one row
    mats = {}
    for e in cons + objs:
        if e['data'] not in mats:
            mats[e['data']] = matrix(f, e['data'])
    nt = 0
    for e in objs:
        e['t'] = n + nt if (e['quad'] or e['conic'] or
                            mats[e['data']][0].shape[0] > 1) else None
        nt += e['t'] is not None
    N = n + nt

    def rows(A, t=None, coef=0.0):
        """A, widened to N columns, with coef in column t"""
        m = A.shape[0]
        T = sp.csr_matrix((m, nt)) if t is None else sp.csr_matrix(
            (np.full(m, coef), (np.arange(m), np.full(m, t - n))),
            shape=(m, nt))
        return sp.hstack([A, T], format='csr')

    # the solution of TROTS, each t at its least value
    X = None
    if 'solutionX' in f:
        x = np.ravel(f['solutionX'][()]).astype(float)
        X = np.concatenate([x, np.zeros(nt)])
        for e in objs:
            if e['quad']:
                X[e['t']] = quad_value(f, mats, e['data'], x)
            elif e['conic']:
                X[e['t']] = conic_value(e, mats, x)
            elif e['t'] is not None:
                Ae, be = mats[e['data']]
                dose = Ae @ x + be
                X[e['t']] = dose.max() if e['max'] else dose.min()

    if a.weighted:
        write_weighted(a, f, n, nt, cons, objs, mats, rows, X)
        return

    # the hard rows: x >= 0, the constraints, the rows of the t
    C0 = [sp.hstack([-sp.identity(n), sp.csr_matrix((n, nt))],
                    format='csr')]
    d0 = [np.zeros(n)]
    Q, Cq, eq = [], [], []
    se = ([], [], [], [], [])  # S, b, c, L, u of the sum-exp hard rows
    pn = ([], [], [], [], [], [])  # A, b, p, s, L, u of the p-norm ones

    def conic_row(e, t, u):
        """the LTCP of e as the sum-exp hard row (1/m) sum_i exp(-alpha
        (d_i - d_p)) - t <= u, its gEUD as the p-norm hard row m^(-1/a)
        ||d||_a - t <= u (the gEUD of nonnegative doses), t the column t
        (none if None)"""
        A, b = mats[e['data']]
        m = A.shape[0]
        L = rows(sp.csr_matrix((1, n)), t, -1.0)
        if e['kind'] == 'ltcp':
            dp, alpha = e['par'][:2]
            for v, w in zip(se, (rows(-alpha * A), -alpha * (b - dp),
                                 np.full(m, 1.0 / m), L, u)):
                v.append(w)
        else:
            p = e['par'][0]
            for v, w in zip(pn, (rows(A), b, p, m ** (-1.0 / p), L, u)):
                v.append(w)

    for e in cons:
        A, b = mats[e['data']]
        if e['conic']:
            conic_row(e, None, e['rhs'])
            continue
        if e['quad']:
            Q.append(pad(A, N))
            Cq.append(rows(sp.csr_matrix(b.reshape(1, -1))))
            eq.append(e['rhs'] - quad_const(f, e['data']))
            continue
        sg = 1.0 if e['max'] else -1.0
        C0.append(sg * rows(A))
        d0.append(sg * (e['rhs'] - b))
    for e in objs:
        if e['conic']:
            conic_row(e, e['t'], 0.0)
            continue
        if e['quad']:
            A, b = mats[e['data']]
            Q.append(pad(A, N))
            Cq.append(rows(sp.csr_matrix(b.reshape(1, -1)), e['t'], -1.0))
            eq.append(-quad_const(f, e['data']))
            continue
        if e['t'] is None:
            continue
        A, b = mats[e['data']]
        sg = 1.0 if e['max'] else -1.0
        C0.append(sg * rows(A, e['t'], -1.0))
        d0.append(-sg * b)
    C = [sp.vstack(C0, format='csr')]
    d = [np.concatenate(d0)]
    w = [np.ones(C[0].shape[0])]

    # level k: the objectives of the k-th priority
    for prio in sorted({e['prio'] for e in objs}):
        Ck, dk, wk = [], [], []
        for e in (e for e in objs if e['prio'] == prio):
            sg = 1.0 if e['max'] else -1.0
            s = e['rhs']
            wg = 1.0
            if s is None:
                A, b = mats[e['data']]
                if (e['quad'] or e['conic'] or not e['max'] or
                        A.min() < 0 or b.min() < 0):
                    raise ValueError(f'{a.mat}: "{e["name"]}" has no '
                                     f'sufficient value')
                s = 0.0
                wg = 1.0 / max(abs(A).sum(axis=1).max(), 1e-300)
            wk.append(np.full(1 if e['t'] is not None else
                              mats[e['data']][0].shape[0], wg))
            if e['t'] is None:
                A, b = mats[e['data']]
                Ck.append(sg * rows(A))
                dk.append(sg * (s - b))
            else:
                r = sp.csr_matrix(([sg], ([0], [e['t']])), shape=(1, N))
                Ck.append(r)
                dk.append(np.array([sg * s]))
        C.append(sp.vstack(Ck, format='csr'))
        d.append(np.concatenate(dk))
        w.append(np.concatenate(wk))

    for M in C:
        M.sort_indices()
    hard = None
    if not a.soft:
        hard = (sp.csr_matrix((0, N)), np.zeros(0), C.pop(0), d.pop(0),
                Q, sp.vstack(Cq, format='csr') if Cq else
                sp.csr_matrix((0, N)), np.array(eq))
        w.pop(0)
    P = len(C)
    A = [sp.csr_matrix((0, N)) for _ in range(P)]
    b = [np.zeros(0) for _ in range(P)]
    write_hqp(a.nc4, N, A, b, [np.zeros(0)] * P, C, d, w, a.eps, hard=hard,
              sumexp=(se[0], se[1], se[2], sp.vstack(se[3], format='csr')
                      if se[3] else sp.csr_matrix((0, N)), np.array(se[4])),
              pnorm=(pn[0], pn[1], np.array(pn[2]), np.array(pn[3]),
                     sp.vstack(pn[4], format='csr') if pn[4] else
                     sp.csr_matrix((0, N)), np.array(pn[5])))

    if X is not None:
        with Dataset(a.nc4, 'a') as g:
            r = g.createGroup('HQPPoint')
            r.createDimension('NumVar', N)
            r.createVariable('X', 'f8', ('NumVar',))[:] = X

    H = [] if hard is None else [hard[2]]
    print(f'{a.mat}: {N} columns ({n} beamlets, {nt} t), '
          f'{sum(M.shape[0] for M in H)} hard rows ({len(Q)} quadratic, '
          f'{len(se[4])} sum-exp, {len(pn[5])} p-norm), '
          f'{P} levels, '
          f'{sum(M.shape[0] for M in C)} rows, '
          f'{sum(M.nnz for M in C + H)} non-zeros')


if __name__ == '__main__':
    main()
