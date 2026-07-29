#!/usr/bin/env python3
"""Statistical error on the measured twin/non-twin eccentricity-index
difference, by bootstrap over systems. Per-system log L(e) grids are already
computed (ecc_marginal_real.csv). Precompute each system's per-alpha marginal
M[i,a]=logsumexp_e( logL_i(e)+log de + a*log e ); then alpha for any
resample/weighting is argmax_a[ sum_i w_i M[i,a] + (sum w_i) log(1+a) ].
Free bootstrap, no refit."""
import os, sys
import numpy as np, pandas as pd
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); import ecc_marginal as EM

EG = EM.default_egrid(); DE = np.gradient(EG); LOE = np.log(np.clip(EG,1e-6,None))
A = np.linspace(-0.95, 3.0, 300)
# f(e|a) is normalized on [0, EMAX], so the normalizer is (1+a)/EMAX^(1+a).
# Dropping the EMAX term tilts the fit, since it depends on alpha.
L1A = np.log1p(A) - (1.0 + A) * np.log(EM.EMAX)
def parse(s): return np.array([float(x) for x in str(s).split(";")])

def marginal_matrix(LG):
    """LG (nsys,ngrid) -> M (nsys, nA): logsumexp_e(LG + log de + a*log e)."""
    T = LG + np.log(DE)[None,:]                      # (nsys, ngrid)
    # tm[i,e,a] = T[i,e] + A[a]*LOE[e]; logsumexp over e
    tm = T[:,:,None] + LOE[None,:,None]*A[None,None,:]   # (nsys, ngrid, nA)
    mx = tm.max(axis=1)                               # (nsys, nA)
    return mx + np.log(np.sum(np.exp(tm - mx[:,None,:]), axis=1))

def alpha_from_M(M, w):
    ll = (w[:,None]*M).sum(axis=0) + w.sum()*L1A
    return A[int(np.argmax(ll))]

def matched_w(twK,twE, ntK,ntE):
    kb=np.array([0,5,8,12,18,25,40,200]); eb=np.array([8,10,12,14,18,30])
    tc=Counter(zip(np.digitize(twK,kb),np.digitize(twE,eb)))
    nc=Counter(zip(np.digitize(ntK,kb),np.digitize(ntE,eb)))
    ntc=list(zip(np.digitize(ntK,kb),np.digitize(ntE,eb)))
    return np.clip(np.array([(tc[c]/len(twK))/max(nc[c]/len(ntK),1e-9) for c in ntc]),0,20)

def run(d, kcut, nboot=3000, seed=3):
    m = d[d.K1>kcut]
    tw, nt = m[m.twin], m[~m.twin]
    LGt=np.array([parse(x) for x in tw.loglike]); LGt[~np.isfinite(LGt)]=-1e9
    LGn=np.array([parse(x) for x in nt.loglike]); LGn[~np.isfinite(LGn)]=-1e9
    Mt, Mn = marginal_matrix(LGt), marginal_matrix(LGn)
    Kt,Et=tw.K1.values,tw.n_ep.values; Kn,En=nt.K1.values,nt.n_ep.values
    at = alpha_from_M(Mt, np.ones(len(tw)))
    an = alpha_from_M(Mn, matched_w(Kt,Et,Kn,En))
    rng=np.random.default_rng(seed); boot=np.empty(nboot)
    for b in range(nboot):
        it=rng.integers(0,len(tw),len(tw)); jn=rng.integers(0,len(nt),len(nt))
        bt=alpha_from_M(Mt[it], np.ones(len(tw)))
        bn=alpha_from_M(Mn[jn], matched_w(Kt[it],Et[it],Kn[jn],En[jn]))
        boot[b]=bt-bn
    return dict(kcut=kcut,n_tw=len(tw),n_nt=len(nt),at=at,an=an,meas=at-an,
                bstd=boot.std(ddof=1),bmean=boot.mean(),p_pos=float(np.mean(boot>=0)))

if __name__=="__main__":
    d=pd.read_csv(os.path.join(_ROOT,"resources/census/ecc_marginal_real.csv"))
    if "P50" in d: d=d[(d.P50>=6)&(d.P50<=400)]
    print("n(P 6-400d):",len(d))
    print("%-5s %-6s %-6s %-8s %-8s %-8s %-8s %-7s"%
          ("kcut","n_tw","n_nt","a_twin","a_nont","dAlpha","boot_sd","p(>=0)"))
    for kc in [8,10,12,14,16]:
        r=run(d,kc)
        print("%-5.0f %-6d %-6d %+8.3f %+8.3f %+8.3f %8.3f %7.3f"%
              (r["kcut"],r["n_tw"],r["n_nt"],r["at"],r["an"],r["meas"],r["bstd"],r["p_pos"]))

def permutation(d, kcut, nperm=4000, seed=11):
    """Relabel twin/non-twin at random (same counts), recompute matched dAlpha.
    Null: the twin property carries no eccentricity difference."""
    m = d[d.K1>kcut]
    LG=np.array([parse(x) for x in m.loglike]); LG[~np.isfinite(LG)]=-1e9
    M = marginal_matrix(LG); K=m.K1.values; E=m.n_ep.values
    tw=m.twin.values.astype(bool); ntw=int(tw.sum())
    at=alpha_from_M(M[tw], np.ones(ntw))
    an=alpha_from_M(M[~tw], matched_w(K[tw],E[tw],K[~tw],E[~tw]))
    meas=at-an
    rng=np.random.default_rng(seed); perm=np.empty(nperm)
    idx=np.arange(len(m))
    for b in range(nperm):
        p=rng.permutation(idx); ti=p[:ntw]; ni=p[ntw:]
        a1=alpha_from_M(M[ti], np.ones(ntw))
        a2=alpha_from_M(M[ni], matched_w(K[ti],E[ti],K[ni],E[ni]))
        perm[b]=a1-a2
    p_one=float(np.mean(perm<=meas))
    return dict(kcut=kcut,meas=meas,pmean=perm.mean(),pstd=perm.std(ddof=1),
                p_perm=p_one)

if __name__=="__main__" and "--perm" in sys.argv:
    d=pd.read_csv(os.path.join(_ROOT,"resources/census/ecc_marginal_real.csv"))
    if "P50" in d: d=d[(d.P50>=6)&(d.P50<=400)]
    print("\nPERMUTATION NULL (relabel twins):")
    print("%-5s %-8s %-9s %-8s %-8s"%("kcut","dAlpha","perm_mean","perm_sd","p<=meas"))
    for kc in [8,12,16]:
        r=permutation(d,kc)
        print("%-5.0f %+8.3f %+9.3f %8.3f %8.4f"%
              (r["kcut"],r["meas"],r["pmean"],r["pstd"],r["p_perm"]))
