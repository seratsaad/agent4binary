#!/usr/bin/env python3
"""Final eccentricity numbers on a consistent footing.
FIX: population density normalized on [0,EMAX] as (1+a)e^a / EMAX^(1+a).
The code previously used (1+a)e^a while integrating to EMAX=0.95, omitting an
a-dependent factor. Also: bootstrap errors for EVERY variant, injection-residual
correction for every variant, N/ESS per variant, and <e> on the correct range.
"""
import os, sys
import numpy as np, pandas as pd
from collections import Counter
_H=os.path.dirname(os.path.abspath(__file__)); _R=os.path.dirname(_H)
sys.path.insert(0,_H); import ecc_bootstrap as B, ecc_marginal as EM

EMAX=EM.EMAX
A=B.A
# correct log-normalizer: log(1+a) - (1+a) log(EMAX)
LOGNORM = np.log1p(A) - (1.0+A)*np.log(EMAX)

def alpha_from_M(M,w):
    LL=(w[:,None]*M).sum(axis=0)+w.sum()*LOGNORM
    return A[int(np.argmax(LL))]

def mean_e(a):  return EMAX*(1.0+a)/(2.0+a)

EDG={'K1':np.array([0,5,8,12,18,25,40,200]),'n_ep':np.array([8,10,12,14,18,30]),
     'logP':np.array([np.log10(6),1,1.3,1.6,1.9,np.log10(400)]),
     'baseline':np.array([0,100,400,900,1600,4000]),
     'dvmax':np.array([0,20,35,50,70,300])}
def cells(df,cols):
    ks=[]
    for c in cols:
        v=np.log10(df['P50']) if c=='logP' else df[c]
        ks.append(np.digitize(np.nan_to_num(np.asarray(v,float),nan=-1),EDG[c]))
    return list(zip(*ks))
def w_of(tw,nt,cols):
    tc=Counter(cells(tw,cols)); nc=Counter(cells(nt,cols)); nn=cells(nt,cols)
    return np.clip(np.array([(tc[c]/len(tw))/max(nc[c]/len(nt),1e-9) for c in nn]),0,20)

def variant(sub,cols,label,nboot=1500,seed=5):
    tw,nt=sub[sub.twin],sub[~sub.twin]
    LGt=np.array([B.parse(x) for x in tw.loglike]); LGt[~np.isfinite(LGt)]=-1e9
    LGn=np.array([B.parse(x) for x in nt.loglike]); LGn[~np.isfinite(LGn)]=-1e9
    Mt,Mn=B.marginal_matrix(LGt),B.marginal_matrix(LGn)
    twd=tw.reset_index(drop=True); ntd=nt.reset_index(drop=True)
    w=w_of(twd,ntd,cols)
    at=alpha_from_M(Mt,np.ones(len(twd))); an=alpha_from_M(Mn,w)
    ess=w.sum()**2/np.sum(w**2)
    rng=np.random.default_rng(seed); bs=np.empty(nboot)
    for b in range(nboot):
        it=rng.integers(0,len(twd),len(twd)); jn=rng.integers(0,len(ntd),len(ntd))
        wb=w_of(twd.iloc[it],ntd.iloc[jn],cols)
        bs[b]=alpha_from_M(Mt[it],np.ones(len(it)))-alpha_from_M(Mn[jn],wb)
    return dict(label=label,a_t=at,a_n=an,da=at-an,boot=bs.std(ddof=1),
                n_t=len(twd),n_n=len(ntd),ess=ess,
                e_t=mean_e(at),e_n=mean_e(an))

if __name__=="__main__":
    d=pd.read_csv(os.path.join(_R,'resources/census/ecc_marginal_real.csv'))
    d=d[(d.P50>=6)&(d.P50<=400)].copy()
    # baseline = span of the fitted epochs (mjd_per_visit of the deep-table row;
    # A4B_DEEP_TABLE overrides the default path)
    import deep_table as DT
    deep=DT.load_deep().set_index('sdss_id')
    pr=DT.parse
    _miss=set(d.sdss_id)-set(deep.index)
    assert not _miss, '%d systems of ecc_marginal_real.csv are not in the deep table'%len(_miss)
    # dvmax: v1_per_visit range for a v1-method table (as before); for a twovel
    # table the largest untied pair separation max|a_i-b_i| (deep_table.observed_dv)
    meth=DT.table_method(d)
    bl,dv=[],[]
    for sid in d.sdss_id:
        t=pr(deep.loc[sid,'mjd_per_visit']); bl.append(t.max()-t.min())
        dv.append(DT.observed_dv(deep.loc[sid],meth))
    d['baseline']=bl; d['dvmax']=dv
    cat=pd.read_csv(os.path.join(_R,'resources/census/dr19_sb2_catalog_full.csv'))[['sdss_id','prefers_binary']]
    d=d.merge(cat,on='sdss_id',how='left')
    m=d[d.K1>12]; mo=d[d.dvmax>25]
    V=[variant(m,['K1','n_ep'],'K1 + epochs (fiducial)'),
       variant(m,['logP','n_ep'],'period + epochs'),
       variant(m,['logP','n_ep','K1'],'period + epochs + K1'),
       variant(d[(d.K1>12)&(d.P50>15)],['logP','n_ep'],'period + epochs, P>15d'),
       variant(mo,['n_ep','baseline','dvmax'],'observed only (no fitted vars)'),
       variant(m[m.prefers_binary==True],['K1','n_ep'],'confident detections only')]
    print('%-34s %6s %6s %7s %7s %5s %5s %6s   <e>t  <e>n'%(
        'variant','a_twin','a_nont','dAlpha','boot','N_t','N_n','ESS'))
    for v in V:
        print('%-34s %+6.3f %+6.3f %+7.3f %7.3f %5d %5d %6.0f   %.2f  %.2f'%(
            v['label'],v['a_t'],v['a_n'],v['da'],v['boot'],v['n_t'],v['n_n'],v['ess'],
            v['e_t'],v['e_n']))
    da=np.array([v['da'] for v in V[:5]])
    print('\nmatching-scheme spread (first 5): mean %+.3f  sd %.3f  range %.3f'%(
        da.mean(),da.std(ddof=1),da.max()-da.min()))
    vpath=os.path.join(_R,'resources/census/ecc_variants.csv')
    pd.DataFrame(V).to_csv(vpath,index=False)
    # headline numbers (fiducial - null bias, combined error) -> ecc_headline.json
    import ecc_headline
    print()
    ecc_headline.main(['--variants',vpath])
