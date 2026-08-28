#!/usr/bin/env python3
"""Robustness of the twin/non-twin eccentricity difference.
(1) spread across matching schemes -> a systematic we had not budgeted
(2) matching on OBSERVED quantities only (no fitted P50/K1) -> removes circularity
(3) purity-matched subsample (both classes restricted to confident detections)
(4) MAP vs posterior-mean alpha, quoted consistently
"""
import os, sys
import numpy as np, pandas as pd
from collections import Counter
_H=os.path.dirname(os.path.abspath(__file__)); _R=os.path.dirname(_H)
sys.path.insert(0,_H); import ecc_bootstrap as B

d=pd.read_csv(os.path.join(_R,'resources/census/ecc_marginal_real.csv'))
d=d[(d.P50>=6)&(d.P50<=400)].copy()
# observed quantities: epoch count, time baseline, measured velocity range
mj=pd.read_csv(os.path.join(_R,'resources/census/stage2_mjds.csv')).set_index('sdss_id')
deep=pd.read_csv(os.path.join(_R,'resources/census/stage2_deep.csv')).set_index('sdss_id')
def parse(s):
    return np.array([float(x) for x in str(s).split(';') if x not in ('','nan')])
base,dvmax=[],[]
for sid in d.sdss_id:
    try:
        t=parse(mj.loc[sid,'mjd_per_visit']); base.append(t.max()-t.min())
    except Exception: base.append(np.nan)
    try:
        v=parse(deep.loc[sid,'v1_per_visit']); dvmax.append(v.max()-v.min())
    except Exception: dvmax.append(np.nan)
d['baseline']=base; d['dvmax']=dvmax
# catalog confidence
cat=pd.read_csv(os.path.join(_R,'resources/census/dr19_sb2_catalog_full.csv'))[['sdss_id','prefers_binary','delta_chi2']]
d=d.merge(cat,on='sdss_id',how='left')

EDG={'K1':np.array([0,5,8,12,18,25,40,200]),'n_ep':np.array([8,10,12,14,18,30]),
     'logP':np.array([np.log10(6),1,1.3,1.6,1.9,np.log10(400)]),
     'baseline':np.array([0,100,400,900,1600,4000]),
     'dvmax':np.array([0,20,35,50,70,300])}
def w_of(tw,nt,cols):
    def cell(df):
        ks=[]
        for c in cols:
            v=np.log10(df.P50) if c=='logP' else df[c]
            ks.append(np.digitize(np.nan_to_num(v,nan=-1),EDG[c]))
        return list(zip(*ks))
    tc=Counter(cell(tw)); nc=Counter(cell(nt)); nn=cell(nt)
    return np.clip(np.array([(tc[c]/len(tw))/max(nc[c]/len(nt),1e-9) for c in nn]),0,20)

def alphas(M,w):
    LL=(w[:,None]*M).sum(axis=0)+w.sum()*B.L1A; LL-=LL.max()
    P=np.exp(LL); dA=B.A[1]-B.A[0]; P/=P.sum()*dA
    mean=(B.A*P).sum()*dA; sd=(((B.A-mean)**2*P).sum()*dA)**0.5
    return B.A[int(np.argmax(LL))], mean, sd

def run(sub,cols,label):
    tw,nt=sub[sub.twin],sub[~sub.twin]
    if len(tw)<25 or len(nt)<25: print('%-42s too few'%label); return None
    LGt=np.array([B.parse(x) for x in tw.loglike]); LGt[~np.isfinite(LGt)]=-1e9
    LGn=np.array([B.parse(x) for x in nt.loglike]); LGn[~np.isfinite(LGn)]=-1e9
    Mt,Mn=B.marginal_matrix(LGt),B.marginal_matrix(LGn)
    w=w_of(tw,nt,cols)
    mt,mnt,st=alphas(Mt,np.ones(len(tw))); mn,mnn,sn=alphas(Mn,w)
    print('%-42s MAP dA=%+.3f | mean dA=%+.3f +/- %.3f  (n=%d/%d)'%(
        label,mt-mn,mnt-mnn,(st**2+sn**2)**0.5,len(tw),len(nt)))
    return mnt-mnn

m=d[d.K1>12]
print('=== (1) matching-scheme spread ===')
vals=[]
for cols,lab in [(['K1','n_ep'],'K1 + epochs'),(['logP','n_ep'],'period + epochs'),
                 (['logP','n_ep','K1'],'period + epochs + K1'),
                 (['n_ep'],'epochs only')]:
    v=run(m,cols,lab)
    if v is not None: vals.append(v)
vals=np.array(vals)
print('   spread across schemes: mean %+.3f, sd %.3f, range %.3f'%(vals.mean(),vals.std(ddof=1),vals.max()-vals.min()))

print('=== (2) OBSERVED-ONLY matching (no fitted P50/K1) ===')
mo=d[d.dvmax>25]   # observational analogue of the K1 cut
run(mo,['n_ep','baseline'],'epochs + baseline (obs)')
run(mo,['n_ep','dvmax'],'epochs + dvmax (obs)')
run(mo,['n_ep','baseline','dvmax'],'epochs + baseline + dvmax (obs)')

print('=== (3) purity-matched (confident detections only) ===')
mp=m[m.prefers_binary==True]
print('   twins %d, non-twins %d'%((mp.twin).sum(),(~mp.twin).sum()))
run(mp,['K1','n_ep'],'confident only, K1 + epochs')
run(mp,['logP','n_ep'],'confident only, period + epochs')
hi=m[m.delta_chi2>m.delta_chi2.median()]
run(hi,['K1','n_ep'],'high dchi2 half, K1 + epochs')
