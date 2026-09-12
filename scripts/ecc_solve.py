#!/usr/bin/env python3
"""Final verdict on the close-twin eccentricity difference.

  measured dAlpha (K1>12, matched)      <- ecc_marginal_real.csv
  statistical error (bootstrap)         <- resample systems, precomputed L(e)
  permutation p (twin label carries it) <- relabel twins
  selection bias (injection null)       <- ecc_null_full.csv, same alpha both pops
  corrected dAlpha and significance     <- measured - bias, errors combined
"""
import os, sys
import numpy as np, pandas as pd
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(_HERE)
sys.path.insert(0,_HERE)
import ecc_bootstrap as B   # reuse marginal_matrix, alpha_from_M, matched_w, parse

KCUT=12.0
def measured_and_boot(nboot=4000):
    d=pd.read_csv(os.path.join(_ROOT,"resources/census/ecc_marginal_real.csv"))
    d=d[(d.P50>=6)&(d.P50<=400)]
    r=B.run(d,KCUT,nboot=nboot)
    pr=B.permutation(d,KCUT,nperm=6000)
    return r,pr

NULL_DEFAULT=os.path.join(_ROOT,"resources/census/ecc_null_full.csv")

def injection_das(f=None):
    """Per-realization twin minus non-twin dAlpha of the injection null
    (ecc_null_full.csv): DataFrame alpha, rep, da, n_t, n_n. None if no file."""
    f=f or NULL_DEFAULT
    if not os.path.exists(f): return None
    n=pd.read_csv(f)
    key=["alpha","rep","sdss_id"] if "sdss_id" in n.columns else ["alpha","rep","twin","K1","n_ep"]
    ndup=int(n.duplicated(key).sum())
    if ndup:
        print("WARNING: %s has %d rows repeated on %s (kept, as before)"%(f,ndup,key))
    rows=[]
    for al,ga in n.groupby("alpha"):
        for rep,g in ga.groupby("rep"):
            tw=g[g.twin==1]; nt=g[g.twin==0]
            if len(tw)<25 or len(nt)<25: continue
            LGt=np.array([B.parse(x) for x in tw.loglike]); LGt[~np.isfinite(LGt)]=-1e9
            LGn=np.array([B.parse(x) for x in nt.loglike]); LGn[~np.isfinite(LGn)]=-1e9
            Mt,Mn=B.marginal_matrix(LGt),B.marginal_matrix(LGn)
            at=B.alpha_from_M(Mt,np.ones(len(tw)))
            an=B.alpha_from_M(Mn,B.matched_w(tw.K1.values,tw.n_ep.values,
                                             nt.K1.values,nt.n_ep.values))
            rows.append(dict(alpha=al,rep=rep,da=at-an,n_t=len(tw),n_n=len(nt)))
    return pd.DataFrame(rows,columns=["alpha","rep","da","n_t","n_n"])

def bias_from_das(das):
    """{alpha: (mean dAlpha, sd over realizations, n realizations)}."""
    out={}
    for al,g in das.groupby("alpha"):
        x=g.da.values
        if len(x): out[al]=(x.mean(),x.std(ddof=1) if len(x)>1 else np.nan,len(x))
    return out

def injection_bias(f=None):
    das=injection_das(f)
    return None if das is None else bias_from_das(das)

def pooled_bias(bias):
    """Pool the per-alpha biases, weighted by the number of realizations; the
    error is the mean per-alpha spread over sqrt(total realizations)."""
    pooled=np.array([m for m,_,_ in bias.values()])
    wn=np.array([nn for _,_,nn in bias.values()])
    bmean=np.average(pooled,weights=wn)
    allreps=sum(nn for _,_,nn in bias.values())
    sds=[s for _,s,_ in bias.values() if np.isfinite(s)]
    bse=(np.mean(sds)/np.sqrt(allreps)) if sds else np.nan
    return bmean,bse

if __name__=="__main__":
    print("="*64)
    r,pr=measured_and_boot()
    print("MEASURED (K1>12, matched):")
    print("  dAlpha = %+.3f   (twin %+.3f, n=%d | non-twin %+.3f, n=%d)"
          %(r["meas"],r["at"],r["n_tw"],r["an"],r["n_nt"]))
    print("  bootstrap error   = %.3f   one-sided p(dAlpha>=0) = %.3f"%(r["bstd"],r["p_pos"]))
    print("  permutation: mean %+.3f sd %.3f  p(perm<=meas) = %.4f"
          %(pr["pmean"],pr["pstd"],pr["p_perm"]))
    bias=injection_bias()
    if bias:
        print("\nINJECTION NULL (same true alpha both pops -> selection bias):")
        alld=[]
        for al,(m,s,nn) in sorted(bias.items()):
            print("  alpha=%.2f : bias = %+.3f +/- %.3f  (n=%d reps)"%(al,m,s,nn))
        # pool across alphas for the bias point estimate; SE from per-alpha spread
        bmean,bse=pooled_bias(bias)
        print("  pooled bias = %+.3f +/- %.3f"%(bmean,bse))
        corr=r["meas"]-bmean
        tot=np.sqrt(r["bstd"]**2+ (bse**2 if np.isfinite(bse) else 0))
        print("\nVERDICT:")
        print("  corrected dAlpha = %+.3f +/- %.3f (stat %.3f (+) sys %.3f)"
              %(corr,tot,r["bstd"],bse))
        print("  significance = %.1f sigma  (twins %s eccentric)"
              %(abs(corr)/tot, "less" if corr<0 else "more"))
    print("="*64)
