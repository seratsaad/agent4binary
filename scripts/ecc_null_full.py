#!/usr/bin/env python3
"""Full-size injection null for the twin/non-twin eccentricity bias.

For each realization we inject the SAME true index alpha into two mock
populations: one whose cadence/K1/epoch-count are inherited from the real
K1>12 twins, one from the real K1>12 non-twins. We refit every mock L(e) and
score the matched dAlpha with the identical estimator the measurement uses.
Any nonzero mean is the residual selection bias that binned matching leaves.
Injects at alpha in {0.1,0.3} to check the bias is not alpha-dependent.
Parallel, incremental CSV, resume-safe."""
import os, sys, time
import numpy as np, pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(_HERE)
sys.path.insert(0,_HERE); sys.path.insert(0,os.path.join(_ROOT,"src"))
import orbit_sampler as OS, ecc_marginal as EM

EG=EM.default_egrid(); SIGMA=1.5; DRAWS=30000
OUT=os.path.join(_ROOT,"resources/census/ecc_null_full.csv")
NREPS=12; ALPHAS=[0.1,0.3]
def parse(s): return np.array([float(x) for x in str(s).split(";") if x not in ("","nan")])

real=pd.read_csv(os.path.join(_ROOT,"resources/census/ecc_marginal_real.csv"))
real=real[(real.P50>=6)&(real.P50<=400)&(real.K1>12)].reset_index(drop=True)
mj=pd.read_csv(os.path.join(_ROOT,"resources/census/stage2_mjds.csv")).set_index("sdss_id")

def build_jobs():
    jobs=[]
    for al in ALPHAS:
        for rep in range(NREPS):
            for _,r in real.iterrows():
                sid=int(r.sdss_id)
                if sid not in mj.index: continue
                t=parse(mj.loc[sid,"mjd_per_visit"])[:int(r.n_ep)]
                if len(t)<8: continue
                gid=hash((al,rep,sid))&0x7fffffff
                jobs.append((gid,al,rep,bool(r.twin),float(r.K1),len(t),
                             float(r.P50),";".join("%.5f"%x for x in t)))
    return jobs

def worker(job):
    gid,al,rep,twin,K1,n_ep,P,tstr=job
    rng=np.random.default_rng(gid)
    t=np.array([float(x) for x in tstr.split(";")])
    u=rng.uniform(); e=(u*EM.EMAX**(1+al))**(1.0/(1+al))
    om=rng.uniform(0,2*np.pi); M0=rng.uniform(0,2*np.pi); vsys=rng.uniform(-30,30)
    S=OS.rv_shape(t,np.array([P]),np.array([e]),np.array([om]),np.array([M0]))[0]
    v=vsys+K1*S+rng.normal(0,SIGMA,len(t))
    lg=EM.system_loglike_on_egrid(t-t.mean(),v,SIGMA,EG,M_per_e=DRAWS,rng=rng)
    return (al,rep,int(twin),K1,n_ep,";".join("%.4f"%x for x in lg))

def main():
    done=set()
    if os.path.exists(OUT):
        prev=pd.read_csv(OUT)
        done={(r.alpha,r.rep,r.twin,round(r.K1,3),r.n_ep) for r in prev.itertuples()}
        print("resuming, %d rows already done"%len(prev),flush=True)
    jobs=build_jobs()
    jobs=[j for j in jobs if (j[1],j[2],int(j[3]),round(j[4],3),j[5]) not in done]
    print("jobs to run: %d (%d reps x %d alphas)"%(len(jobs),NREPS,len(ALPHAS)),flush=True)
    hdr=not os.path.exists(OUT); t0=time.time(); n=0
    nw=int(os.environ.get("WORKERS","8"))
    print("workers: %d"%nw,flush=True)
    with ProcessPoolExecutor(max_workers=nw) as ex, open(OUT,"a") as f:
        if hdr: f.write("alpha,rep,twin,K1,n_ep,loglike\n")
        for res in ex.map(worker,jobs,chunksize=4):
            al,rep,tw,K1,n_ep,lgs=res
            f.write("%.2f,%d,%d,%.4f,%d,%s\n"%(al,rep,tw,K1,n_ep,lgs)); n+=1
            if n%200==0:
                f.flush(); dt=time.time()-t0
                print("  %d/%d  %.1f min  %.2f s/fit"%(n,len(jobs),dt/60,dt/n),flush=True)
    print("DONE %d fits in %.1f min"%(n,(time.time()-t0)/60),flush=True)

if __name__=="__main__": main()
