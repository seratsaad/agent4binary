#!/usr/bin/env python3
"""Full-size injection null for the twin/non-twin eccentricity bias.

For each realization we inject the SAME true index alpha into two mock
populations: one whose cadence/K1/epoch-count are inherited from the real
K1>12 twins, one from the real K1>12 non-twins. We refit every mock L(e) and
score the matched dAlpha with the identical estimator the measurement uses.
Any nonzero mean is the residual selection bias that binned matching leaves.
Injects at alpha in {0.1,0.3} to check the bias is not alpha-dependent.
Parallel, incremental CSV, resume-safe.

--method twovel (default): each mock is built like a real untied pair and fitted
    with the two-velocity likelihood (ecc_marginal.system_loglike_on_egrid_twovel),
    as ecc_real_marginal.py does for the data: the real system's usable-pair epochs,
    K1, P50 and q (deep_table.q_twovel); m1 = gamma + K1 S, m2 = gamma - (K1/q) S,
    Gaussian noise SIGMA on each, then the pair is exchanged at each visit with
    probability 0.5.
--method v1: the previous primary-velocity-only mocks and likelihood.
The method must match the one ecc_marginal_real.csv was made with (its method
column; a file without one is taken as v1).

Each system's cadence is the mjd_per_visit list of its deep-table row
(A4B_DEEP_TABLE overrides resources/census/stage2_deep.csv).

The output is appended to and resumed. A sidecar <out>.inputs.json records the
deep table and ecc_marginal_real.csv it was built from (row count and md5) and
the run settings (method included); the script refuses to append to an output
whose sidecar is missing or differs. Pass --fresh to start the output over.

  python scripts/ecc_null_full.py [--out FILE] [--real FILE] [--fresh] [--method twovel|v1]
"""
import os, sys, time, json, hashlib, argparse
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
_HERE=os.path.dirname(os.path.abspath(__file__)); _ROOT=os.path.dirname(_HERE)
sys.path.insert(0,_HERE); sys.path.insert(0,os.path.join(_ROOT,"src"))
import orbit_sampler as OS, ecc_marginal as EM
import deep_table as DT

EG=EM.default_egrid(); SIGMA=1.5; DRAWS=30000
NREPS=12; ALPHAS=[0.1,0.3]
KCUT=12.0
R=lambda p: p if os.path.isabs(p) else os.path.join(_ROOT,p)

def _fingerprint(path):
    raw=open(path,"rb").read()
    return dict(path=os.path.abspath(path), rows=max(raw.count(b"\n")-1,0),
                md5=hashlib.md5(raw).hexdigest())

def _inputs(deep_path, real_path, method):
    d=dict(deep_table=_fingerprint(deep_path), real_table=_fingerprint(real_path),
           nreps=NREPS, alphas=ALPHAS, sigma=SIGMA, draws=DRAWS, kcut=KCUT)
    if method!="v1": d["method"]=method      # a v1 sidecar keeps its earlier form
    return d

def _same(a, b):
    """Inputs match if the file contents (rows, md5) and settings agree; the
    absolute paths are informational (the same file may sit elsewhere)."""
    strip=lambda d: {k:({kk:vv for kk,vv in v.items() if kk!="path"} if isinstance(v,dict) else v)
                     for k,v in d.items()}
    return strip(a)==strip(b)

def build_jobs(real, T, Q=None, method="v1"):
    jobs=[]
    for al in ALPHAS:
        for rep in range(NREPS):
            for _,r in real.iterrows():
                sid=int(r.sdss_id)
                t=T[sid][:int(r.n_ep)]
                if len(t)<8: continue
                gid=hash((al,rep,sid))&0x7fffffff
                job=(gid,al,rep,bool(r.twin),sid,float(r.K1),len(t),
                     float(r.P50),";".join("%.5f"%x for x in t))
                if method=="twovel": job=job+(float(Q[sid]),"twovel")
                jobs.append(job)
    return jobs

def worker(job):
    if len(job)>9 and job[10]=="twovel":
        return worker_twovel(job)
    gid,al,rep,twin,sid,K1,n_ep,P,tstr=job[:9]
    rng=np.random.default_rng(gid)
    t=np.array([float(x) for x in tstr.split(";")])
    u=rng.uniform(); e=(u*EM.EMAX**(1+al))**(1.0/(1+al))
    om=rng.uniform(0,2*np.pi); M0=rng.uniform(0,2*np.pi); vsys=rng.uniform(-30,30)
    S=OS.rv_shape(t,np.array([P]),np.array([e]),np.array([om]),np.array([M0]))[0]
    v=vsys+K1*S+rng.normal(0,SIGMA,len(t))
    lg=EM.system_loglike_on_egrid(t-t.mean(),v,SIGMA,EG,M_per_e=DRAWS,rng=rng)
    return (al,rep,int(twin),sid,K1,n_ep,";".join("%.4f"%x for x in lg))

def worker_twovel(job):
    gid,al,rep,twin,sid,K1,n_ep,P,tstr,q,_=job
    rng=np.random.default_rng(gid)
    t=np.array([float(x) for x in tstr.split(";")])
    u=rng.uniform(); e=(u*EM.EMAX**(1+al))**(1.0/(1+al))
    om=rng.uniform(0,2*np.pi); M0=rng.uniform(0,2*np.pi); vsys=rng.uniform(-30,30)
    S=OS.rv_shape(t,np.array([P]),np.array([e]),np.array([om]),np.array([M0]))[0]
    m1=vsys+K1*S+rng.normal(0,SIGMA,len(t))
    m2=vsys-(K1/q)*S+rng.normal(0,SIGMA,len(t))
    pa,pb=DT.exchange_pairs(m1,m2,rng)
    lg=EM.system_loglike_on_egrid_twovel(t-t.mean(),pa,pb,q,SIGMA,EG,M_per_e=DRAWS,rng=rng)
    return (al,rep,int(twin),sid,K1,n_ep,";".join("%.4f"%x for x in lg))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="resources/census/ecc_null_full.csv")
    ap.add_argument("--real",default="resources/census/ecc_marginal_real.csv")
    ap.add_argument("--fresh",action="store_true",help="discard an existing --out and start over")
    ap.add_argument("--method",choices=["twovel","v1"],default="twovel")
    a=ap.parse_args()
    OUT=R(a.out); SIDE=OUT+".inputs.json"; REAL=R(a.real)

    real=pd.read_csv(REAL)
    real_method=sorted(set(real["method"].dropna())) if "method" in real.columns else ["v1"]
    if real_method!=[a.method]:
        sys.exit("%s was made with method %s; this run uses --method %s. The null must use "
                 "the same likelihood as the measurement."%(REAL,",".join(real_method),a.method))
    real=real[(real.P50>=6)&(real.P50<=400)&(real.K1>KCUT)].reset_index(drop=True)
    deep=DT.load_deep()
    Q=None
    if a.method=="twovel":
        # usable-pair epochs (visits with a missing pair dropped, as in ecc_real_marginal)
        T={}; Q={}
        for r in deep.itertuples():
            t,_,_=DT.untied_pairs(r)
            if len(t): T[int(r.sdss_id)]=t; Q[int(r.sdss_id)]=DT.q_twovel(r)
    else:
        T=DT.epochs_by_id(deep)
    miss=set(int(s) for s in real.sdss_id)-set(T)
    if miss:
        sys.exit("%d systems of %s are not in the deep table %s; the two are from different runs"
                 %(len(miss),REAL,DT.deep_path()))
    inputs=_inputs(DT.deep_path(),REAL,a.method)

    if a.fresh:
        for p in (OUT,SIDE):
            if os.path.exists(p): os.remove(p)
    if os.path.exists(OUT):
        old=json.load(open(SIDE)) if os.path.exists(SIDE) else None
        if old is None or not _same(old,inputs):
            sys.exit("refusing to append to %s: %s. Rerun with --fresh, or pick another --out."
                     %(OUT,"no sidecar %s"%SIDE if old is None else
                       "it was built from different inputs (see %s)"%SIDE))
    else:
        json.dump(inputs,open(SIDE,"w"),indent=1)

    done=set()
    if os.path.exists(OUT):
        prev=pd.read_csv(OUT)
        done={("%.2f"%r.alpha,int(r.rep),int(r.sdss_id)) for r in prev.itertuples()}
        print("resuming, %d rows already done"%len(prev),flush=True)
    jobs=build_jobs(real,T,Q,a.method)
    jobs=[j for j in jobs if ("%.2f"%j[1],j[2],j[4]) not in done]
    print("jobs to run: %d (%d reps x %d alphas, method %s)"%(len(jobs),NREPS,len(ALPHAS),a.method),flush=True)
    hdr=not os.path.exists(OUT); t0=time.time(); n=0
    nw=int(os.environ.get("WORKERS","8"))
    print("workers: %d"%nw,flush=True)
    with ProcessPoolExecutor(max_workers=nw) as ex, open(OUT,"a") as f:
        if hdr: f.write("alpha,rep,twin,sdss_id,K1,n_ep,loglike\n")
        for res in ex.map(worker,jobs,chunksize=4):
            al,rep,tw,sid,K1,n_ep,lgs=res
            f.write("%.2f,%d,%d,%d,%.4f,%d,%s\n"%(al,rep,tw,sid,K1,n_ep,lgs)); n+=1
            if n%200==0:
                f.flush(); dt=time.time()-t0
                print("  %d/%d  %.1f min  %.2f s/fit"%(n,len(jobs),dt/60,dt/n),flush=True)
    print("DONE %d fits in %.1f min -> %s"%(n,(time.time()-t0)/60,OUT),flush=True)

if __name__=="__main__": main()
