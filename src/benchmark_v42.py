"""Fresh independent seed benchmark for V0.42; frozen V0.39 candidate paths."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import beta
from v42 import Source,route_gate
from v41 import evidence_gate
from benchmark_v39 import make,H,G
from v39 import simulate
from v35 import C
from v38 import EPS,PROBE_COST
OUT=Path(__file__).resolve().parents[1]/'results'/'v42'
STRESSES=['clean','variance4','matched','shared_domain_fault','two_domain_fault','bounded_common','serial','hidden_common']
METHODS=['v41_shared','v41_diverse','round_robin','adaptive','adaptive_f2','adaptive_bound','unknown_decline']
SOURCES=[Source('A',0),Source('A',0),Source('B',0),Source('C',0)]
def generate(loss,sd,rng,stress):
    n=len(loss); shape=(n,4,9); e=rng.normal(size=shape); c=rng.normal(size=shape)
    if stress=='variance4':e*=2;c*=2
    if stress=='serial':
        for k in range(1,9):e[:,:,k]=.8*e[:,:,k-1]+.6*e[:,:,k];c[:,:,k]=.8*c[:,:,k-1]+.6*c[:,:,k]
    x=loss[:,None,None]+sd[:,None,None]*e; y=sd[:,None,None]*c
    if stress=='matched':x-=1.5*sd[:,None,None];y-=1.5*sd[:,None,None]
    if stress in ('shared_domain_fault','two_domain_fault'):x[:,:2,:]-=3*sd[:,None,None]
    if stress=='two_domain_fault':x[:,2,:]-=3*sd[:,None]
    if stress=='bounded_common':x-=.06
    if stress=='hidden_common':x-=3*sd[:,None,None]
    return x,y

def gate(name,x,c,sd,g):
    if name.startswith('v41'):
        ids=[0,1,2] if name=='v41_shared' else [0,2,3]
        return evidence_gate(x[:,ids,:],c[:,ids,:],sd,g,mode='robust',paired=True,looks=(3,4,6),epsilon=EPS,unit_cost=PROBE_COST)
    sources=[Source(s.domain,.06,s.cost) for s in SOURCES] if name=='adaptive_bound' else SOURCES
    return route_gate(x,c,g,sources,fault_budget=None if name=='unknown_decline' else (2 if name=='adaptive_f2' else 1),adaptive=name!='round_robin',epsilon=EPS,unit_cost=PROBE_COST)

def boundary(n):
    rows=[]
    for i,stress in enumerate(STRESSES):
        for j,delta in enumerate([1e-8,-.75,-1.5]):
            rng=np.random.default_rng(42500000+i*1000+j);sd=np.full(n,.12);loss=EPS+delta*sd;g=np.ones(n,int);x,c=generate(loss,sd,rng,stress)
            for name in METHODS:
                a=gate(name,x,c,sd,g);k=int(a['accept'].sum());lo=beta.ppf(.025,k,n-k+1) if k else 0;hi=beta.ppf(.975,k+1,n-k) if k<n else 1
                rows.append(dict(stress=stress,delta_sd=delta,method=name,n=n,accepted=k,rate=k/n,lo=lo,hi=hi,samples=a['samples'].mean()))
        print('boundary',stress,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'boundary.csv',index=False)

DYNAMIC=[('irregular_recurrence','clean'),('random_walk','clean'),('mean_reversion','clean'),('mean_reversion','shared_domain_fault'),('mean_reversion','two_domain_fault'),('mean_reversion','bounded_common'),('mean_reversion','hidden_common'),('mean_reversion','serial')]
def dynamic(n):
    rows=[]
    for i,(scene,stress) in enumerate(DYNAMIC):
        seed=42600000+i*10000;truth,y,v,bank,r,*_=make(scene,n,seed);b=simulate(y,v,bank,r,threshold=.95,seed=seed+123)
        net=truth[:,12:]-C;oracle=np.maximum(net,0).mean(2);ca=b['candidate'][:,12:];ref=b['reference'][:,12:];cv=(ca*net).mean(2);rv=(ref*net).mean(2);loss=rv-cv;groups=(ca!=ref).sum(2);eligible=(b['model_accept'][:,12:]>0)&(groups>0);idx=np.nonzero(eligible);g=groups[idx];sd=np.sqrt(.45*g)/G
        x,c=generate(loss[idx],sd,np.random.default_rng(seed+99900000),stress)
        for name in ['recompute','v38',*METHODS]:
            selected=np.zeros_like(eligible);cost=np.zeros_like(loss);samples=np.zeros_like(loss)
            if name=='v38':selected=eligible.copy()
            elif name!='recompute':
                a=gate(name,x,c,sd,g);selected[idx]=a['accept'];cost[idx]=a['cost'];samples[idx]=a['samples']
            unsafe=loss>EPS
            rows.append(pd.DataFrame(dict(scene=scene,stress=stress,seed=np.arange(seed,seed+n),method=name,periods=H-12,regret=(oracle-np.where(selected,cv,rv)).sum(1),probe_cost=b['probes'][:,12:].sum(1)*PROBE_COST,audit_cost=cost.sum(1),samples=samples.sum(1),accepted=selected.sum(1),unsafe=(selected&unsafe).sum(1),unsafe_eligible=(eligible&unsafe).sum(1),eligible=eligible.sum(1))))
        print('dynamic',scene,stress,n,'eligible',len(g),flush=True)
    d=pd.concat(rows,ignore_index=True);d.to_csv(OUT/'trajectories.csv',index=False)
    def metrics(g):
        s=g.sum(numeric_only=True);return dict(trajectories=len(g),net_loss=(s.regret+s.probe_cost+s.audit_cost)/s.periods,regret=s.regret/s.periods,audit_cost=s.audit_cost/s.periods,accepted=int(s.accepted),unsafe=int(s.unsafe),eligible=int(s.eligible),unsafe_eligible=int(s.unsafe_eligible),false_accept=s.unsafe/s.unsafe_eligible if s.unsafe_eligible else np.nan,unsafe_given_accept=s.unsafe/s.accepted if s.accepted else np.nan,samples=int(s.samples))
    pd.DataFrame([dict(scene=k[0],stress=k[1],method=k[2],**metrics(g)) for k,g in d.groupby(['scene','stress','method'])]).to_csv(OUT/'scenes.csv',index=False)
    pd.DataFrame([dict(stress=k[0],method=k[1],**metrics(g)) for k,g in d.groupby(['stress','method'])]).to_csv(OUT/'summary.csv',index=False)
    rng=np.random.default_rng(4261);out=[]
    for (scene,stress),f in d.groupby(['scene','stress']):
        for name in METHODS:
            a=f[f.method==name].sort_values('seed');b=f[f.method=='v41_diverse'].sort_values('seed');assert np.array_equal(a.seed,b.seed)
            z=((a.regret+a.audit_cost+a.probe_cost).to_numpy()-(b.regret+b.audit_cost+b.probe_cost).to_numpy())/(H-12)
            boot=z[rng.integers(n,size=(1500,n))].mean(1);out.append(dict(scene=scene,stress=stress,method=name,baseline='v41_diverse',delta=z.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975)))
    pd.DataFrame(out).to_csv(OUT/'paired.csv',index=False)

if __name__=='__main__':
    import sys
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'config.json').write_text(json.dumps(dict(boundary_n=20000,dynamic_n=60,stresses=STRESSES,dynamic=DYNAMIC,methods=METHODS,seed_boundary=42500000,seed_dynamic=42600000,alpha=.05,epsilon=EPS,cost=PROBE_COST,looks=[3,6,9],budget=36),indent=2))
    if len(sys.argv)==1 or sys.argv[1]=='boundary':boundary(20000)
    if len(sys.argv)==1 or sys.argv[1]=='dynamic':dynamic(60)
