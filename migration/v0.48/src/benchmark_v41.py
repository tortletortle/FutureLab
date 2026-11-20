"""Independent controlled boundary and frozen-policy dynamic audit benchmarks."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import beta
from benchmark_v39 import make,H,G
from v39 import simulate
from v35 import C
from v38 import EPS,PROBE_COST
from v41 import evidence_gate
OUT=Path(__file__).resolve().parents[1]/'results'/'v41'
ARMS={
 'known3':('known',False,(1,2,3)),
 'single_t6':('single',False,(3,4,6)),
 'paired_t6':('single',True,(3,4,6)),
 'single_t18':('single',False,(6,12,18)),
 'paired_t18':('single',True,(6,12,18)),
 'pooled_t6':('pooled',False,(3,4,6)),
 'robust_t6':('robust',False,(3,4,6)),
 'pooled_paired6':('pooled',True,(3,4,6)),
 'robust_paired6':('robust',True,(3,4,6)),
}
STRESSES=['clean','variance4','matched_bias','one_bad','matched_plus_one_bad','source_shared','two_bad','common_unmatched','serial','heavy_tail']

def generate(loss,sd,rng,stress):
    n=len(loss); sh=(n,3,18); e=rng.normal(size=sh); c=rng.normal(size=sh)
    if stress=='variance4':e*=2;c*=2
    if stress=='source_shared':
        e=np.sqrt(.7)*rng.normal(size=(n,1,18))+np.sqrt(.3)*e
        c=np.sqrt(.7)*rng.normal(size=(n,1,18))+np.sqrt(.3)*c
    if stress=='serial':
        for k in range(1,18):
            e[:,:,k]=.8*e[:,:,k-1]+.6*e[:,:,k];c[:,:,k]=.8*c[:,:,k-1]+.6*c[:,:,k]
    if stress=='heavy_tail':e=rng.standard_t(3,sh)/np.sqrt(3);c=rng.standard_t(3,sh)/np.sqrt(3)
    target=loss[:,None,None]+sd[:,None,None]*e;control=sd[:,None,None]*c
    # Adversarial bias points toward acceptance; matched offset is present in control.
    if stress in ('matched_bias','matched_plus_one_bad'):
        offset=-1.5*sd[:,None,None]*np.ones((n,3,1));target+=offset;control+=offset
    if stress in ('one_bad','matched_plus_one_bad','two_bad','common_unmatched'):
        q={'one_bad':1,'matched_plus_one_bad':1,'two_bad':2,'common_unmatched':3}[stress]
        # Faulted source chosen without using outcomes; bias constant within episode.
        ranks=rng.random((n,3)).argsort(1); mask=(ranks<q)
        target-=3*sd[:,None,None]*mask[:,:,None]
    return target,control

def gate(name,x,c,sd,g):
    mode,paired,looks=ARMS[name];return evidence_gate(x,c,sd,g,mode=mode,paired=paired,looks=looks)

def boundary(n=30000):
    rows=[]
    for i,stress in enumerate(STRESSES):
        for delta in [1e-8,-.5,-1.5]:
            rng=np.random.default_rng(41000000+i*1000+int((delta+2)*100));sd=np.ones(n)*.12;loss=EPS+delta*sd;g=np.ones(n,int);x,c=generate(loss,sd,rng,stress)
            for name in ARMS:
                a=gate(name,x,c,sd,g);k=int(a['accept'].sum());lo=beta.ppf(.025,k,n-k+1) if k else 0;hi=beta.ppf(.975,k+1,n-k) if k<n else 1
                rows.append(dict(stress=stress,delta_sd=delta,method=name,n=n,accepted=k,rate=k/n,lo=lo,hi=hi,samples=a['samples'].mean(),disagreement=a['disagreement'].mean()))
    pd.DataFrame(rows).to_csv(OUT/'boundary.csv',index=False)

DYNAMIC=[('irregular_recurrence','clean'),('random_walk','clean'),('gradual','clean'),('mean_reversion','clean'),('mean_reversion','variance4'),('mean_reversion','matched_bias'),('mean_reversion','one_bad'),('mean_reversion','matched_plus_one_bad'),('mean_reversion','source_shared'),('mean_reversion','two_bad'),('mean_reversion','common_unmatched'),('mean_reversion','serial'),('mean_reversion','heavy_tail')]

def dynamic(n=80):
    rows=[]
    for i,(scene,stress) in enumerate(DYNAMIC):
        seed=42000000+i*10000;truth,y,v,bank,r,*_=make(scene,n,seed);b=simulate(y,v,bank,r,threshold=.95,seed=seed+123)
        net=truth[:,12:]-C;oracle=np.maximum(net,0).mean(2);ca=b['candidate'][:,12:];ref=b['reference'][:,12:];cv=(ca*net).mean(2);rv=(ref*net).mean(2);loss=rv-cv;groups=(ca!=ref).sum(2);eligible=(b['model_accept'][:,12:]>0)&(groups>0);idx=np.nonzero(eligible);g=groups[idx];sd=np.sqrt(.45*g)/G
        x,c=generate(loss[idx],sd,np.random.default_rng(seed+99900000),stress)
        for name in ['v38','recompute',*ARMS]:
            selected=np.zeros_like(eligible);cost=np.zeros_like(loss);samples=np.zeros_like(loss);disagree=np.zeros_like(eligible)
            if name=='v38':selected=eligible.copy()
            elif name!='recompute':
                a=gate(name,x,c,sd,g);selected[idx]=a['accept'];cost[idx]=a['cost'];samples[idx]=a['samples'];disagree[idx]=a['disagreement']
            unsafe=loss>EPS
            rows.append(pd.DataFrame(dict(scene=scene,stress=stress,seed=np.arange(seed,seed+n),method=name,periods=H-12,regret=(oracle-np.where(selected,cv,rv)).sum(1),probe_cost=b['probes'][:,12:].sum(1)*PROBE_COST,audit_cost=cost.sum(1),samples=samples.sum(1),accepted=selected.sum(1),unsafe=(selected&unsafe).sum(1),unsafe_eligible=(eligible&unsafe).sum(1),eligible=eligible.sum(1),disagreement=disagree.sum(1))))
        print(scene,stress,n,'eligible',len(g),flush=True)
    d=pd.concat(rows,ignore_index=True);d.to_csv(OUT/'trajectories.csv',index=False)
    def metrics(g):
        s=g.sum(numeric_only=True);return dict(trajectories=len(g),net_loss=(s.regret+s.probe_cost+s.audit_cost)/s.periods,regret=s.regret/s.periods,audit_cost=s.audit_cost/s.periods,accepted=int(s.accepted),unsafe=int(s.unsafe),eligible=int(s.eligible),unsafe_eligible=int(s.unsafe_eligible),false_accept=s.unsafe/s.unsafe_eligible if s.unsafe_eligible else np.nan,unsafe_given_accept=s.unsafe/s.accepted if s.accepted else np.nan,samples=int(s.samples),disagreement=int(s.disagreement))
    pd.DataFrame([dict(scene=k[0],stress=k[1],method=k[2],**metrics(g)) for k,g in d.groupby(['scene','stress','method'])]).to_csv(OUT/'scenes.csv',index=False)
    pd.DataFrame([dict(stress=k[0],method=k[1],**metrics(g)) for k,g in d.groupby(['stress','method'])]).to_csv(OUT/'summary.csv',index=False)
    rng=np.random.default_rng(4101);out=[]
    for (scene,stress),f in d.groupby(['scene','stress']):
        for name in ARMS:
            a=f[f.method==name].sort_values('seed');b=f[f.method=='known3'].sort_values('seed');assert np.array_equal(a.seed,b.seed)
            x=((a.regret+a.audit_cost+a.probe_cost).to_numpy()-(b.regret+b.audit_cost+b.probe_cost).to_numpy())/(H-12)
            z=x[rng.integers(n,size=(1500,n))].mean(1);out.append(dict(scene=scene,stress=stress,method=name,baseline='known3',delta=x.mean(),lo=np.quantile(z,.025),hi=np.quantile(z,.975)))
    pd.DataFrame(out).to_csv(OUT/'paired.csv',index=False)
if __name__=='__main__':
    import sys
    OUT.mkdir(parents=True,exist_ok=True);n=int(sys.argv[1]) if len(sys.argv)>1 else 80
    (OUT/'config.json').write_text(json.dumps(dict(n=n,boundary_n=30000,scenes=DYNAMIC,stresses=STRESSES,arms=ARMS,epsilon=EPS,alpha=.05,unit_cost=PROBE_COST,seed_boundary=41000000,seed_dynamic=42000000,seed_bootstrap=4101),indent=2));boundary();dynamic(n)
