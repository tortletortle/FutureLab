"""Held-out synthetic evaluation, with explicitly billed independent audits."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from benchmark_v38 import data,HOLD38,H,G
from v35 import C
from v38 import simulate as old_sim,EPS,PROBE_COST
from v39 import simulate,AUDIT_VAR
OUT=Path(__file__).resolve().parents[1]/'results'
NEW=('mean_reversion','seasonal_smooth','asynchronous_local','heavy_tail_discovery','biased_audit_stress')
ARMS=(('v38',False,False,False,.95),('strict_v38',False,False,False,.99),('adjusted',True,False,False,.95),('quarantine',True,True,False,.95),('audit_only',False,False,True,.95),('combined',True,True,True,.95))
def make(scene,n,seed):
    if scene in HOLD38:truth,y,v,bank,r=data(scene,n,seed)
    else:
        rng=np.random.default_rng(seed);base=rng.normal(0,1.1,(n,G));x=np.tile(base[:,None,:],(1,H,1));v=np.full_like(x,.625);drift=rng.normal(0,1,(n,G));times=np.arange(H)
        if scene=='mean_reversion':
            for t in range(12,H):x[:,t]=base+.92*(x[:,t-1]-base)+rng.normal(0,.36,(n,G))
        elif scene=='seasonal_smooth':x+=1.7*drift[:,None,:]*np.sin(2*np.pi*times[None,:,None]/24+rng.uniform(0,2*np.pi,(n,1,G)))
        elif scene=='asynchronous_local':
            start=rng.integers(12,50,(n,G));x+=2*drift[:,None,:]*np.clip((times[None,:,None]-start[:,None,:])/15,0,1)
        else:x[:,20:40]+=1.8*drift[:,None,:];x[:,55:]+=.9*drift[:,None,:]
        truth=x+C;noise=rng.normal(size=x.shape) if scene!='heavy_tail_discovery' else rng.standard_t(3,size=x.shape)/np.sqrt(3)
        y=truth+np.sqrt(v)*noise;r=np.full_like(x,.45);bank=truth[:,:,:,None]+np.sqrt(r[:,:,:,None])*rng.normal(size=(*x.shape,1))
    rng=np.random.default_rng(seed+61999999);av=np.full_like(truth,AUDIT_VAR);audit=truth+np.sqrt(av)*rng.normal(size=truth.shape)
    if scene=='biased_audit_stress':audit+=rng.choice([-1,1],(n,1,G))*.65
    return truth,y,v,bank,r,audit,av

def run(scenes,n,base,split):
    rows=[]
    for i,scene in enumerate(scenes):
        seed=base+10000*i;truth,y,v,bank,r,audit,av=make(scene,n,seed);net=truth[:,12:]-C;oracle=np.maximum(net,0).mean(2)
        for name,cal,q,a,t in ARMS:
            b=simulate(y,v,bank,r,audit,av,calibrate=cal,quarantine=q,audit_gate=a,threshold=t,seed=seed+123)
            if name=='v38':
                old=old_sim(y,v,bank,r,threshold=t,seed=seed+123)
                for key in ('decision','reference','candidate','selected','birth'):np.testing.assert_array_equal(b[key],old[key])
            ca=b['candidate'][:,12:];ref=b['reference'][:,12:];dec=b['decision'][:,12:];cv=(ca*net).mean(2);rv=(ref*net).mean(2);dv=(dec*net).mean(2);reuse=b['reuse'][:,12:].astype(bool);distinct=(ca!=ref).any(2);bad=rv-cv>EPS;tested=b['audit_groups'][:,12:]>0
            d=dict(regret=(oracle-dv).sum(1),reference_regret=(oracle-rv).sum(1),reuse=reuse.sum(1),distinct=(reuse&distinct).sum(1),unsafe=(reuse&bad).sum(1),unsafe_candidates=(bad&distinct).sum(1),audited_unsafe=(tested&bad).sum(1),audit_false_accept=(tested&bad&reuse).sum(1),audited=tested.sum(1),audit_groups=b['audit_groups'][:,12:].sum(1),audit_cost=b['audit_cost'][:,12:].sum(1),probe_cost=b['probes'][:,12:].sum(1)*PROBE_COST,quarantine=b['quarantine'][:,12:].sum(1),old_reuse=b['old_reuse'][:,12:].sum(1),births=b['birth'][:,12:].sum(1),periods=np.full(n,H-12))
            rows.append(pd.DataFrame(dict(split=split,scene=scene,seed=np.arange(seed,seed+n),method=name,**d)))
            if name=='v38':
                d0={**d,'regret':d['reference_regret'],'reuse':np.zeros(n),'distinct':np.zeros(n),'unsafe':np.zeros(n),'old_reuse':np.zeros(n)}
                rows.append(pd.DataFrame(dict(split=split,scene=scene,seed=np.arange(seed,seed+n),method='recompute',**d0)))
        print(split,scene,n,flush=True)
    return pd.concat(rows,ignore_index=True)
def measure(g):
    s=g.sum(numeric_only=True);p=s.periods;d=s.distinct;au=s.audited_unsafe
    return dict(trajectories=len(g),periods=int(p),regret=s.regret/p,reference_regret=s.reference_regret/p,reuse=s.reuse/p,distinct_reuse=d/p,unsafe_given_distinct=s.unsafe/d if d else np.nan,unsafe=int(s.unsafe),distinct=int(d),unsafe_all=s.unsafe/p,audit_attempts=int(s.audited),audit_unsafe_attempts=int(au),audit_false_accept=int(s.audit_false_accept),false_accept_given_unsafe_audits=s.audit_false_accept/au if au else np.nan,audit_groups=s.audit_groups/p,audit_cost=s.audit_cost/p,probe_cost=s.probe_cost/p,net_loss=(s.regret+s.audit_cost+s.probe_cost)/p,quarantine=s.quarantine/p,old_reuse=s.old_reuse/p)
def summarize(d):
    rows=[];sc=[]
    for (split,method),g in d.groupby(['split','method']):rows.append(dict(split=split,method=method,**measure(g)))
    for (split,scene,method),g in d.groupby(['split','scene','method']):sc.append(dict(split=split,scene=scene,method=method,**measure(g)))
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(sc).to_csv(OUT/'scenes.csv',index=False)
    rng=np.random.default_rng(731);pairs=[]
    for split,g in d.groupby('split'):
        for method,*_ in ARMS[1:]+(('recompute',),):
            z=g[g.method==method].merge(g[g.method=='v38'],on=['scene','seed'],suffixes=('_a','_b'))
            for metric in ('regret','net_loss','unsafe_all','distinct_reuse'):
                if metric=='regret':a=z.regret_a;b=z.regret_b
                elif metric=='net_loss':a=z.regret_a+z.audit_cost_a+z.probe_cost_a;b=z.regret_b+z.audit_cost_b+z.probe_cost_b
                elif metric=='unsafe_all':a=z.unsafe_a;b=z.unsafe_b
                else:a=z.distinct_a;b=z.distinct_b
                q=z.assign(delta=a/z.periods_a-b/z.periods_b);means=[];boot=[]
                for _,f in q.groupby('scene'):
                    x=f.delta.to_numpy();means.append(x.mean());boot.append(x[rng.integers(len(x),size=(1500,len(x)))].mean(1))
                e=np.mean(boot,axis=0);pairs.append(dict(split=split,method=method,baseline='v38',metric=metric,delta=np.mean(means),lo=np.quantile(e,.025),hi=np.quantile(e,.975)))
    pd.DataFrame(pairs).to_csv(OUT/'paired.csv',index=False)
def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 100
    cfg=dict(n=n,hold_scenes=HOLD38,new_scenes=NEW,hold_seed=13900000,new_seed=14900000,stress_seed=15900000,warmup=12,arms=ARMS,epsilon=EPS,audit_var=AUDIT_VAR,audit_alpha=.05,probe_cost=PROBE_COST,audit_cost_per_group=PROBE_COST*.45/AUDIT_VAR,quarantine_threshold=1.75,bias_ema=.20,scale_ema=.15,bootstrap=1500)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2))
    d=pd.concat([run(HOLD38,n,cfg['hold_seed'],'same_family'),run(NEW[:-1],n,cfg['new_seed'],'new_family'),run(NEW[-1:],n,cfg['stress_seed'],'invalid_audit_stress')],ignore_index=True);d.to_csv(OUT/'trajectories.csv',index=False);summarize(d)
    print(pd.read_csv(OUT/'summary.csv')[['split','method','regret','distinct_reuse','unsafe_given_distinct','audit_cost','net_loss','false_accept_given_unsafe_audits']].to_string(index=False))
if __name__=='__main__':main()
