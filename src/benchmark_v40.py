from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from benchmark_v39 import make,NEW,HOLD38,H,G
from v39 import simulate,audit_upper
from v40 import audit,potential_gain,ROUND_VAR
from v38 import EPS,PROBE_COST
from v35 import C
OUT=Path(__file__).resolve().parents[1]/'results'
ARMS=[('v38',None,0,False,False,EPS),('recompute',None,0,False,False,EPS),('fixed3','fixed',3,False,False,EPS),('spend3','spend',3,False,False,EPS),('mixture3','mixture',3,False,False,EPS),('spend6','spend',6,False,False,EPS),('screen_spend3','spend',3,True,False,EPS),('profit_spend3','spend',3,False,False,0.),('soft_fixed3','fixed',3,False,True,EPS),('peek3_invalid','peek',3,False,False,EPS)]
def run(scene,n,seed,split,stress=None):
    truth,y,v,bank,r,_,_=make(scene,n,seed);rng=np.random.default_rng(seed+98300000);z=rng.normal(size=(*truth.shape,6))
    if stress=='correlated':z=np.sqrt(.7)*rng.normal(size=(*truth.shape,1))+np.sqrt(.3)*z
    if stress=='variance':z*=2
    ab=truth[:,:,:,None]+np.sqrt(ROUND_VAR)*z
    if stress=='bias':ab+=rng.choice([-1,1],(n,1,G,1))*.65
    av=np.full_like(truth,ROUND_VAR);rows=[];net=truth[:,12:]-C;oracle=np.maximum(net,0).mean(2)
    bases={soft:simulate(y,v,bank,r,threshold=.95,seed=seed+123,diagnostic=potential_gain,soft_write=soft) for soft in (False,True)}
    for name,method,m,screen,soft,eps in ARMS:
        b=bases[soft];ca=b['candidate'][:,12:];ref=b['reference'][:,12:];cv=(ca*net).mean(2);rv=(ref*net).mean(2);bad=rv-cv>EPS;distinct=(ca!=ref).any(2);pick=b['model_accept'][:,12:].astype(bool).copy();eligible=pick&distinct
        samples=np.zeros_like(cv);cost=np.zeros_like(cv);tested=np.zeros_like(pick);screened=np.zeros_like(pick);rounds=np.zeros_like(cv)
        if method:
            idx=np.nonzero(eligible);aa=audit(ca[idx],ref[idx],ab[:,12:][idx],av[:,12:][idx],method=method,max_rounds=m,epsilon=eps,value_bound=b['diagnostic'][:,12:][idx] if screen else None)
            if name=='fixed3':np.testing.assert_array_equal(aa['accept'],audit_upper(ca[idx],ref[idx],ab[:,12:][idx][...,:3].mean(2),av[:,12:][idx]/3)<=EPS)
            pick[idx]=aa['accept'];samples[idx]=aa['samples'];rounds[idx]=aa['rounds'];cost[idx]=aa['cost'];tested[idx]=aa['rounds']>0;screened[idx]=aa['screened']
        elif name=='recompute':pick[:]=False
        d=dict(regret=(oracle-np.where(pick,cv,rv)).sum(1),reference_regret=(oracle-rv).sum(1),reuse=pick.sum(1),distinct=(pick&distinct).sum(1),unsafe=(pick&bad).sum(1),eligible=eligible.sum(1),unsafe_eligible=(eligible&bad).sum(1),tested=tested.sum(1),unsafe_tested=(tested&bad).sum(1),false_accept=(tested&bad&pick).sum(1),samples=samples.sum(1),rounds=rounds.sum(1),audit_cost=cost.sum(1),screened=screened.sum(1),probe_cost=b['probes'][:,12:].sum(1)*PROBE_COST,periods=np.full(n,H-12))
        rows.append(pd.DataFrame(dict(split=split,scene=stress or scene,seed=np.arange(seed,seed+n),method=name,**d)))
    print(split,stress or scene,n,flush=True);return pd.concat(rows,ignore_index=True)
def measure(g):
    s=g.sum(numeric_only=True);p=s.periods;d=s.distinct;t=s.unsafe_tested
    return dict(trajectories=len(g),periods=int(p),regret=s.regret/p,net_loss=(s.regret+s.audit_cost+s.probe_cost)/p,distinct_rate=d/p,unsafe=int(s.unsafe),distinct=int(d),unsafe_given_distinct=s.unsafe/d if d else np.nan,unsafe_per_period=s.unsafe/p,unsafe_tested=int(t),false_accept=int(s.false_accept),false_accept_given_unsafe=s.false_accept/t if t else np.nan,audit_cost=s.audit_cost/p,samples=int(s.samples),rounds_per_attempt=s.rounds/s.tested if s.tested else np.nan,tested=int(s.tested),screened=int(s.screened),eligible=int(s.eligible))
def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 100
    cfg=dict(n=n,seed_old=24000000,seed_new=25000000,seed_stress=26000000,scenes=list(HOLD38)+list(NEW[:-1]),stress=['bias','correlated','variance'],arms=ARMS,alpha=.05,epsilon=EPS,round_var=ROUND_VAR,unit_cost=PROBE_COST,archive_threshold=.95,warmup=12,bootstrap=1500)
    (OUT/'config.json').write_text(json.dumps(cfg,indent=2));frames=[]
    for split,scenes,base in [('same_family',HOLD38,cfg['seed_old']),('new_family',NEW[:-1],cfg['seed_new'])]:
        for i,scene in enumerate(scenes):frames.append(run(scene,n,base+i*10000,split))
    for i,stress in enumerate(cfg['stress']):frames.append(run('mean_reversion',n,cfg['seed_stress']+i*10000,'invalid_audit',stress))
    data=pd.concat(frames,ignore_index=True);data.to_csv(OUT/'trajectories.csv',index=False)
    for fields,fname in [(['split','method'],'summary'),(['split','scene','method'],'scenes')]:pd.DataFrame([dict(zip(fields,k),**measure(g)) for k,g in data.groupby(fields)]).to_csv(OUT/(fname+'.csv'),index=False)
    rng=np.random.default_rng(4039);rows=[]
    for split,g in data.groupby('split'):
        for name,*_ in ARMS:
            for base in ['fixed3','recompute']:
                if name==base:continue
                x=g[g.method==name].merge(g[g.method==base],on=['scene','seed'],suffixes=('_a','_b'),validate='one_to_one')
                for metric in ['net_loss','regret','unsafe_per_period','audit_cost']:
                    def vals(side):
                        def col(c):return x[c+'_'+side]
                        z=col('regret')+col('audit_cost')+col('probe_cost') if metric=='net_loss' else col('unsafe' if metric=='unsafe_per_period' else metric)
                        return z/col('periods')
                    delta=vals('a')-vals('b');res=[];obs=[]
                    for _,q in x.assign(delta=delta).groupby('scene'):
                        a=q.delta.to_numpy();obs.append(a.mean());res.append(a[rng.integers(len(a),size=(1500,len(a)))].mean(1))
                    z=np.mean(res,axis=0);rows.append(dict(split=split,method=name,baseline=base,metric=metric,delta=np.mean(obs),lo=np.quantile(z,.025),hi=np.quantile(z,.975)))
    pd.DataFrame(rows).to_csv(OUT/'paired.csv',index=False)
    print(pd.read_csv(OUT/'summary.csv')[['split','method','net_loss','distinct_rate','unsafe_given_distinct','false_accept_given_unsafe','audit_cost','rounds_per_attempt']].to_string(index=False))
if __name__=='__main__':main()
