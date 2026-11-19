import json
from pathlib import Path
import numpy as np,pandas as pd
from benchmark_v36 import get_batch,DEV,HOLD,H,G
from v35 import C
from v38 import simulate, EPS, PROBE_COST
OUT=Path(__file__).resolve().parents[1]/'results'
DEV38=tuple(DEV);HOLD38=tuple(HOLD)+('same_actions','boundary_flip','correlated_noise')
ARMS=[('evsi','evsi','risk',1,True),('identity','evsi','identity',1,True),('random','random','risk',1,True),('uncertainty','uncertainty','risk',1,True),('novelty','active','risk',1,True),('passive','random','risk',0,True),('no_feedback','evsi','risk',1,False)]
def data(scene,n,seed):
    if scene in HOLD:truth,y,v,label=get_batch(scene,n,seed)
    else:
        truths=[];ys=[];vs=[]
        for i in range(n):
            rng=np.random.default_rng(seed+i);base=rng.normal(0,1.15,G);net=np.tile(base,(H,1));r=np.full_like(net,.625)
            if scene=='same_actions':net[16:32]*=2.5;net[48:64]*=.25
            if scene=='boundary_flip':
                base[:4]=rng.choice([-1,1],4)*.18;net[:]=base;net[16:32,:4]*=-1;net[48:64,:4]*=-1
            noise=rng.normal(size=net.shape)*np.sqrt(r)
            if scene=='correlated_noise':
                net[16:32]+=rng.choice([-1,1],G)*1.6;net[48:64]=net[16:32];noise=np.sqrt(r)*(.65*rng.normal(size=(H,1))+np.sqrt(1-.65**2)*rng.normal(size=net.shape))
            truths.append(net+C);ys.append(net+C+noise);vs.append(r)
        truth,y,v=map(np.stack,(truths,ys,vs))
    rng=np.random.default_rng(seed+55555555);r=np.full_like(y,.45);bank=truth[:,:,:,None]+rng.normal(size=(*truth.shape,1))*np.sqrt(r[:,:,:,None]);return truth,y,v,bank,r

def collect(scenes,n,base):
    out=[]
    for i,scene in enumerate(scenes):
        seed=base+i*10000;truth,y,v,bank,r=data(scene,n,seed);net=truth[:,12:]-C;oracle=np.maximum(net,0).mean(2)
        for name,selector,gate,k,feedback in ARMS:
            a=simulate(y,v,bank,r,selector,gate,threshold=0.,budget=k,seed=seed+123,feedback=feedback)
            ca=a['candidate'][:,12:];ref=a['reference'][:,12:];cv=(ca*net).mean(2);rv=(ref*net).mean(2)
            out.append(pd.DataFrame(dict(scene=scene,seed=np.repeat(np.arange(seed,seed+n),H-12),period=np.tile(np.arange(12,H),n),method=name,confidence=a['confidence'][:,12:].ravel(),risk=a['risk'][:,12:].ravel(),candidate_regret=(oracle-cv).ravel(),reference_regret=(oracle-rv).ravel(),excess=(rv-cv).ravel(),old=a['old_reuse'][:,12:].ravel(),match=a['match'][:,12:].ravel(),probes=a['probes'][:,12:].ravel(),birth=a['birth'][:,12:].ravel())))
        print(scene,n,flush=True)
    return pd.concat(out,ignore_index=True)

def measure(d,t):
    a=d.confidence>=t;bad=d.excess>EPS;nonmatch=d['match']<.5;reg=np.where(a,d.candidate_regret,d.reference_regret);nd=int((a&nonmatch).sum())
    return dict(threshold=t,coverage=a.mean(),distinct_reuse=(a&nonmatch).mean(),old_reuse=(a&(d.old>.5)).mean(),unsafe_given_reuse=(a&bad).sum()/max(1,a.sum()),unsafe_given_distinct=(a&nonmatch&bad).sum()/max(1,nd),unsafe_all=(a&bad).mean(),regret=reg.mean(),reference_regret=d.reference_regret.mean(),mean_excess=np.mean(reg-d.reference_regret),probes=d.probes.mean(),net_loss=reg.mean()+PROBE_COST*d.probes.mean(),births=d.birth.mean(),n_periods=len(d),n_reuse=int(a.sum()),n_distinct=nd,n_unsafe=int((a&bad).sum()))

def calibrate(d):
    ts=np.r_[np.linspace(0,1,101),1.000001];thresholds={}
    for name,g in d.groupby('method'):
        rows=[measure(g,t) for t in ts];ok=[x for x in rows if x['unsafe_given_distinct']<=.08 and (x['n_distinct']>=50 or x['n_distinct']==0)]
        thresholds[name]=float(max(ok,key=lambda a:(a['coverage'],a['threshold']))['threshold'])
    return thresholds

def main():
    import sys
    ndev,nhold=(70,100) if len(sys.argv)==1 else map(int,sys.argv[1:3]);d=collect(DEV38,ndev,6380000);ts=calibrate(d);d.to_csv(OUT/'development_v38.csv.gz',index=False);print('thresholds',ts,flush=True)
    h=collect(HOLD38,nhold,7380000);h.to_csv(OUT/'hold_periods_v38.csv.gz',index=False)
    (OUT/'config_v38.json').write_text(json.dumps(dict(dev=DEV38,hold=HOLD38,dev_seed=6380000,hold_seed=7380000,ndev=ndev,nhold=nhold,thresholds=ts,eps=EPS,probe_cost=PROBE_COST,dev_target_nonmatch=.08,probe_interval=4),indent=2))
    rows=[];sc=[];curves=[];trajectory=[]
    for name,g in h.groupby('method'):
        t=ts[name]
        for split,f in [('hold',g),('dev',d[d.method==name])]:rows.append(dict(method=name,split=split,**measure(f,t)))
        for scene,f in g.groupby('scene'):sc.append(dict(method=name,scene=scene,**measure(f,t)))
        for t0 in [0,.5,.7,.8,.9,.95,.975,.99,1,1.000001]:curves.append(dict(method=name,**measure(g,t0)))
        a=g.confidence>=t;g=g.assign(regret=np.where(a,g.candidate_regret,g.reference_regret),reuse=a.astype(float),unsafe=(a&(g.excess>EPS)).astype(float),distinct=(a&(g['match']<.5)).astype(float));trajectory.append(g.groupby(['scene','seed'])[['regret','reuse','unsafe','distinct','probes','reference_regret']].mean().reset_index().assign(method=name))
    s=pd.DataFrame(rows);s.to_csv(OUT/'summary_v38.csv',index=False);pd.DataFrame(sc).to_csv(OUT/'scenes_v38.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'curves_v38.csv',index=False)
    traj=pd.concat(trajectory);traj.to_csv(OUT/'trajectories_v38.csv',index=False);rng=np.random.default_rng(734);pairs=[]
    for b in ['identity','random','uncertainty','novelty','passive','no_feedback']:
        z=traj[traj.method=='evsi'].merge(traj[traj.method==b],on=['scene','seed'],suffixes=('_a','_b'))
        for metric in ['regret','reuse','unsafe','distinct','reference_regret']:
            means=[];boots=[]
            for _,f in z.groupby('scene'):
                x=(f[metric+'_a']-f[metric+'_b']).to_numpy();means.append(x.mean());boots.append(x[rng.integers(len(x),size=(1000,len(x)))].mean(1))
            q=np.mean(boots,axis=0);pairs.append(dict(baseline=b,metric=metric,delta=np.mean(means),lo=np.quantile(q,.025),hi=np.quantile(q,.975)))
    pd.DataFrame(pairs).to_csv(OUT/'paired_v38.csv',index=False);print(s[s.split=='hold'][['method','threshold','coverage','distinct_reuse','unsafe_given_distinct','regret','net_loss']].to_string(index=False),flush=True)
if __name__=='__main__':main()
