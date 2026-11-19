import json,sys,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from v35 import *
from v34 import WINDOWS
OUT=Path(__file__).resolve().parents[1]/'results'
def batch_scores(y,v):
    B,H,K=y.shape;z=np.zeros_like(y);dense=np.zeros((B,H))
    for w in WINDOWS:
        for t in range(2*w-1,H):
            a=slice(t-2*w+1,t-w+1);b=slice(t-w+1,t+1)
            zz=(y[:,b].mean(1)-y[:,a].mean(1))/np.sqrt((v[:,b].sum(1)+v[:,a].sum(1))/w**2)
            z[:,t]=np.maximum(z[:,t],abs(zz));dense[:,t]=np.maximum(dense[:,t],(zz*zz).sum(1))
    return z,dense
def batch_hybrid(y,v,threshold):
    B,H,K=y.shape;z,dense=batch_scores(y,v);ll=4.5*threshold;dd=(6.5*threshold)**2
    last=np.full((B,K),-6);lastd=np.full(B,-6);starts=np.zeros((B,K),int);until=np.full(B,-1)
    p=np.concatenate([np.zeros((B,1,K)),np.cumsum(1/v,1)],1);a=np.concatenate([np.zeros((B,1,K)),np.cumsum(y/v,1)],1)
    mu=np.full((B,K),C);pred=[];b=np.arange(B)[:,None];j=np.arange(K)[None,:]
    for t in range(H):
        pred.append(mu.copy());local=(z[:,t]>ll)&(t-last>=6);de=(dense[:,t]>dd)&(t-lastd>=6);last=np.where(local,t,last);lastd=np.where(de,t,lastd)
        starts[de]=max(0,t-1);until[de]=t+8;active=t<=until;starts[active]=np.maximum(starts[active],max(0,t-3));starts[local]=t
        mu=(a[:,t+1]-a[b,starts,j])/(p[:,t+1]-p[b,starts,j])
    return np.stack(pred,1)
def write(d,name,base):
    d.to_csv(OUT/f'{name}_trials.csv',index=False);d.groupby(['scene','method']).mean(numeric_only=True).drop(columns='seed').to_csv(OUT/f'{name}_summary.csv');rows=[];rg=np.random.default_rng(20261119)
    for scene,g in d.groupby('scene'):
        p=g.pivot(index='seed',columns='method',values='regret');idx=rg.integers(len(p),size=(2500,len(p)))
        for m in p:
            if m==base:continue
            diff=(p[m]-p[base]).to_numpy();b=diff[idx].mean(1);rows.append(dict(scene=scene,method=m,baseline=base,delta=diff.mean(),lo=np.quantile(b,.025),hi=np.quantile(b,.975)))
    pd.DataFrame(rows).to_csv(OUT/f'{name}_paired.csv',index=False)
def dynamic(n=400):
    with zipfile.ZipFile(Path(__file__).resolve().parents[1]/'previous_version/futurelab_v34.zip') as f:
        path=next(x for x in f.namelist() if x.endswith('/hybrid_thresholds.json'));th=json.loads(f.read(path))['joint']
    rows=[];traces=[]
    for scene in DYNAMIC_SCENES:
        truth,y,v=(np.stack(x) for x in zip(*(extra_generate(710000+i,scene) for i in range(n))))
        net=truth-C;oracle=np.maximum(net,0);pred,loss,trace=adaptive(y,v)
        for name in ('cumulative','rolling4','rolling8','exp85','exp65'):pred[name],loss[name]=fixed(y,v,name)
        pred['hybrid_v34']=batch_hybrid(y,v,th)
        for name in METHODS:
            dec=pred[name]>C;regret=(oracle-dec*net)[:,16:].mean((1,2));harm=(dec*np.maximum(-net,0))[:,16:].mean((1,2));value=(dec*net)[:,16:].mean((1,2));switch=(dec[:,1:]!=dec[:,:-1])[:,15:].mean((1,2));nll=loss[name][:,16:].mean((1,2)) if name in loss else np.full(n,np.nan)
            for i in range(n):rows.append(dict(scene=scene,seed=710000+i,method=name,regret=regret[i],harm=harm[i],value=value[i],switch=switch[i],marginal_nll=nll[i]))
        for t in range(T):traces.append(dict(scene=scene,t=t,average_discount=trace[:,t].mean(),group0_discount=trace[:,t,0].mean(),group10_discount=trace[:,t,10].mean()))
    write(pd.DataFrame(rows),'dynamic','rolling4');pd.DataFrame(traces).to_csv(OUT/'memory_trace.csv',index=False)
    (OUT/'dynamic_metadata.json').write_text(json.dumps(dict(seeds=[710000,710000+n-1],n=n,scenes=DYNAMIC_SCENES,discounts=DISCOUNTS.tolist(),share=SHARE,blend=.5,hybrid_joint_threshold=th,eval_periods=[16,47]),indent=2))
def static(n=160):
    rows=[]
    for scene in ('matched','independent','negative','intermediate','strong','block'):
        rows.extend(r for i in range(n) for r in static_trial(810000+i,scene));print('static',scene,flush=True)
    write(pd.DataFrame(rows),'static','independent');(OUT/'static_metadata.json').write_text(json.dumps(dict(seeds=[810000,810000+n-1],n=n,budget=8,hypotheses=RHOS.tolist(),nodes=25,node_check=61),indent=2))
if __name__=='__main__':
    if len(sys.argv)==1 or sys.argv[1]=='dynamic':dynamic()
    if len(sys.argv)==1 or sys.argv[1]=='static':static()
