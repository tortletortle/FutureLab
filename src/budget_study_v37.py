"""Independent budget study with disjoint development and holdout seed blocks."""
import json
import numpy as np,pandas as pd
from benchmark_v37 import DEV,HOLD,generate,calibrate,metrics,OUT,C
from v37 import run_episode
BUDGETS=(0,3,6,12,24)
PROBES=('random','uncertainty','active','decision')

def collect(scenes,n,base):
    rows=[]
    for i,scene in enumerate(scenes):
        seeds=base+10000*i+np.arange(n);am,av,y,dv,bank,pv,truth,label=map(np.stack,zip(*(generate(int(s),scene,max(BUDGETS)) for s in seeds)));net=truth-C
        for method in PROBES:
            out=run_episode(am,av,y,dv,bank,pv,method,max(BUDGETS),True,base+i+5)
            for budget in BUDGETS:
                h=out['history'][budget];p=h['mean'];k=label>=0;r=(np.maximum(net,0)-(p>C)*net).mean(1)
                for z in range(n):rows.append(dict(scene=scene,seed=int(seeds[z]),method=f'{method}:{budget}',budget=budget,truth_slot=int(label[z]),candidate=int(h['slot'][z]),confidence=h['conf'][z],known_prob=h['known'][z],regret=r[z],brier=(h['known'][z]-k[z])**2,logloss=-np.log(np.clip(np.where(k[z],h['known'][z],1-h['known'][z]),1e-12,1))))
        print(scene,n,flush=True)
    return pd.DataFrame(rows)

def main():
    d=collect(DEV,100,3370000);ts=calibrate(d);h=collect(HOLD,150,4370000)
    d.to_csv(OUT/'budget_development.csv',index=False);h.to_csv(OUT/'budget_trials.csv',index=False)
    rows=[];sc=[]
    for name,g in h.groupby('method'):
        m,b=name.split(':');t=ts[name];rows.append(dict(method=m,budget=int(b),**metrics(g,t)))
        for scene,f in g.groupby('scene'):sc.append(dict(method=m,budget=int(b),scene=scene,**metrics(f,t)))
    pd.DataFrame(rows).to_csv(OUT/'budget_summary.csv',index=False);pd.DataFrame(sc).to_csv(OUT/'budget_scenes.csv',index=False)
    (OUT/'budget_config.json').write_text(json.dumps(dict(dev_base=3370000,hold_base=4370000,ndev=100,nhold=150,budgets=BUDGETS,methods=PROBES,target=.08,thresholds=ts),indent=2))
if __name__=='__main__':main()
