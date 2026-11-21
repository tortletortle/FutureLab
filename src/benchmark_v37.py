"""Separate development and holdout episode simulator; no label enters policy."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import beta
from v35 import C
from v37 import run_episode
OUT=Path(__file__).resolve().parents[1]/'results'
DEV=['return','dense_near','sparse_near','far','noisy_return','near_return']
HOLD=DEV+['tiny_near','two_sparse','noisy_near','correlated_near','confusable_return','heteroskedastic_near']
METHODS=[('broad_random','random',False,3),('structured_passive','random',True,0),('structured_random','random',True,3),('structured_uncertainty','uncertainty',True,3),('structured_disagreement','disagreement',True,3),('structured_decision','decision',True,3),('structured_active','active',True,3)]
G,K=12,4

def generate(seed,scene,max_probes=3):
    rng=np.random.default_rng(seed);base=C+rng.normal(0,.8,G);a=base+rng.normal(0,1.2,(K,G));j=int(rng.integers(K));truth=a[j].copy();is_known='return' in scene
    if scene in ('near_return','confusable_return'):
        a[1]=a[0]+rng.normal(0,.35 if scene=='near_return' else .16,G);j=int(rng.integers(2));truth=a[j].copy()
    if scene=='dense_near':truth+=rng.normal(0,.6,G)
    if scene=='tiny_near':truth+=rng.normal(0,.25,G)
    if scene in ('sparse_near','two_sparse'):
        ids=rng.choice(G,1 if scene=='sparse_near' else 2,replace=False);truth[ids]+=rng.choice([-1,1],len(ids))*rng.uniform(1.4,2.6,len(ids))
    if scene=='far':truth=C+rng.normal(0,2,G)
    if scene=='noisy_near':truth+=rng.normal(0,.65,G)
    if scene=='correlated_near':truth+=rng.choice([-1,1])*.65+rng.normal(0,.15,G)
    if scene=='heteroskedastic_near':truth+=rng.normal(0,.65,G)
    # historical archive estimates from 8 independent contrasts per group; known sampling variance.
    av=np.full((K,G),.8/8);am=a+rng.normal(size=(K,G))*np.sqrt(av)
    dv=np.full(G,1.25);pv=np.full(G,.45)
    if scene.startswith('noisy'):dv*=2.5;pv*=2.5
    if scene=='heteroskedastic_near':dv*=rng.lognormal(0,.9,G);pv*=rng.lognormal(0,.9,G)
    y=truth+rng.normal(size=G)*np.sqrt(dv);bank=truth[:,None]+rng.normal(size=(G,max_probes))*np.sqrt(pv[:,None])
    return am,av,y,dv,bank,pv,truth,j if is_known else -1

def collect(scenes,n,base):
    rows=[]
    for i,scene in enumerate(scenes):
        seeds=base+10000*i+np.arange(n);am,av,y,dv,bank,pv,truth,label=map(np.stack,zip(*(generate(int(s),scene) for s in seeds)))
        net=truth-C
        for name,method,structured,budget in METHODS:
            out=run_episode(am,av,y,dv,bank,pv,method,budget,structured,base+7+i);h=out['history'][-1];p=h['mean'];regret=(np.maximum(net,0)-(p>C)*net).mean(1);known=label>=0
            brier=(h['known']-known)**2;logloss=-np.log(np.clip(np.where(known,h['known'],1-h['known']),1e-12,1))
            for z in range(n):rows.append(dict(scene=scene,seed=int(seeds[z]),method=name,budget=budget,truth_slot=int(label[z]),candidate=int(h['slot'][z]),confidence=h['conf'][z],known_prob=h['known'][z],regret=regret[z],brier=brier[z],logloss=logloss[z],probes=';'.join(map(str,out['choices'][z]))))
        print(scene,n,flush=True)
    return pd.DataFrame(rows)

def metrics(d,t):
    a=d.confidence>=t;k=d.truth_slot>=0;correct=d.candidate==d.truth_slot
    rate=lambda x,mask:float(x[mask].mean()) if mask.any() else np.nan
    n=int((~k).sum());x=int((a&~k).sum());hi=1. if n and x==n else float(beta.ppf(.95,x+1,n-x)) if n else np.nan
    return dict(threshold=t,n=len(d),novel_n=n,novel_false_accept=x,novel_far=rate(a,~k),novel_far_upper95=hi,known_coverage=rate(a,k),correct_known_recall=rate(a&correct,k),wrong_known_recall=rate(a&~correct,k),precision=rate(correct,a),coverage=float(a.mean()),regret=d.regret.mean(),brier=d.brier.mean(),logloss=d.logloss.mean())

def calibrate(d):
    thresholds=np.r_[np.linspace(.5,.995,100),1.000001];out={}
    for name,g in d.groupby('method'):
        # constrain empirical dev novel FAR, choose best correct recall; break ties by higher threshold.
        candidates=[metrics(g,t) for t in thresholds];valid=[r for r in candidates if r['novel_far']<=.08];best=max(valid,key=lambda r:(r['correct_known_recall'],r['threshold']));out[name]=best['threshold']
    return out

def main():
    ndev=120;nhold=300
    d=collect(DEV,ndev,1270000);d.to_csv(OUT/'development_trials.csv',index=False);thresholds=calibrate(d);(OUT/'config_v37.json').write_text(json.dumps(dict(dev=DEV,hold=HOLD,dev_base=1270000,hold_base=2370000,ndev=ndev,nhold=nhold,thresholds=thresholds,dev_empirical_novel_far_target=.08,probe_budget=3),indent=2));print('thresholds',thresholds,flush=True)
    h=collect(HOLD,nhold,2370000);h.to_csv(OUT/'trials_v37.csv',index=False);out=[];curves=[];sc=[]
    for name,g in h.groupby('method'):
        t=thresholds[name];out.append(dict(method=name,split='hold',**metrics(g,t)));out.append(dict(method=name,split='dev',**metrics(d[d.method==name],t)))
        for scene,f in g.groupby('scene'):sc.append(dict(method=name,scene=scene,**metrics(f,t)))
        for t in [.5,.6,.7,.8,.9,.95,.975,.99,1.000001]:curves.append(dict(method=name,**metrics(g,t)))
    pd.DataFrame(out).to_csv(OUT/'summary_v37.csv',index=False);pd.DataFrame(sc).to_csv(OUT/'scenes_v37.csv',index=False);pd.DataFrame(curves).to_csv(OUT/'curves_v37.csv',index=False)
    # Paired stratified bootstrap for pre-specified active relative to random and uncertainty.
    rng=np.random.default_rng(5401);pairs=[]
    for baseline in ['structured_random','structured_uncertainty','structured_disagreement','structured_decision']:
        for metric in ['regret','brier','logloss','correct_known','false_accept_novel']:
            a=h[h.method=='structured_active'].copy();b=h[h.method==baseline].copy()
            def get(f,name):
                if metric=='correct_known':return f[(f.truth_slot>=0)].assign(z=lambda g:((g.confidence>=thresholds[name])&(g.candidate==g.truth_slot)).astype(float))
                if metric=='false_accept_novel':return f[(f.truth_slot<0)].assign(z=lambda g:(g.confidence>=thresholds[name]).astype(float))
                return f.assign(z=f[metric])
            p=get(a,'structured_active').merge(get(b,baseline),on=['scene','seed'],suffixes=('_a','_b'));vals=[];boots=[]
            for _,f in p.groupby('scene'):
                z=(f.z_a-f.z_b).to_numpy();vals.append(z.mean());boots.append(z[rng.integers(len(z),size=(2000,len(z)))].mean(1))
            boot=np.mean(boots,axis=0);pairs.append(dict(method='structured_active',baseline=baseline,metric=metric,delta=np.mean(vals),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975),scenes=len(vals)))
    pd.DataFrame(pairs).to_csv(OUT/'paired_v37.csv',index=False)
if __name__=='__main__':main()
