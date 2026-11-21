import sys,json,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np,pandas as pd
from v35 import C,adaptive,fixed
from v36 import Config,archive,guarded
OUT=Path(__file__).resolve().parents[1]/'results'
DEV=('stable','recurrence','near_impostor','gradual','noise','local_recurrence')
HOLD=('stable','noise','recurrence','irregular_recurrence','three_regimes','near_impostor','nonrecurring','local_recurrence','noisy_recurrence','gradual','random_walk','capacity_pressure','short_recurrence')
H,G=80,12

def generate(seed,scene):
    rng=np.random.default_rng(seed);base=C+rng.normal(0,1.1,G);direction=rng.choice([-1.,1.],G);shift=1.65*direction+rng.normal(0,.22,G);second=rng.normal(0,1.5,G);third=rng.normal(0,1.6,G)
    truth=np.tile(base,(H,1));sig=np.full((H,G),2.5);label=np.zeros(H,int)
    def block(start,end,lab,delta):truth[start:end]=base+delta;label[start:end]=lab
    if scene=='noise':sig[20:45]*=2.5;sig[60:]*=.65
    if scene in ('recurrence','noisy_recurrence','near_impostor'):
        block(16,32,1,shift);block(48,64,1 if scene!='near_impostor' else 2,shift if scene!='near_impostor' else shift+.75*rng.normal(size=G))
        if scene=='noisy_recurrence':sig[15:66]*=1.9
    if scene=='irregular_recurrence':
        p=0;l=0
        while p<H:
            q=min(H,p+int(rng.integers(7,19)));block(p,q,l,shift*l);p=q;l=1-l
    if scene=='three_regimes':
        for i,l in enumerate([0,1,2,0,2,1,0,2]):block(i*10,(i+1)*10,l,[np.zeros(G),shift,second][l])
    if scene=='nonrecurring':
        for i in range(1,5):block(i*16,(i+1)*16,i,rng.normal(0,1.65,G))
    if scene=='local_recurrence':
        d=np.zeros(G);ids=rng.choice(G,2,replace=False);d[ids]=3*direction[ids];block(16,32,1,d);block(48,64,1,d)
    if scene=='gradual':truth[12:]+=np.linspace(0,1,H-12)[:,None]*shift;label[:]=-1
    if scene=='random_walk':truth[12:]+=np.cumsum(rng.normal(0,.23,(H-12,G)),0);label[:]=-1
    if scene=='capacity_pressure':
        ds=[np.zeros(G),shift,second,third]+[rng.normal(0,1.7,G) for _ in range(4)]
        for i,l in enumerate([0,1,2,3,4,5,6,7,0,1]):block(i*8,(i+1)*8,l,ds[l])
    if scene=='short_recurrence':
        for i in range(10):block(i*8,(i+1)*8,i%2,shift*(i%2))
    y=truth+rng.normal(size=(H,G))*sig*np.sqrt(2/20);v=sig**2*rng.chisquare(38,(H,G))/38*2/20
    return truth,y,v,label

def get_batch(scene,n,seed):return tuple(np.stack(x) for x in zip(*(generate(seed+i,scene) for i in range(n))))

def score(pred,truth,labels):
    net=truth-C;dec=pred>C;r=(np.maximum(net,0)-dec*net).mean(2);harm=(dec*np.maximum(-net,0)).mean(2)
    # Genuine returns: first four periods of a non-initial visit to a previously seen exact regime.
    mask=np.zeros(labels.shape,bool)
    for b,ls in enumerate(labels):
        seen=set([ls[0]]);end=0
        for t in range(1,len(ls)):
            if ls[t]!=ls[t-1]:
                end=min(len(ls),t+4) if ls[t]>=0 and ls[t] in seen else t
                seen.add(ls[t]);
            if t<end:mask[b,t]=True
    counts=mask.sum(1);ret=np.divide((r*mask).sum(1),counts,out=np.full(len(r),np.nan),where=counts>0)
    # First visits to new non-initial discrete regimes, evaluated separately.
    novel=np.zeros(labels.shape,bool)
    for b,ls in enumerate(labels):
        seen=set([ls[0]]);end=0
        for t in range(1,len(ls)):
            if ls[t]!=ls[t-1]:
                end=min(len(ls),t+4) if ls[t]>=0 and ls[t] not in seen else t
                seen.add(ls[t])
            if t<end:novel[b,t]=True
    nc=novel.sum(1);nr=np.divide((r*novel).sum(1),nc,out=np.full(len(r),np.nan),where=nc>0)
    return dict(regret=r[:,12:].mean(1),harm=harm[:,12:].mean(1),switch=(dec[:,12:]!=dec[:,11:-1]).mean((1,2)),return_regret=ret,return_periods=counts,novel_regret=nr,novel_periods=nc,full_regret=r.mean(1))

def dev(n=100):
    configs=[Config(hazard=h,birth=p,birth_gate=g) for h,p in [(.10,.025),(.18,.06),(.28,.10)] for g in [.55,.75]]
    rows=[]
    for j,scene in enumerate(DEV):
        truth,y,v,label=get_batch(scene,n,920000+j*1000);base,loss,_=adaptive(y,v)
        for k,cfg in enumerate(configs):
            a=archive(y,v,cfg);p,w=guarded(a['pred'],a['loss'],base['adaptive_global'],loss['adaptive_global'],cfg.ensemble_share)
            rows.append(dict(config=k,scene=scene,bank=score(a['pred'],truth,label)['regret'].mean(),guarded=score(p,truth,label)['regret'].mean(),births=(a['event']==1).sum(1).mean()))
    d=pd.DataFrame(rows);d.to_csv(OUT/'development.csv',index=False);i=int(d.groupby('config').guarded.mean().idxmin());(OUT/'config.json').write_text(json.dumps(dict(selected=asdict(configs[i]),selected_index=i,candidates=[asdict(c) for c in configs],selection='minimum equal-scene guarded regret on development',dev_seed_base=920000,dev_n=n,dev_scenes=DEV),indent=2));print('selected',i,configs[i],flush=True)

def hold(n=300):
    cfg=Config(**json.loads((OUT/'config.json').read_text())['selected']);rows=[];events=[];traces=[]
    for j,scene in enumerate(HOLD):
        seed=970000+j*1000;truth,y,v,label=get_batch(scene,n,seed);base,loss,_=adaptive(y,v)
        pred={k:base[k] for k in ['adaptive_global','adaptive_local','adaptive_blend']};pred['rolling4']=fixed(y,v,'rolling4')[0];pred['rolling8']=fixed(y,v,'rolling8')[0];pred['cumulative']=fixed(y,v,'cumulative')[0]
        banks={}
        for name,c in [('archive',cfg),('no_recall',replace(cfg,slots=1))]:
            a=archive(y,v,c);banks[name]=a;pred[name]=a['pred'];pred[name+'_guarded'],w=guarded(a['pred'],a['loss'],base['adaptive_global'],loss['adaptive_global'],c.ensemble_share)
            if name=='archive':
                for b,t in zip(*np.where(a['event']!=0)):
                    src=a['source'][b,t];events.append(dict(scene=scene,seed=seed+b,t=int(t),event='birth' if a['event'][b,t]==1 else 'recall',source_t=int(src),age=int(a['age'][b,t]),confidence=a['confidence'][b,t],truth_label=int(label[b,t]),birth_label=int(label[b,src]),same_label=int(label[b,t]==label[b,src]) if label[b,t]>=0 else None))
                for t in range(H):traces.append(dict(scene=scene,t=t,archive_weight=w[:,t].mean(),slots=a['count'][:,t].mean(),birth_rate=(a['event'][:,t]==1).mean(),recall_rate=(a['event'][:,t]==2).mean()))
        for name,p in pred.items():
            metrics=score(p,truth,label)
            for b in range(n):rows.append(dict(scene=scene,seed=seed+b,method=name,**{k:x[b] for k,x in metrics.items()}))
        print(scene,'done',flush=True)
    d=pd.DataFrame(rows);d.to_csv(OUT/'trials.csv',index=False);d.groupby(['scene','method']).mean(numeric_only=True).drop(columns='seed').to_csv(OUT/'summary.csv');pd.DataFrame(events).to_csv(OUT/'events.csv',index=False);pd.DataFrame(traces).to_csv(OUT/'trace.csv',index=False)
    out=[];rng=np.random.default_rng(20261119)
    for scene,g in d.groupby('scene'):
        for metric in ('regret','return_regret','novel_regret'):
            p=g.pivot(index='seed',columns='method',values=metric).dropna();
            if len(p)==0:continue
            idx=rng.integers(len(p),size=(3000,len(p)))
            for m,b in [('archive','no_recall'),('archive_guarded','no_recall_guarded'),('archive_guarded','adaptive_global'),('archive','adaptive_global')]:
                z=(p[m]-p[b]).to_numpy();boot=z[idx].mean(1);out.append(dict(scene=scene,metric=metric,method=m,baseline=b,n=len(z),delta=z.mean(),lo=np.quantile(boot,.025),hi=np.quantile(boot,.975)))
    pd.DataFrame(out).to_csv(OUT/'paired.csv',index=False);(OUT/'metadata.json').write_text(json.dumps(dict(scenes=HOLD,n=n,seed_base=970000,scene_stride=1000,periods=H,groups=G,warmup=12,units_per_arm=20,config=asdict(cfg),bootstrap=3000,return_window=4),indent=2))
if __name__=='__main__':
    if len(sys.argv)==1 or sys.argv[1]=='dev':dev()
    if len(sys.argv)==1 or sys.argv[1]=='hold':hold()
