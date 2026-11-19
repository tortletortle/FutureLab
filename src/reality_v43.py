"""Locked static policy benchmark; fixture mode never reads real data."""
from pathlib import Path
import argparse, csv, hashlib, json, urllib.request, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

SEED=431901
SHA='bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f'
URL='https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz'
NUM=['recency','history','mens','womens','newbie']
CAT=['history_segment','zip_code','channel']
FEAT=NUM+CAT
ARMS=['No E-Mail','Mens E-Mail','Womens E-Mail']
PRIMARY=(.25,.05,.3)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def download(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial')
    try:
        with urllib.request.urlopen(URL,timeout=45) as response, tmp.open('wb') as f:
            while (chunk:=response.read(65536)): f.write(chunk)
        if sha(tmp)!=SHA: raise ValueError('mirror SHA256 mismatch')
        tmp.replace(path)
    finally: tmp.unlink(missing_ok=True)

def load_real(path):
    if sha(path)!=SHA: raise ValueError('unverified bytes; expected compressed mirror SHA256')
    df=pd.read_csv(path,compression='gzip')
    df.columns=df.columns.str.strip().str.lower()
    if len(df)!=64000 or not set(FEAT+['segment','spend']).issubset(df) or set(df.segment)!=set(ARMS):
        raise ValueError('invalid Hillstrom schema or row count')
    validate(df)
    return df

def validate(df):
    if df[FEAT+['segment','spend']].isna().any().any(): raise ValueError('missing values')
    if not np.isfinite(df[NUM+['spend']].to_numpy(dtype=float)).all() or (df.spend<0).any(): raise ValueError('nonfinite or negative outcome')

def fixture(n=9000):
    rng=np.random.default_rng(SEED+5); r=rng.integers(1,13,n); h=rng.lognormal(4.8,.8,n); mens=rng.binomial(1,.6,n); womens=rng.binomial(1,.6,n)
    a=rng.integers(0,3,n); tau1=.2+2*(r<=3)+.6*mens; tau2=.1+1.8*womens-.2*mens
    y=np.maximum(0,1.5+.12*(12-r)+.003*h+(a==1)*tau1+(a==2)*tau2+rng.normal(0,.55,n))
    return pd.DataFrame(dict(recency=r,history=h,mens=mens,womens=womens,newbie=rng.integers(0,2,n),history_segment=pd.cut(h,[-1,100,200,500,np.inf],labels=['a','b','c','d']).astype(str),zip_code=rng.choice(['urban','rural','suburban'],n),channel=rng.choice(['web','phone'],n),segment=np.array(ARMS)[a],spend=y))

def split(df,arm):
    d=df[df.segment.isin([ARMS[0],arm])].copy(); d['a']=(d.segment==arm).astype(int)
    train,rest=train_test_split(d,test_size=.4,random_state=SEED,stratify=d.a)
    val,test=train_test_split(rest,test_size=.5,random_state=SEED+1,stratify=rest.a)
    assert not (set(train.index)&set(val.index) or set(train.index)&set(test.index) or set(val.index)&set(test.index))
    return train,val,test

def model(seed):
    return XGBRegressor(n_estimators=100,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.9,reg_lambda=5,n_jobs=2,random_state=seed,objective='reg:squarederror')

def nuisance_fit(X,a,y,seed):
    return [model(seed+i).fit(X[a==i],y[a==i]) for i in range(2)]

def predict(ms,X): return np.column_stack([m.predict(X) for m in ms])

def dr_score(a,y,pred,p=.5):
    return pred[:,1]-pred[:,0]+a*(y-pred[:,1])/p-(1-a)*(y-pred[:,0])/(1-p)

def choose(score,cap,cost,margin,priority=None):
    score=np.asarray(score,float); assert np.isfinite(score).all()
    net=margin*score-cost; k=int(np.floor(cap*len(score)))
    if priority is None: priority=np.arange(len(score))
    order=np.lexsort((priority,-net)); selected=order[:k]; out=np.zeros(len(score),dtype=int); out[selected[net[selected]>0]]=1
    return out

def force_cap(priority,cap):
    out=np.zeros(len(priority),dtype=int); out[np.argsort(-priority,kind='stable')[:int(np.floor(cap*len(priority)))]]=1; return out

def contributions(action,a,y,pred,cost,margin):
    return action*(margin*dr_score(a,y,pred)-cost), action*(margin*(a*y/.5-(1-a)*y/.5)-cost)

def interval(x):
    x=np.asarray(x); m=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(len(x))); return m,se,m-1.96*se,m+1.96*se

def fit_scores(train,val,test,arm):
    tr=ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
    X=tr.fit_transform(train[FEAT]); V=tr.transform(val[FEAT]); S=tr.transform(test[FEAT]); a=train.a.to_numpy(); y=train.spend.to_numpy()
    ms=nuisance_fit(X,a,y,SEED); pv,ps=predict(ms,V),predict(ms,S)
    ridge=Ridge(alpha=20).fit(np.c_[X,a,X*a[:,None]],y)
    def lin(z): return ridge.predict(np.c_[z,np.ones(len(z)),z])-ridge.predict(np.c_[z,np.zeros(len(z)),np.zeros_like(z)])
    oof=np.zeros((len(X),2))
    for i,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=SEED).split(X,a)):
        oof[iv]=predict(nuisance_fit(X[it],a[it],y[it],SEED+i+10),X[iv])
    z=dr_score(a,y,oof)
    rng=np.random.default_rng(SEED+22); models=[]
    for i in range(5):
        idx=rng.integers(0,len(X),len(X)); models.append(ExtraTreesRegressor(n_estimators=60,min_samples_leaf=100,max_features=.8,n_jobs=2,random_state=SEED+i).fit(X[idx],z[idx]))
    def score(d,x,p):
        b=np.array([m.predict(x) for m in models]); cat='mens' if arm==ARMS[1] else 'womens'
        return {'none':np.zeros(len(d)),'random':np.random.default_rng(SEED+8).random(len(d)), 'recent_rule':-d.recency.to_numpy(float), 'category_rule':d[cat].to_numpy(float), 'outcome_xgb':p[:,1], 'linear_interaction':lin(x),'t_xgb':p[:,1]-p[:,0],'dr_extra_trees':b.mean(0),**{f'futurelab_{l}':b.mean(0)-l*b.std(0,ddof=1) for l in [.5,1.,2.]}}
    sv,st=score(val,V,pv),score(test,S,ps)
    causal_status='unavailable (econml not installed)'
    try:
        from econml.dml import CausalForestDML
    except ImportError: pass
    else:
        cf=CausalForestDML(discrete_treatment=True,n_estimators=200,min_samples_leaf=50,random_state=SEED,n_jobs=2).fit(y,a,X=X)
        sv['causal_forest']=cf.effect(V); st['causal_forest']=cf.effect(S); causal_status='included (CausalForestDML)'
    return sv,st,pv,ps,causal_status

def policy_actions(scores,name,cap,cost,margin):
    if name=='none': return np.zeros(len(scores[name]),dtype=int)
    if name in ['random','recent_rule','category_rule','outcome_xgb']:
        # Rank-only policies lack effect calibration; fixed quota except none.
        return force_cap(scores[name],cap)
    return choose(scores[name],cap,cost,margin)

def benchmark(df,output,mode):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); allrows=[]; sensitivity=[]; comps=[]; manifest={}
    for arm in ARMS[1:]:
        train,val,test=split(df,arm); sv,st,pv,ps,cf=fit_scores(train,val,test,arm)
        def evaluate(d,s,p,name,condition):
            cap,cost,margin=condition; action=policy_actions(s,name,cap,cost,margin)
            aipw,ipw=contributions(action,d.a.to_numpy(),d.spend.to_numpy(),p,cost,margin)
            return action,aipw,ipw
        vals={name:evaluate(val,sv,pv,name,PRIMARY)[1].mean() for name in sv}
        base=max((k for k in vals if not k.startswith('futurelab')),key=vals.get); future=max((k for k in vals if k.startswith('futurelab')),key=vals.get)
        manifest[arm]={'n_train':len(train),'n_validation':len(val),'n_test':len(test),'baseline':base,'futurelab':future,'causal_forest':cf,'validation_scores':{k:float(v) for k,v in vals.items()},'split_index_sha256':{k:hashlib.sha256(np.asarray(d.index,dtype=np.int64).tobytes()).hexdigest() for k,d in [('train',train),('val',val),('test',test)]}}
        for name in st:
            a,v,i=evaluate(test,st,ps,name,PRIMARY); m,se,lo,hi=interval(v); allrows.append(dict(arm=arm,method=name,n=len(test),coverage=a.mean(),aipw=m,se=se,lo=lo,hi=hi,ipw=i.mean()))
        _,f,_=evaluate(test,st,ps,future,PRIMARY); _,b,_=evaluate(test,st,ps,base,PRIMARY); m,se,lo,hi=interval(f-b)
        lift=m/b.mean() if b.mean()>0 else None
        comps.append(dict(arm=arm,baseline=base,futurelab=future,baseline_value=b.mean(),futurelab_value=f.mean(),difference=m,lo=lo,hi=hi,relative_lift=lift,passes=bool(lo>0 and (lift is None or lift>=.10)),confirmatory=(mode=='real' and arm==ARMS[1])))
        for cap in [.1,.25,.5]:
            for cost in [0,.05,.2]:
                for margin in [.1,.3,.5]:
                    for name in [base,future]:
                        a,v,i=evaluate(test,st,ps,name,(cap,cost,margin)); sensitivity.append(dict(arm=arm,method=name,cap=cap,cost=cost,margin=margin,coverage=a.mean(),aipw=v.mean(),ipw=i.mean()))
    pd.DataFrame(allrows).to_csv(output/'policy_results.csv',index=False); pd.DataFrame(comps).to_csv(output/'comparison.csv',index=False); pd.DataFrame(sensitivity).to_csv(output/'sensitivity.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(dict(mode=mode,seed=SEED,primary=PRIMARY,arms=manifest,source_sha256=SHA if mode=='real' else None),indent=2,ensure_ascii=False))
    print(pd.DataFrame(comps).to_string(index=False))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--fixture',action='store_true'); p.add_argument('--download',action='store_true'); p.add_argument('--data',default='data/hillstrom.csv.gz'); p.add_argument('--output'); args=p.parse_args()
    if args.fixture and args.download: p.error('fixture and download are mutually exclusive')
    if args.download: download(args.data)
    mode='synthetic_fixture' if args.fixture else 'real'; out=args.output or f'results/v43/{mode}'
    if Path(out,'manifest.json').exists(): raise SystemExit('refusing to overwrite prior evaluation; use a new output directory')
    df=fixture() if args.fixture else load_real(args.data); validate(df); benchmark(df,out,mode)
if __name__=='__main__': main()
