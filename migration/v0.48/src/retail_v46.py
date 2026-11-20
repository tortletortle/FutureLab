"""Chronological recorded-sales order-quantity benchmark; not inventory simulation."""
from pathlib import Path
import argparse, hashlib, json, tempfile, zipfile
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from retail_v25_loader import load_uci_streaming

ZIP_SHA='572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb'
XLSX_SHA='bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980'
Q=.75
SEED=4606
CANDIDATE='futurelab_group_calibrated'

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare(path):
    if sha(path)!=ZIP_SHA:raise ValueError('unexpected input ZIP hash')
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/'retail.xlsx'
        with zipfile.ZipFile(path) as z,z.open('online_retail_II.xlsx') as src,p.open('wb') as dst:
            import shutil
            shutil.copyfileobj(src,dst)
        if sha(p)!=XLSX_SHA:raise ValueError('unexpected XLSX hash')
        panel,quality=load_uci_streaming(p)
    return panel,{**quality,'zip_sha256':ZIP_SHA,'xlsx_sha256':XLSX_SHA}

def features(panel):
    p=panel[['series_id','date','demand']].copy(); p['date']=pd.to_datetime(p.date)
    if p.empty or p.duplicated(['series_id','date']).any() or p.series_id.isna().any() or not np.isfinite(p.demand).all() or (p.demand<0).any() or (p.date.dt.dayofweek!=6).any():raise ValueError('invalid Sunday panel')
    ids=sorted(p.series_id.unique()); weeks=pd.date_range(p.date.min(),p.date.max(),freq='W-SUN')
    if len(weeks)<40:raise ValueError('too few weeks')
    d=p.set_index(['series_id','date']).reindex(pd.MultiIndex.from_product([ids,weeks],names=['series_id','date'])).reset_index()
    d['recorded_row']=d.demand.notna(); d['global_recorded_week']=d.date.isin(p.date); d['demand']=d.demand.fillna(0.)
    d['week']=d.date.map(dict(zip(weeks,range(len(weeks)))))
    # Exclude first/last as targets but retain first week lag? First partial is not a complete history: exclude before features.
    d=d[d.week.between(1,len(weeks)-2)].copy(); d['week']-=1
    g=d.groupby('series_id',sort=False).demand
    for lag in (1,2,4,8,13):d[f'lag{lag}']=g.shift(lag)
    for w in (4,8,13):
        roll=g.transform(lambda s:s.shift().rolling(w).mean()); std=g.transform(lambda s:s.shift().rolling(w).std(ddof=0))
        d[f'mean{w}']=roll; d[f'std{w}']=std
    d['zero8']=g.transform(lambda s:s.shift().eq(0).rolling(8).mean())
    for w in (8,13):d[f'quantile{w}']=g.transform(lambda s:s.shift().rolling(w).quantile(Q))
    phase=d.date.dt.isocalendar().week.astype(float)*2*np.pi/52.18
    d['sin_week']=np.sin(phase); d['cos_week']=np.cos(phase)
    d['scale']=d.mean13.clip(lower=1)
    d['sid']=d.series_id.map({k:i for i,k in enumerate(ids)}).astype(int)
    n=len(weeks)-2; cuts=[int(n*f) for f in (.65,.75,.85)]
    d['split']=np.select([d.week<cuts[0],d.week<cuts[1],d.week<cuts[2]],['train','calibration','selection'],default='test')
    d=d[d.week>=13].copy().reset_index(drop=True)
    if d.split.nunique()!=4:raise ValueError('empty split')
    return d,cuts

COLS=['sid','lag1','lag2','lag4','lag8','lag13','mean4','mean8','mean13','std4','std8','std13','zero8','sin_week','cos_week']

def loss(y,q):
    y,q=np.asarray(y,float),np.asarray(q,float)
    if y.shape!=q.shape or not np.isfinite(y).all() or not np.isfinite(q).all() or (y<0).any() or (q<0).any():raise ValueError('invalid observed sales or order')
    return 3*np.maximum(y-q,0)+np.maximum(q-y,0)

def groups(train, other):
    avg=train.groupby('series_id').demand.mean(); edges=np.quantile(avg,[1/3,2/3]);
    if not other.series_id.isin(avg.index).all():raise ValueError('unknown series')
    return np.searchsorted(edges,other.series_id.map(avg).to_numpy(),side='right'),edges

def calibrate(train,cal,cal_point,target,target_point):
    cg,edges=groups(train,cal); tg,_=groups(train,target)
    resid=cal.demand.to_numpy()-cal_point; normalized=resid/cal.scale.to_numpy(); pooled=float(np.quantile(normalized,Q))
    offsets={int(i):float(np.quantile(normalized[cg==i],Q)) if sum(cg==i)>=50 else pooled for i in range(3)}
    result={'global_additive':target_point+np.quantile(resid,Q),'global_scaled':target_point+target.scale.to_numpy()*pooled,CANDIDATE:target_point+target.scale.to_numpy()*np.array([offsets[i] for i in tg])}
    return result,{'edges':edges.tolist(),'pooled':pooled,'offsets':offsets,'counts':{int(i):int(sum(cg==i)) for i in range(3)}}

def block_ci(delta,week,seed=SEED,reps=2000):
    if reps<1:raise ValueError('invalid replicates')
    means=pd.DataFrame({'delta':delta,'week':week}).groupby('week').delta.mean().to_numpy(); n=len(means)
    if n<4:raise ValueError('insufficient weeks')
    rng=np.random.default_rng(seed); starts=rng.integers(0,n,(reps,int(np.ceil(n/4))))
    idx=((starts[:,:,None]+np.arange(4))%n).reshape(reps,-1)[:,:n]
    return np.quantile(means[idx].mean(1),[.025,.975]).tolist()

def benchmark(panel,quality,out):
    d,cuts=features(panel); train=d[d.split=='train']; cal=d[d.split=='calibration']; select=d[d.split=='selection']; test=d[d.split=='test']; targets=pd.concat([select,test]);
    # Keep series ID categorical in HGB; XGB uses numeric series ID (no future outcomes).
    cat=[True]+[False]*(len(COLS)-1)
    common=dict(max_iter=120,max_leaf_nodes=20,min_samples_leaf=30,learning_rate=.08,l2_regularization=2,random_state=SEED,categorical_features=cat)
    point=HistGradientBoostingRegressor(**common).fit(train[COLS],np.log1p(train.demand)); quant=HistGradientBoostingRegressor(**common,loss='quantile',quantile=Q).fit(train[COLS],train.demand)
    xgb=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.055,objective='reg:quantileerror',quantile_alpha=Q,subsample=.9,colsample_bytree=.9,reg_lambda=5,n_jobs=2,random_state=SEED).fit(train[COLS],train.demand)
    pred=np.expm1(point.predict(targets[COLS])).clip(0); cp=np.expm1(point.predict(cal[COLS])).clip(0)
    methods={'last_week':targets.lag1.to_numpy(),'rolling8':targets.quantile8.to_numpy(),'rolling13':targets.quantile13.to_numpy(),'hgb_point':pred,'hgb_quantile':quant.predict(targets[COLS]),'xgb_quantile':xgb.predict(targets[COLS])}
    extra,meta=calibrate(train,cal,cp,targets,pred); methods.update(extra)
    y=targets.demand.to_numpy(); sm=targets.split.to_numpy()=='selection'; tm=~sm
    orders={k:np.rint(np.maximum(v,0)) for k,v in methods.items()}; costs={k:loss(y,v) for k,v in orders.items()}
    champion=min((k for k in costs if k!=CANDIDATE),key=lambda k:float(costs[k][sm].mean()))
    rows=[]; detail=[]
    for k,q in orders.items():
        for label,mask in [('selection',sm),('test',tm)]:
            yy=y[mask]; qq=q[mask]; rows.append({'split':label,'method':k,'n':len(yy),'mean_cost':costs[k][mask].mean(),'mean_shortage':np.maximum(yy-qq,0).mean(),'mean_overage':np.maximum(qq-yy,0).mean(),'coverage':(yy<=qq).mean(),'wape_order':np.abs(yy-qq).sum()/max(yy.sum(),1)})
        temp=targets.loc[tm,['series_id','date','week','demand','recorded_row','global_recorded_week']].copy(); temp['method']=k; temp['order']=q[tm]; temp['cost']=costs[k][tm]; detail.append(temp)
    delta=costs[CANDIDATE][tm]-costs[champion][tm]; base=float(costs[champion][tm].mean())
    result={'mode':'real_historical_recorded_sales_proxy','quality':quality,'splits':{k:{'n':len(p),'weeks':p.week.nunique(),'start':str(p.date.min().date()),'end':str(p.date.max().date())} for k,p in [('train',train),('calibration',cal),('selection',select),('test',test)]},'calendar_cut_indices':cuts,'globally_unrecorded_model_weeks':sorted(d.loc[~d.global_recorded_week,'date'].dt.strftime('%Y-%m-%d').unique().tolist()),'champion':champion,'candidate':CANDIDATE,'paired_cost_delta':float(delta.mean()),'relative_cost_reduction':float(-delta.mean()/base) if base>0 else None,'block_bootstrap_descriptive_ci':block_ci(delta,targets.week.to_numpy()[tm]),'calibration':meta,'versions':{k:version(k) for k in ('numpy','pandas','scikit-learn','xgboost','openpyxl')},'limitations':'Recorded sales; no stockout or inventory truth; costs assumed, not money; same data as V25; retrospective single-period proxy.'}
    out=Path(out); out.mkdir(parents=True,exist_ok=False); pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False); pd.concat(detail).to_csv(out/'test_orders.csv',index=False); (out/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)); print(json.dumps(result,indent=2)); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,default=Path('results/v46/retail')); a=p.parse_args();
    if a.output.exists():p.error('choose fresh output directory')
    panel,quality=prepare(a.data); benchmark(panel,quality,a.output)
if __name__=='__main__':main()
