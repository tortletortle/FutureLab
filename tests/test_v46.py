import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from retail_v46 import features,loss,groups,calibrate,block_ci,COLS,CANDIDATE

def fixture():
    return pd.DataFrame([{'series_id':f'p{i}','date':d,'demand':float(i+j%9)} for i in range(6) for j,d in enumerate(pd.date_range('2010-01-03',periods=80,freq='W-SUN'))])

def test_asymmetric_cost():
    assert loss([10,10,10],[8,12,10]).tolist()==[6,2,0]
    with pytest.raises(ValueError):loss([1],[-1])

def test_calendar_and_missing():
    p=fixture(); p=p[~((p.series_id=='p0')&(p.date=='2010-06-06'))]; d,_=features(p); row=d[(d.series_id=='p0')&(d.date=='2010-06-06')].iloc[0]; assert row.demand==0 and not row.recorded_row
    assert d[(d.series_id=='p0')&(d.date=='2010-06-13')].lag1.iloc[0]==0

def test_no_target_or_future_leakage():
    p=fixture(); d,_=features(p); pivot=pd.Timestamp('2010-11-07'); p.loc[p.date>=pivot,'demand']+=9999; e,_=features(p)
    pd.testing.assert_frame_equal(d.loc[d.date<=pivot,COLS].reset_index(drop=True),e.loc[e.date<=pivot,COLS].reset_index(drop=True))

def test_splits_chronological():
    d,_=features(fixture()); labels=['train','calibration','selection','test']; assert all(d[d.split==a].date.max()<d[d.split==b].date.min() for a,b in zip(labels,labels[1:]))
    assert d.groupby('date').size().nunique()==1

def test_reject_duplicate():
    p=fixture();
    with pytest.raises(ValueError):features(pd.concat([p,p.iloc[:1]]))

def test_groups_do_not_see_target():
    d,_=features(fixture()); t=d[d.split=='train']; e=d[d.split=='test'].copy(); a,edges=groups(t,e); e['demand']+=100; b,edges2=groups(t,e); np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(edges,edges2)

def test_calibration_falls_back_on_small_groups_and_ignores_target():
    d,_=features(fixture()); t=d[d.split=='train']; c=d[d.split=='calibration']; e=d[d.split=='test'].copy(); cp=np.zeros(len(c)); ep=np.zeros(len(e)); a,m=calibrate(t,c,cp,e,ep); e['demand']*=1000; b,_=calibrate(t,c,cp,e,ep); np.testing.assert_allclose(a[CANDIDATE],a['global_scaled']);np.testing.assert_allclose(a[CANDIDATE],b[CANDIDATE]); assert len(m['offsets'])==3

def test_group_calibration_differs_when_supported():
    t=pd.DataFrame({'series_id':['a','b','c'],'demand':[1,10,100]}); c=pd.DataFrame({'series_id':np.repeat(['a','b','c'],60),'demand':np.repeat([2.,10.,50.],60),'scale':1.}); e=t.assign(scale=1.); a,m=calibrate(t,c,np.zeros(len(c)),e,np.zeros(3));np.testing.assert_allclose(a[CANDIDATE],[2,10,50]); assert m['counts']=={0:60,1:60,2:60}

def test_block_interval_constant():
    assert block_ci(np.full(16,2.),np.repeat(np.arange(8),2))==[2.,2.]

def test_wrong_hash_rejected(tmp_path):
    from retail_v46 import prepare
    p=tmp_path/'no.zip';p.write_bytes(b'not the dataset')
    with pytest.raises(ValueError):prepare(p)

def test_globally_unrecorded_week_inserted_and_flagged():
    p=fixture(); missing=pd.Timestamp('2010-06-06'); p=p[p.date!=missing]; d,_=features(p); a=d[d.date==missing]; assert len(a)==6 and (a.demand==0).all() and not a.global_recorded_week.any()
