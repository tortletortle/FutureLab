import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dataclasses import replace
import numpy as np
import pytest
from v36 import Config,archive,guarded
from v35 import C,adaptive
from benchmark_v36 import generate,score,DEV,HOLD

def small(scene='recurrence'):
    t,y,v,l=generate(19,scene);return t[None],y[None],v[None],l[None]

def test_prefix_causality():
    _,y,v,_=small();a=archive(y,v);b=archive(y[:,:23],v[:,:23]);np.testing.assert_allclose(a['pred'][:,:23],b['pred']);np.testing.assert_equal(a['event'][:,:23],b['event']);y[:,23:]+=30;v[:,23:]*=10;d=archive(y,v);np.testing.assert_allclose(a['pred'][:,:24],d['pred'][:,:24]);np.testing.assert_equal(a['event'][:,:23],d['event'][:,:23])

def test_guard_prefix():
    _,y,v,_=small();a=archive(y,v);p,l,_=adaptive(y,v);mu,w=guarded(a['pred'],a['loss'],p['adaptive_global'],l['adaptive_global']);a['loss'][:,20:]+=100;mu2,w2=guarded(a['pred'],a['loss'],p['adaptive_global'],l['adaptive_global']);np.testing.assert_allclose(mu[:,:21],mu2[:,:21]);np.testing.assert_allclose(w[:,:21],w2[:,:21])

def test_constant_learning():
    y=np.full((3,40,4),3.);v=np.full_like(y,.35);a=archive(y,v);assert np.all(a['pred'][:,-1]>2.9);assert (a['count']<=2).all();assert np.isfinite(a['loss']).all()

def test_recall_and_no_recall():
    y=np.tile(np.repeat([1.,5.,1.,5.],12)[None,:,None],(2,1,8));v=np.full_like(y,.2);a=archive(y,v);b=archive(y,v,replace(Config(),slots=1));assert (a['event']==2).sum()>0;assert (b['event']==2).sum()==0;assert (b['count']==1).all();assert (a['source'][a['event']==2]<np.broadcast_to(np.arange(y.shape[1]),a['source'].shape)[a['event']==2]).all()

def test_limited_capacity():
    _,y,v,_=small('capacity_pressure');a=archive(y,v,replace(Config(),slots=2));assert (a['count']<=2).all();assert (a['count']>=1).all();assert np.isfinite(a['pred']).all();assert (a['source'][a['event']==1]==np.where(a['event']==1)[1]).all()

def test_extreme_inputs_finite():
    y=np.tile(np.array([0.,100.,-100.,0.])[None,:,None],(2,1,6));v=np.full_like(y,1e-4);a=archive(y,v);assert np.isfinite(a['pred']).all() and np.isfinite(a['loss']).all();assert ((a['confidence']>=0)&(a['confidence']<=1+1e-10)).all()

@pytest.mark.parametrize('cfg',[Config(slots=0),Config(birth=.2,hazard=.1),Config(var_floor=-.1),Config(learn_gate=.2)])
def test_bad_config(cfg):
    with pytest.raises(ValueError):archive(np.ones((1,2,1)),np.ones((1,2,1)),cfg)

@pytest.mark.parametrize('v',[np.zeros((1,2,1)),np.full((1,2,1),np.nan),np.ones((1,2,2))])
def test_bad_observations(v):
    with pytest.raises(ValueError):archive(np.ones((1,2,1)),v)

def test_reproducible():
    for scene in HOLD:
        t,y,v,l=small(scene);a=archive(y,v);b=archive(y,v);np.testing.assert_equal(a['pred'],b['pred']);assert np.isfinite(t).all() and (v>0).all()

def test_return_and_novel_metrics():
    labels=np.array([[0]*4+[1]*4+[0]*4+[2]*4]);truth=np.full((1,16,1),C+1);p=np.full_like(truth,C-1);s=score(p,truth,labels);assert s['return_periods'][0]==4;assert s['novel_periods'][0]==8;assert s['return_regret'][0]==s['novel_regret'][0]==1

def test_novel_not_marked_return():
    t,y,v,l=small('nonrecurring');s=score(y,t,l);assert np.isnan(s['return_regret'][0]) and s['novel_periods'][0]==16

def test_guard_normalizes():
    _,y,v,_=small();a=archive(y,v);p,l,_=adaptive(y,v);mu,w=guarded(a['pred'],a['loss'],p['adaptive_global'],l['adaptive_global']);assert ((w>=0)&(w<=1)).all();np.testing.assert_allclose(mu,w[:,:,None]*a['pred']+(1-w[:,:,None])*p['adaptive_global'])
