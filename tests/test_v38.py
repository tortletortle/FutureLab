import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from scipy.special import ndtr
from v35 import C
from v37 import Belief
from v38 import relative_risk,oracle_risk,decision_gain,select_candidate,simulate,working_belief,DEFAULT_ARCHIVE

def belief(m,v=.4):
    a=np.array(m,float)[None,None,:]+C;return Belief(a,np.full_like(a,v),np.zeros((1,1)),1)

def test_exact_normal_risk():
    b=belief([.2,-.3]);p,e=relative_risk(b,np.array([[[0,1]]]),np.array([[1,0]]),.1)
    assert e[0,0]==pytest.approx(.25);assert p[0,0]==pytest.approx(ndtr(.15/np.sqrt(.2)))

def test_identical_policy_zero_loss():
    b=belief([.2,-.3]);p,e=relative_risk(b,np.array([[[1,0]]]),np.array([[1,0]]),0)
    assert p[0,0]==e[0,0]==0

def test_oracle_risk_mc():
    b=belief([.2,-.3]);a=np.array([[[1,0]]]);rng=np.random.default_rng(8);x=rng.normal([.2,-.3],np.sqrt(.4),(150000,2));r=(np.maximum(x,0)-a[0,0]*x).mean()
    assert oracle_risk(b,a)[0,0]==pytest.approx(r,abs=.003)

def test_evsi_prefers_boundary():
    b=belief([0,2]);g=decision_gain(b,np.ones((1,2)));assert g[0,0]>g[0,1]>0

def test_evsi_closed_form():
    b=belief([0.]);s=.4/np.sqrt(1.4);assert decision_gain(b,np.ones((1,1)))[0,0]==pytest.approx(s/np.sqrt(2*np.pi))

def test_active_mask():
    b=belief([.2,-.3]);j,c,ref,r=select_candidate(b,np.array([[[0,1],[1,0]]]),np.array([[True,False]]));assert j[0]==0

def inputs():
    rng=np.random.default_rng(9);y=rng.normal(C,1,(3,22,5));v=np.ones_like(y);bank=rng.normal(C,1,(*y.shape,2));return y,v,bank,v.copy()

def test_budget():
    y,v,p,r=inputs();a=simulate(y,v,p,r,budget=2);assert (a['probes'].sum(1)==6).all();assert (a['selected'][:,:12]==-1).all()

def test_prefix_causality():
    y,v,p,r=inputs();a=simulate(y,v,p,r);b=simulate(y[:,:17],v[:,:17],p[:,:17],r[:,:17]);assert np.array_equal(a['decision'][:,:17],b['decision']);assert np.array_equal(a['selected'][:,:17],b['selected'])

def test_unseen_probes_ignored():
    y,v,p,r=inputs();a=simulate(y,v,p,r);q=p.copy();mask=np.ones(q.shape,bool)
    for i in range(len(y)):
        for t,j in enumerate(a['selected'][i,:,0]):
            if j>=0:mask[i,t,j,0]=False
    q[mask]=np.nan;b=simulate(y,v,q,r);assert np.array_equal(a['decision'],b['decision']);assert np.array_equal(a['selected'],b['selected'])

def test_threshold_does_not_change_collection():
    y,v,p,r=inputs();a=simulate(y,v,p,r,threshold=0);b=simulate(y,v,p,r,threshold=1.000001);assert np.array_equal(a['selected'],b['selected']);assert np.array_equal(a['reference'],b['reference']);assert not b['reuse'].any()

def test_invalid_variances():
    y,v,p,r=inputs();r[0,0,0]=0
    with pytest.raises(ValueError):simulate(y,v,p,r)

def test_belief_normalization():
    m=np.ones((2,3,4));v=m.copy();active=np.array([[1,0,0],[1,1,0]],bool);prob=np.array([[1,0,0],[.3,.7,0]]);b=working_belief(m,v,active,prob,DEFAULT_ARCHIVE);np.testing.assert_allclose(b.weights().sum(1),1);assert (b.weights()[:,:3][~active]==0).all()
