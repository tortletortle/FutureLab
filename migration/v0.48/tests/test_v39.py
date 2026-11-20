import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pytest
from scipy.special import ndtri
from v35 import C
from v37 import Belief
from v38 import simulate as baseline,relative_risk
from v39 import simulate,audit_upper,adjusted_risk,AUDIT_UNIT_COST

def sample():
    r=np.random.default_rng(100);y=r.normal(C,1,(10,32,5));v=np.full_like(y,.625);p=r.normal(C,1,(*y.shape,1));a=r.normal(C,1,y.shape);av=np.full_like(y,.15);return y,v,p,v,a,av

def test_upper_formula():
    a=np.array([[1,0,0]]);r=np.array([[0,1,0]]);y=np.array([[C+.4,C+.2,np.nan]]);v=np.full((1,3),.15)
    assert audit_upper(a,r,y,v)[0]==pytest.approx(-.2/3+ndtri(.95)*np.sqrt(.3)/3)

def test_same_action_no_audit():
    a=np.array([[1,0]]);np.testing.assert_allclose(audit_upper(a,a,np.full_like(a,np.nan,dtype=float),np.ones_like(a)),0)

def test_audit_wrong_variance_and_missing_data():
    a=np.array([[1,0]]);r=1-a
    with pytest.raises(ValueError):audit_upper(a,r,np.full((1,2),np.nan),np.ones((1,2)))
    with pytest.raises(ValueError):audit_upper(a,r,np.ones((1,2)),np.zeros((1,2)))

def test_null_audit_size():
    # Unsafe true excess=.035+small; one-sided Gaussian bound has <= .05 false acceptance.
    rng=np.random.default_rng(223);n=100000;a=np.zeros((n,1));r=np.ones_like(a);y=C+.03501+rng.normal(0,.4,(n,1));u=audit_upper(a,r,y,np.full_like(a,.16));assert .045<(u<=.035).mean()<.055

def test_risk_envelope():
    b=Belief(np.array([[[C-.1,C+.1]]]),np.full((1,1,2),.2),np.zeros((1,1)),1);a=np.array([[1,0]],bool);r=~a
    p=relative_risk(b,a[:,None],r)[0][:,0];q=adjusted_risk(b,a,r,np.array([[.5,-.5]]),np.array([2.]))
    assert q[0]>=p[0];assert adjusted_risk(b,a,a,np.zeros((1,2)),np.ones(1))[0]==0

def test_baseline_exact():
    y,v,p,r,a,av=sample();x=simulate(y,v,p,r);z=baseline(y,v,p,r)
    for k in z:np.testing.assert_array_equal(x[k],z[k])

def test_audit_gate_and_cost():
    y,v,p,r,a,av=sample();x=simulate(y,v,p,r,a,av,audit_gate=True,threshold=0.)
    tested=x['audit_groups']>0;assert tested.any();assert (x['audit_groups'][~x['model_accept'].astype(bool)]==0).all();assert (x['audit_groups'][x['match'].astype(bool)]==0).all()
    np.testing.assert_allclose(x['audit_cost'],x['audit_groups']*AUDIT_UNIT_COST);np.testing.assert_array_equal(x['reuse'][tested].astype(bool),x['audit_upper'][tested]<=.035)

def test_audit_unobserved_irrelevant():
    y,v,p,r,a,av=sample();x=simulate(y,v,p,r,a,av,audit_gate=True,threshold=0.)
    observed=(x['audit_groups']>0)[:,:,None]&(x['candidate']!=x['reference']);aa=np.where(observed,a,np.nan);z=simulate(y,v,p,r,aa,av,audit_gate=True,threshold=0.)
    for k in x:np.testing.assert_array_equal(x[k],z[k])

def test_all_extensions_prefix_causal():
    y,v,p,r,a,av=sample();kw=dict(calibrate=True,quarantine=True,audit_gate=True,threshold=.5);x=simulate(y,v,p,r,a,av,**kw);z=simulate(y[:,:26],v[:,:26],p[:,:26],r[:,:26],a[:,:26],av[:,:26],**kw)
    for k in z:np.testing.assert_array_equal(x[k][:,:26],z[k])

def test_audits_do_not_change_learning():
    y,v,p,r,a,av=sample();x=simulate(y,v,p,r,a,av,audit_gate=True,threshold=0.,quarantine=True);z=simulate(y,v,p,r,quarantine=True,threshold=0.)
    for k in ('reference','candidate','selected','birth','quarantine'):np.testing.assert_array_equal(x[k],z[k])
