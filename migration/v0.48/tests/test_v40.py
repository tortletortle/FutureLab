import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pytest
from v40 import audit,potential_gain
from v39 import audit_upper,simulate
from v35 import C
from v37 import Belief

def test_fixed_equivalence():
    rng=np.random.default_rng(12);a=rng.integers(0,2,(100,5));r=1-a;v=np.full(a.shape,.45);z=rng.normal(C,.45**.5,(*a.shape,3));q=audit(a,r,z,v,method='fixed')
    np.testing.assert_array_equal(q['accept'],audit_upper(a,r,z.mean(2),v/3)<=.035)
    assert (q['rounds']==3).all()

def test_stop_and_unobserved():
    a=np.array([[0,1,0],[1,1,0],[0,0,0]]);r=np.array([[1,1,0],[1,1,0],[1,0,0]])
    z=np.full((3,3,3),np.nan);z[0,0,0]=C-20;z[2,0,0]=C+20
    for method in ('spend','mixture'):
        q=audit(a,r,z,np.full(a.shape,.45),method=method)
        np.testing.assert_array_equal(q['accept'],[True,True,False]);np.testing.assert_array_equal(q['rounds'],[1,0,1]);np.testing.assert_allclose(q['cost'],[.0015,0,.0015])

def test_screen():
    a=np.array([[0,1],[1,1]]);r=1-a;z=np.full((2,2,3),np.nan);q=audit(a,r,z,np.ones_like(a),value_bound=np.zeros(2))
    assert not q['accept'].any() and q['screened'].all() and q['cost'].sum()==0

@pytest.mark.parametrize('method',['fixed','spend','mixture','peek'])
def test_boundary_size(method):
    rng=np.random.default_rng(907);n=80000;a=np.zeros((n,1));r=1-a;v=np.full_like(a,.45);z=C+.035+1e-8+rng.normal(0,.45**.5,(n,1,3));q=audit(a,r,z,v,method=method,early_reject=False);rate=q['accept'].mean()
    if method=='peek':assert rate>.075
    else:assert rate<.055

def test_potential_gain():
    b=Belief(np.array([[[C]]]),np.array([[[1.]]]),np.zeros((1,1)),1);a=np.array([[1]]);r=1-a
    assert potential_gain(b,a,r)[0]==pytest.approx(1/np.sqrt(2*np.pi));assert potential_gain(b,a,a)[0]==0

def test_invalid():
    a=np.zeros((1,1));r=1-a;z=np.ones((1,1,3));v=np.ones_like(a)
    for kw in [dict(alpha=0),dict(max_rounds=4),dict(unit_cost=-1)]:
        with pytest.raises(ValueError):audit(a,r,z,v,**kw)
    with pytest.raises(ValueError):audit(a,r,z,v*0)
    with pytest.raises(ValueError):audit(a,r,z*np.nan,v)

def test_diagnostic_and_soft_prefix():
    rng=np.random.default_rng(90);y=rng.normal(C,1,(4,28,5));v=np.full_like(y,.625);p=rng.normal(C,1,(*y.shape,1));x=simulate(y,v,p,v);q=simulate(y,v,p,v,diagnostic=potential_gain)
    for k in x:np.testing.assert_array_equal(x[k],q[k])
    a=simulate(y,v,p,v,diagnostic=potential_gain,soft_write=True);b=simulate(y[:,:20],v[:,:20],p[:,:20],v[:,:20],diagnostic=potential_gain,soft_write=True)
    for k in b:np.testing.assert_array_equal(a[k][:,:20],b[k])
