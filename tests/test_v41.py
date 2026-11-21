import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pytest
from scipy.stats import t
from v41 import evidence_gate
from v40 import audit
from v35 import C
from benchmark_v41 import generate,gate


def test_known_exact_v40():
    rng=np.random.default_rng(14);x=rng.normal(.02,.2,(120,1,3));g=np.ones(120,int);sd=np.full(120,.2)
    a=evidence_gate(x,None,sd,g,mode='known',paired=False,looks=(1,2,3));old=audit(np.zeros((120,1)),np.ones((120,1)),x+C,sd[:,None]**2)
    np.testing.assert_array_equal(a['accept'],old['accept']);np.testing.assert_array_equal(a['samples'],old['samples'])


def test_t_closed_form():
    x=np.array([[[.02,.04,.07,.1]]]);a=evidence_gate(x,None,np.ones(1),np.ones(1),mode='single',paired=False,looks=(4,))
    assert a['upper'][0]==pytest.approx(x.mean()+t.ppf(.95,3)*x.std(ddof=1)/2)


def test_matched_offset_invariance():
    rng=np.random.default_rng(98);x=rng.normal(size=(100,3,6));c=rng.normal(size=x.shape);b=rng.normal(size=(100,3,1))*5
    a=gate('robust_paired6',x,c,np.ones(100),np.ones(100));q=gate('robust_paired6',x+b,c+b,np.ones(100),np.ones(100));np.testing.assert_array_equal(a['accept'],q['accept']);np.testing.assert_allclose(a['upper'],q['upper'])


def test_single_corrupt_bound_and_cost():
    x=np.array([[[-20,-21,-19],[.3,.4,.5],[.3,.4,.5]]]);a=evidence_gate(x,None,np.ones(1),np.array([2]),mode='robust',paired=False,looks=(3,));assert not a['accept'][0];assert a['samples'][0]==18


def test_early_stop_unread():
    x=np.full((2,3,6),np.nan);x[0,:,:3]=[-21.,-20.,-19.];x[1,:,:3]=[19.,20.,21.];c=np.zeros_like(x);a=gate('robust_paired6',x,c,np.ones(2),np.array([2,3]));np.testing.assert_array_equal(a['accept'],[True,False]);np.testing.assert_array_equal(a['samples'],[36,54]);np.testing.assert_allclose(a['cost'],[.054,.081])


def test_degenerate_refused():
    x=np.full((1,3,6),-100.);a=gate('robust_t6',x,None,np.ones(1),np.ones(1));assert not a['accept'][0] and np.isinf(a['upper'][0])

@pytest.mark.parametrize('stress',['clean','variance4','matched_bias','one_bad','matched_plus_one_bad','source_shared'])
def test_boundary_robust(stress):
    n=15000;sd=np.full(n,.1);x,c=generate(np.full(n,.0350001),sd,np.random.default_rng(129),stress);a=gate('robust_paired6',x,c,sd,np.ones(n));assert a['accept'].mean()<.055


def test_inputs():
    x=np.ones((2,3,6));sd=np.ones(2);g=np.ones(2)
    for kw in [dict(looks=(1,2)),dict(looks=(3,3)),dict(alpha=0),dict(unit_cost=-1)]:
        with pytest.raises(ValueError):evidence_gate(x,x,sd,g,**kw)
    with pytest.raises(ValueError):evidence_gate(x,None,sd,g)
    with pytest.raises(ValueError):evidence_gate(x,x,sd,g*0)
