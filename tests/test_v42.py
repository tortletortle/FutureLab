import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from v42 import Source,route_gate

def data():
    x=np.tile(np.array([-.3,-.31,-.29,-.3,-.31,-.29,-.3,-.31,-.29]),(2,4,1));return x,np.zeros_like(x),np.ones(2,int)
def test_duplicate_domains_never_vote_twice():
    x,c,g=data(); s=[Source('a',0),Source('a',0),Source('b',0),Source('c',0)];r=route_gate(x,c,g,s);assert r['accept'].all() and (r['counts'][:,1]==0).all()
def test_no_certification_declines():
    x,c,g=data();r=route_gate(x,c,g,[Source('a',None)]*4);assert r['declined'].all() and not r['accept'].any() and not r['samples'].any()
def test_unknown_fault_budget():
    x,c,g=data();r=route_gate(x,c,g,[Source(str(i),0) for i in range(4)],fault_budget=None);assert r['declined'].all()
def test_inadequate_domains():
    x,c,g=data();r=route_gate(x,c,g,[Source('a',0)]*4);assert r['declined'].all()
def test_stop_never_reads_later_nan():
    x,c,g=data();x[:,:,3:]=np.nan;r=route_gate(x,c,g,[Source(str(i),0) for i in range(4)]);assert r['accept'].all() and (r['samples']==12).all()
def test_budget_strict():
    x,c,g=data();r=route_gate(x,c,g,[Source(str(i),0) for i in range(4)],budget=11);assert not r['accept'].any() and (r['samples']<=11).all()
def test_bias_sensitivity():
    x,c,g=data();s=[Source(str(i),1) for i in range(4)];assert not route_gate(x,c,g,s)['accept'].any()
def test_fault_budget_two():
    x,c,g=data();r=route_gate(x,c,g,[Source(str(i),0) for i in range(4)],fault_budget=2);assert r['accept'].all() and (r['samples']==18).all()
def test_reject_invalid():
    x,c,g=data();with_bad=[Source('x',-1)]*4
    with pytest.raises(ValueError):route_gate(x,c,g,with_bad)
def test_zero_variance():
    x,c,g=data();x[:]=-.3;assert not route_gate(x,c,g,[Source(str(i),0) for i in range(4)])['accept'].any()
def test_cost_scales():
    x,c,g=data();r=route_gate(x,c,g*3,[Source(str(i),0) for i in range(4)]);assert np.allclose(r['cost'],r['samples']*.0015) and (r['samples']==36).all()
