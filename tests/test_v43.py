import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from reality_v43 import choose,force_cap,dr_score,contributions,split,fixture,ARMS,load_real,interval,FEAT

def test_cap_and_positive():
    assert list(choose([3,2,1,-1],.5,.5,.3))==[1,1,0,0]
    assert choose([0,-1],1,.05,.3).sum()==0

def test_floor(): assert force_cap(np.arange(9),.25).sum()==2

def test_exact_nuisance():
    a=np.array([0,1,1,0]); p=np.array([[1.,3.]]*4); y=p[np.arange(4),a]
    np.testing.assert_allclose(dr_score(a,y,p),2)

def test_contribution_cost():
    a=np.array([0,1]); pred=np.array([[1.,3.]]*2); y=pred[np.arange(2),a]
    v,i=contributions(np.array([1,0]),a,y,pred,.1,.5); np.testing.assert_allclose(v,[.9,0])

def test_split():
    d=fixture(1000); tr,va,te=split(d,ARMS[1]); assert set(tr.index).isdisjoint(va.index) and set(tr.index).isdisjoint(te.index) and set(va.index).isdisjoint(te.index); assert set(te.segment)=={ARMS[0],ARMS[1]}

def test_unverified_data_rejected(tmp_path):
    p=tmp_path/'fake.csv.gz'; p.write_bytes(b'not real');
    with pytest.raises(ValueError,match='unverified'): load_real(p)

def test_interval(): assert interval(np.ones(10))==(1.,0.,1.,1.)

def test_no_outcome_feature(): assert not set(FEAT)&{'segment','spend','visit','conversion','a','exposure'}

def test_ipw_mean():
    a=np.array([0,1]*100); p=np.array([[1.,3.]]*200); y=p[np.arange(200),a]; v,i=contributions(np.ones(200),a,y,p,.1,.5); assert np.isclose(v.mean(),i.mean())
