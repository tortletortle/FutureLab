import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pytest
from v37 import Belief,build_belief,update_all,update_probe,binary_gain,run_episode
from benchmark_v37 import generate,calibrate,metrics,DEV,HOLD
from scipy.special import logsumexp
from scipy.integrate import quad
from scipy.stats import norm

def data(scene='return',n=4,budget=5):
    return tuple(map(np.stack,zip(*(generate(100+i,scene,budget) for i in range(n)))))

@pytest.mark.parametrize('structured',[False,True])
def test_prior_and_update(structured):
    am,av,y,dv,bank,pv,*_=data();b=build_belief(am,av,structured);np.testing.assert_allclose(b.weights().sum(1),1);np.testing.assert_allclose(b.weights()[:,:4].sum(1),.5)
    before=b.var.copy();update_all(b,y,dv);assert (b.var<before).all();np.testing.assert_allclose(b.weights().sum(1),1)
    r=np.arange(4);before=b.var.copy();update_probe(b,r,bank[r,r,0],pv[r,r]);np.testing.assert_allclose(b.weights().sum(1),1);assert (b.var<=before).all()

def test_sparse_shape():
    am,av,*_=data();b=build_belief(am,av);assert b.mean.shape==(4,57,12);np.testing.assert_allclose(b.var[:,9:21]-av[:,0,None],np.broadcast_to(4*np.eye(12),(4,12,12)))

def test_gain_identical_zero():
    b=Belief(np.zeros((1,2,3)),np.ones((1,2,3)),np.log(np.array([[.4,.6]])),1)
    np.testing.assert_allclose(binary_gain(b,np.ones((1,3))),0,atol=1e-12)

def test_gain_quadrature():
    b=Belief(np.array([[[0.,0.],[1.,2.]]]),np.array([[[.4,.4],[.8,.8]]]),np.log(np.array([[.4,.6]])),1)
    r=np.ones((1,2));s=binary_gain(b,r,31)[0];assert s[1]>s[0]>0
    for j in range(2):
        m=b.mean[0,:,j];v=b.var[0,:,j]+1;w=np.array([.4,.6])
        def f(x):
            lp=norm.logpdf(x,m,np.sqrt(v));p=logsumexp(np.log(w)+lp);return np.sum(w*np.exp(lp)*(lp-p))
        val=quad(f,-20,20,epsabs=1e-10)[0];assert abs(val-s[j])<2e-7

@pytest.mark.parametrize('method',['active','random','uncertainty','disagreement','decision'])
def test_budget_and_prefix(method):
    am,av,y,dv,bank,pv,*_=data();a=run_episode(am,av,y,dv,bank,pv,method,5);b=run_episode(am,av,y,dv,bank,pv,method,2);np.testing.assert_equal(a['choices'][:,:2],b['choices']);np.testing.assert_allclose(a['history'][2]['mean'],b['history'][2]['mean']);assert (a['visits'].sum(1)==5).all();assert ((a['history'][-1]['known']>=0)&(a['history'][-1]['known']<=1+1e-12)).all()

def test_unused_values_do_not_enter_policy():
    am,av,y,dv,bank,pv,*_=data();a=run_episode(am,av,y,dv,bank,pv,'active',3);w=bank.copy()
    for i in range(len(w)):
        for j in range(w.shape[1]):w[i,j,a['visits'][i,j]:]=np.nan
    b=run_episode(am,av,y,dv,w,pv,'active',3);np.testing.assert_equal(a['choices'],b['choices']);np.testing.assert_allclose(a['history'][-1]['mean'],b['history'][-1]['mean'])

def test_bad_variance_and_budget():
    am,av,y,dv,bank,pv,*_=data()
    with pytest.raises(ValueError):run_episode(am,av,y,dv,bank,pv,'active',6)
    with pytest.raises(ValueError):build_belief(am,-av)
    with pytest.raises(ValueError):run_episode(am,av,y,dv,bank,pv*0,'active',2)

def test_dev_hold_seeds_disjoint_and_new_scene():
    assert set(DEV)<set(HOLD);a={1270000+10000*i+j for i in range(len(DEV)) for j in range(120)};b={2370000+10000*i+j for i in range(len(HOLD)) for j in range(300)};assert not a&b

def test_abstain_and_risk():
    import pandas as pd
    d=pd.DataFrame(dict(method=['m']*4,confidence=[.9,.8,.7,.6],truth_slot=[0,1,-1,-1],candidate=[0,0,1,2],regret=[0]*4,brier=[0]*4,logloss=[0]*4));s=metrics(d,.8);assert s['novel_far']==0 and s['correct_known_recall']==.5 and s['wrong_known_recall']==.5;assert calibrate(d)['m']<=.9
