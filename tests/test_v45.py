import sys, math, json, shutil
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from value_v45 import Estimate,economics,from_locked_policy,locked_readout,external_scenarios


def test_economics_exact():
    r=economics(Estimate(.77,.49,1.05),10000,2500,.05,.3)
    assert r['incremental_net_contribution']['point']==pytest.approx(452.5)
    assert r['incremental_net_contribution']['lo']==pytest.approx(242.5)
    assert r['net_roi']['point']==pytest.approx(3.62)
    assert r['marketing_cost']==125
    assert r['break_even_cost_per_contact']['point']==pytest.approx(.231)

@pytest.mark.parametrize('estimate',[(1,2,3),(1,0,.5),(float('nan'),0,1)])
def test_bad_estimate(estimate):
    with pytest.raises(ValueError):Estimate(*estimate)

@pytest.mark.parametrize('contacts,cost,margin',[(11,.05,.3),(1,-1,.3),(1,.05,1.1),(1,math.inf,.3)])
def test_invalid_economics(contacts,cost,margin):
    with pytest.raises(ValueError):economics(Estimate(1,0,2),10,contacts,cost,margin)

def test_roi_no_cost():
    assert economics(Estimate(1,0,2),10,5,0,.3)['net_roi'] is None

def test_no_contacts():
    r=economics(Estimate(1,.5,2),100,0,.05,.5)
    assert r['economic_evidence']=='no_contacts' and r['net_roi'] is None
    assert r['incremental_net_contribution']=={'point':0,'lo':0,'hi':0}

def test_risk_gate():
    assert economics(Estimate(1,.1,2),100,50,.2,.3)['economic_evidence']=='inconclusive'
    assert economics(Estimate(.1,0,.2),100,50,.2,.3)['economic_evidence']=='negative_upper_bound'

def test_inversion():
    r=from_locked_policy({'coverage':.25,'n':100,'aipw':.04525,'lo':.02425,'hi':.06625},.05,.3)
    assert r['incremental_net_contribution']['point']==pytest.approx(452.5)
    assert r['net_roi']['point']==pytest.approx(3.62)

def test_fixture_readout():
    r=locked_readout(ROOT/'results/v43/synthetic_fixture')
    assert r['mode']=='synthetic_fixture'
    assert len(r['comparisons'])==2
    assert all(not c['eligible_for_prospective_trial'] and not c['protocol_pass'] for c in r['comparisons'])

def test_unverified_real(tmp_path):
    d=tmp_path/'f';shutil.copytree(ROOT/'results/v43/synthetic_fixture',d)
    m=json.loads((d/'manifest.json').read_text());m['mode']='real';(d/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(FileNotFoundError):locked_readout(d)

def test_frozen_selection(tmp_path):
    d=tmp_path/'f';shutil.copytree(ROOT/'results/v43/synthetic_fixture',d)
    c=d/'comparison.csv';c.write_text(c.read_text().replace('dr_extra_trees','random'))
    with pytest.raises(ValueError,match='frozen'):locked_readout(d)

def test_external_scenarios():
    r=external_scenarios(ROOT/'docs/V45_EXTERNAL_AGGREGATE.json')
    assert len(r['scenarios'])==72
    assert r['mode']=='external_published_aggregate'
    assert all('eligible_for_prospective_trial' not in x for x in r['scenarios'])

def test_reject_unknown_real_digest(tmp_path):
    d=tmp_path/'f';shutil.copytree(ROOT/'results/v43/synthetic_fixture',d)
    m=json.loads((d/'manifest.json').read_text()); m.update(mode='real',source_sha256='fake');(d/'manifest.json').write_text(json.dumps(m))
    (d/'provenance.json').write_text(json.dumps({'status':'completed','data':{'sha256':'fake'}}))
    with pytest.raises(ValueError,match='unverified'):locked_readout(d)


def test_reject_changed_protocol(tmp_path):
    from value_v45 import LOCKED_SHA
    d=tmp_path/'f';shutil.copytree(ROOT/'results/v43/synthetic_fixture',d)
    m=json.loads((d/'manifest.json').read_text());m.update(mode='real',source_sha256=LOCKED_SHA);(d/'manifest.json').write_text(json.dumps(m))
    (d/'provenance.json').write_text(json.dumps({'status':'completed','data':{'sha256':LOCKED_SHA},'protocol_sha256':'fake'}))
    with pytest.raises(ValueError,match='fingerprint'):locked_readout(d)
