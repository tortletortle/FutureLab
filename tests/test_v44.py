import sys, hashlib, io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import pytest
import reality_v44 as m

@pytest.fixture(scope='module')
def mock_dataset(tmp_path_factory):
    path=tmp_path_factory.mktemp('v44')/'fixture.csv'
    # Deliberately fabricated only to test the acquisition/schema gates.
    d=pd.DataFrame({'segment':sum(([a]*n for a,n in m.COUNTS.items()),[]),'recency':1,'history':100.,'mens':0,'womens':1,'newbie':0,'history_segment':'test','zip_code':'Rural','channel':'Web','spend':0.,'visit':0,'conversion':0})
    d.to_csv(path,index=False); raw=path.read_bytes()
    return path,raw,m.Source('test','https://example.invalid/test.csv',hashlib.sha256(raw).hexdigest(),False)

def test_hash_reject(tmp_path):
    p=tmp_path/'fake'; p.write_bytes(b'fake')
    with pytest.raises(ValueError,match='unrecognized'): m.load_verified(p)

def test_schema_verified(mock_dataset):
    path,_,s=mock_dataset; d,meta=m.load_verified(path,(s,)); assert len(d)==64000 and meta['positive_spend_rows']==0

def test_failover(tmp_path,mock_dataset):
    _,raw,s=mock_dataset
    bad=m.Source('bad','https://example.invalid/bad','0'*64,False)
    def open_(url,timeout): return io.BytesIO(b'bad' if url==bad.url else raw)
    p,log=m.acquire(tmp_path,(bad,s),open_); assert p.exists() and [r['status'] for r in log]==['failed','verified download']
    assert not list(tmp_path.glob('.download-*'))

def test_all_fail(tmp_path):
    def open_(url,timeout): raise OSError('offline')
    p,log=m.acquire(tmp_path,m.SOURCES,open_); assert p is None and len(log)==3 and not list(tmp_path.iterdir())

def test_rejected_cache_preserved(tmp_path,mock_dataset):
    _,raw,s=mock_dataset; target=tmp_path/'hillstrom.csv'; target.write_bytes(b'invalid')
    p,log=m.acquire(tmp_path,(s,),lambda *a,**kw:io.BytesIO(raw)); assert p is None and target.read_bytes()==b'invalid'

def test_wrong_counts_rejected(tmp_path,mock_dataset):
    p,_,_=mock_dataset; d=pd.read_csv(p); d.loc[0,'segment']='Mens E-Mail'; q=tmp_path/'altered.csv'; d.to_csv(q,index=False)
    s=m.Source('test','https://example.invalid',hashlib.sha256(q.read_bytes()).hexdigest(),False)
    with pytest.raises(ValueError,match='arm counts'): m.load_verified(q,(s,))

def test_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError): m.run('missing.csv',tmp_path)

def test_oversize(tmp_path):
    with pytest.raises(ValueError,match='too large'):
        p=tmp_path/'huge'; p.write_bytes(b'0'*(m.MAX_BYTES+1)); m.identify(p)
