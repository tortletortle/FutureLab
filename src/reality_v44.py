"""V0.44: verified, fail-closed acquisition and audit of the locked V43 benchmark."""
from dataclasses import dataclass
from pathlib import Path
import argparse, hashlib, json, os, platform, tempfile, urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pandas as pd
import reality_v43 as v43

@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    compressed: bool

SOURCES = (
    Source('sklift-mirror',v43.URL,v43.SHA,True),
    Source('publisher-csv','https://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv','0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece',False),
    Source('github-csv','https://raw.githubusercontent.com/AdityaDabrase/ab-testing-email-marketing/main/data/raw/hillstrom.csv','0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece',False),
)
COUNTS = {'No E-Mail':21306,'Mens E-Mail':21307,'Womens E-Mail':21387}
MAX_BYTES = 12_000_000


def identify(path, sources=SOURCES):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES: raise ValueError('raw file too large')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    source=next((s for s in sources if s.sha256==digest),None)
    if source is None: raise ValueError('unrecognized SHA256; refusing evaluation')
    return source,digest


def load_verified(path, sources=SOURCES):
    source,digest=identify(path,sources)
    df=pd.read_csv(path,compression='gzip' if source.compressed else None)
    df.columns=df.columns.str.strip().str.lower()
    if len(df)!=64000 or not set(v43.FEAT+['segment','spend','visit','conversion']).issubset(df):
        raise ValueError('invalid row count or schema')
    if df.segment.value_counts().to_dict()!=COUNTS: raise ValueError('unexpected arm counts')
    v43.validate(df)
    for field in ['mens','womens','newbie','visit','conversion']:
        if not df[field].isin([0,1]).all(): raise ValueError('invalid binary: '+field)
    if not df.recency.between(1,12).all() or (df.history<0).any(): raise ValueError('invalid pre-treatment fields')
    if ((df.spend>0)&(df.conversion!=1)).any(): raise ValueError('positive spend without conversion')
    # No row sorting, deduplication, category rewriting or feature reinterpretation.
    meta={'sha256':digest,'source_format':'gzip-csv' if source.compressed else 'csv','bytes':Path(path).stat().st_size,
          'rows':len(df),'arm_counts':COUNTS,'positive_spend_rows':int((df.spend>0).sum())}
    return df,meta


def acquire(directory, sources=SOURCES, opener=urllib.request.urlopen):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    attempts=[]
    for source in sources:
        dest=directory/('hillstrom.csv.gz' if source.compressed else 'hillstrom.csv')
        if dest.exists():
            try:
                _,meta=load_verified(dest,sources)
                return dest,attempts+[{'source':source.name,'status':'verified cached','sha256':meta['sha256']}]
            except Exception as e: attempts.append({'source':source.name,'status':'cache rejected','error':str(e)})
        fd,tmpname=tempfile.mkstemp(dir=directory,prefix='.download-'); tmp=Path(tmpname)
        try:
            size=0
            with os.fdopen(fd,'wb') as out, opener(source.url,timeout=15) as response:
                while chunk:=response.read(65536):
                    size+=len(chunk)
                    if size>MAX_BYTES: raise ValueError('download too large')
                    out.write(chunk)
            _,meta=load_verified(tmp,(source,))
            if dest.exists(): raise FileExistsError(f'will not overwrite rejected cache: {dest}')
            tmp.replace(dest)
            return dest,attempts+[{'source':source.name,'status':'verified download','sha256':meta['sha256']}]
        except Exception as e: attempts.append({'source':source.name,'status':'failed','error':str(e)})
        finally: tmp.unlink(missing_ok=True)
    return None,attempts


def run(path, output):
    output=Path(output)
    if output.exists(): raise FileExistsError('choose a new output directory: holdout cannot be overwritten')
    df,meta=load_verified(path)
    # Keep protocol and benchmark code unchanged. Never fit on the test split.
    output.mkdir(parents=True,exist_ok=False)
    provenance={'data':meta,'protocol_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'docs/V43_PROTOCOL.md').read_bytes()).hexdigest(),
      'benchmark_sha256':hashlib.sha256(Path(v43.__file__).read_bytes()).hexdigest(),
      'versions':{p:version(p) for p in ['numpy','scipy','pandas','scikit-learn','xgboost']},'python':platform.python_version(),
      'created_utc':datetime.now(timezone.utc).isoformat(),'status':'started','locked_v43_input':meta['sha256']==v43.SHA,
      'caution':'CSV hash from third-party documentation; CSV vs compressed mirror row equivalence not independently established.'}
    p=output/'provenance.json'; p.write_text(json.dumps(provenance,indent=2))
    try:
        v43.benchmark(df,output,'real')
        manifest=output/'manifest.json'; m=json.loads(manifest.read_text()); m['source_sha256']=meta['sha256']; m['locked_v43_input']=meta['sha256']==v43.SHA; manifest.write_text(json.dumps(m,indent=2,ensure_ascii=False))
        if meta['sha256']!=v43.SHA:
            comparison=output/'comparison.csv'; c=pd.read_csv(comparison); c['confirmatory']=False; c.to_csv(comparison,index=False)
        provenance['status']='completed'
    except Exception:
        provenance['status']='failed'; raise
    finally: p.write_text(json.dumps(provenance,indent=2))


def main():
    p=argparse.ArgumentParser(); x=p.add_mutually_exclusive_group(required=True); x.add_argument('--download',action='store_true'); x.add_argument('--data',type=Path)
    p.add_argument('--cache',type=Path,default=Path('data')); p.add_argument('--output',type=Path,default=Path('results/v44/real')); p.add_argument('--attempt-log',type=Path,default=Path('results/v44/acquisition.json')); a=p.parse_args()
    if a.output.exists(): p.error('output already exists')
    path=a.data
    if a.download:
        path,attempts=acquire(a.cache); a.attempt_log.parent.mkdir(parents=True,exist_ok=True); a.attempt_log.write_text(json.dumps(attempts,indent=2))
        if path is None: p.exit(2,'No verified data; real benchmark NOT run. See acquisition log.\n')
    run(path,a.output)
if __name__=='__main__': main()
