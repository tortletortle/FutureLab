"""Recompute reported ordered quantities' proxy costs without fitting models."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def audit(path):
    path=Path(path); m=json.loads((path/'manifest.json').read_text()); detail=pd.read_csv(path/'test_orders.csv'); metrics=pd.read_csv(path/'metrics.csv'); keys=['series_id','date','method']; assert not detail.duplicated(keys).any()
    assert (detail.order>=0).all() and (detail.order==np.rint(detail.order)).all()
    actual=detail.pivot(index=['series_id','date'],columns='method',values='demand'); assert actual.notna().all().all() and (actual.nunique(axis=1)==1).all()
    reported=metrics[metrics.split=='test'].set_index('method'); assert set(reported.index)==set(detail.method)
    costs=3*(detail.demand-detail.order).clip(lower=0)+(detail.order-detail.demand).clip(lower=0); np.testing.assert_allclose(costs,detail.cost)
    means=detail.assign(recomputed=costs).groupby('method').recomputed.mean(); np.testing.assert_allclose(means,reported.loc[means.index,'mean_cost'])
    selection=metrics[(metrics.split=='selection')&(metrics.method!=m['candidate'])]; assert selection.loc[selection.mean_cost.idxmin(),'method']==m['champion']
    gap=means[m['candidate']]-means[m['champion']]; assert np.isclose(gap,m['paired_cost_delta'])
    n=m['splits']['test']['n']; assert len(actual)==n; assert int(actual.shape[1])==len(reported)
    return {'status':'passed','methods':len(means),'test_product_weeks':n,'audited_decision_rows':len(detail),'paired_delta_recomputed':float(gap),'sha256':{f:hashlib.sha256((path/f).read_bytes()).hexdigest() for f in ('test_orders.csv','metrics.csv','manifest.json')}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();print(json.dumps(audit(a.directory),indent=2))
