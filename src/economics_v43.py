"""Reanalyse existing V0.42 trajectories. Savings per accepted differing policy are counterfactual, not measured."""
from pathlib import Path
import numpy as np
import pandas as pd

def analyse(path, output):
    d=pd.read_csv(path); d=d[d.stress=='clean'].copy()
    d['cost']=d.regret+d.probe_cost+d.audit_cost
    ix=['scene','seed']; p=d.pivot(index=ix,columns='method',values='cost'); a=d.pivot(index=ix,columns='method',values='accepted'); n=d.groupby(ix).periods.first().reindex(p.index).to_numpy(); rng=np.random.default_rng(431903)
    # Stratified bootstrap over independent trajectories, preserving equal scene mix.
    strata=[np.flatnonzero(p.index.get_level_values(0)==s) for s in p.index.get_level_values(0).unique()]
    idx=np.concatenate([rng.choice(s,size=(2000,len(s)),replace=True) for s in strata],axis=1)
    rows=[]
    for m in p.columns:
        delta=(p[m]-p.recompute).to_numpy(); accepts=a[m].to_numpy(); boot=delta[idx].sum(1)/n[idx].sum(1); denom=accepts.sum()
        rows.append(dict(method=m,trajectories=len(p),periods=int(n.sum()),cost_per_period=p[m].sum()/n.sum(),extra_cost_total=delta.sum(),extra_cost_per_period=delta.sum()/n.sum(),paired_lo=np.quantile(boot,.025),paired_hi=np.quantile(boot,.975),accepted_different=int(denom),break_even_saving_per_accept=max(0.,delta.sum()/denom) if denom else None))
    out=pd.DataFrame(rows); out.to_csv(output,index=False); return out
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]; print(analyse(root/'results/v42/trajectories.csv',root/'results/v43/economics.csv').to_string(index=False))
