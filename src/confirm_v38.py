"""Exploratory holdout led to fixed 0.90/0.95 thresholds; confirm with new seeds."""
import numpy as np,pandas as pd
import benchmark_v38 as b

def main():
    b.ARMS=[('evsi','evsi','risk',1,True),('identity','evsi','identity',1,True)]
    d=b.collect(b.HOLD38,80,9380000);d.to_csv(b.OUT/'confirmation_periods.csv.gz',index=False)
    rows=[];sc=[]
    for name,g in d.groupby('method'):
        for t in [.9,.95]:
            rows.append(dict(method=name,**b.measure(g,t)))
            for scene,f in g.groupby('scene'):sc.append(dict(method=name,scene=scene,**b.measure(f,t)))
    s=pd.DataFrame(rows);s.to_csv(b.OUT/'confirmation_summary.csv',index=False);pd.DataFrame(sc).to_csv(b.OUT/'confirmation_scenes.csv',index=False)
    print(s[['method','threshold','coverage','distinct_reuse','unsafe_given_distinct','regret']].to_string(index=False),flush=True)
if __name__=='__main__':main()
