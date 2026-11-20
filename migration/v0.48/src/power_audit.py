"""Conditional power planning from frozen Hillstrom comparison (no refitting)."""
import argparse, csv, json, math
from statistics import NormalDist

NORM = NormalDist()

def audit(row, n, alpha=.05, power=.8, effects=(.10,.05,.02,.01)):
    point, lo, hi = (float(row[k]) for k in ('difference','lo','hi'))
    if n < 2 or not all(math.isfinite(x) for x in (point,lo,hi)) or not lo < point < hi:
        raise ValueError('Invalid interval or sample size')
    if not 0 < alpha < 1 or not .5 < power < 1:
        raise ValueError('Invalid design parameters')
    # The original report computed limits with 1.96, not an exact quantile.
    se = (hi-lo)/3.92
    if not math.isclose((hi+lo)/2,point,rel_tol=1e-7,abs_tol=1e-9):
        raise ValueError('Expected symmetric original normal interval')
    sigma = se*math.sqrt(n)
    z = NORM.inv_cdf(1-alpha/2)+NORM.inv_cdf(power)
    if any(not math.isfinite(d) or d<=0 for d in effects):
        raise ValueError('Effects must be positive finite values')
    return {'n_observed':n,'original_point':point,'original_ci':[lo,hi],
            'estimated_se':se,'estimated_sd':sigma,'alpha_two_sided':alpha,
            'target_power':power,'approx_minimum_detectable_effect':z*se,
            'required_independent_test_n':[{'effect':d,'n':math.ceil((z*sigma/d)**2)} for d in effects],
            'interpretation':'Approximate planning conditional on frozen policies and iid paired contributions; not new evidence or a prospective guarantee.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparison_csv'); p.add_argument('--n',type=int,required=True)
    args=p.parse_args()
    with open(args.comparison_csv,newline='',encoding='utf-8') as f:
        row=next(r for r in csv.DictReader(f) if r['arm']=='Mens E-Mail')
    print(json.dumps(audit(row,args.n),indent=2))

if __name__=='__main__': main()
