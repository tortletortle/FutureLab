"""Economic readout of a frozen policy, not a model or a test-set selector."""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKED_SHA = "bab6578f60db5d792f1c2372c502f029152a5249cf5ea84390f3b7f885d7234f"
FALLBACK_SHA = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.point,self.lo,self.hi)) or not self.lo <= self.point <= self.hi:
            raise ValueError('invalid estimate or interval')

    def affine(self, scale: float, shift: float = 0) -> Estimate:
        if not math.isfinite(scale) or not math.isfinite(shift) or scale < 0:
            raise ValueError('invalid nonnegative affine transform')
        return Estimate(scale*self.point+shift,scale*self.lo+shift,scale*self.hi+shift)


def economics(spend_per_contact: Estimate, cohort: int, contacts: int, cost: float, margin: float):
    """Translate an incremental revenue interval, conditional on fixed contacts and costs.

    Not a prediction interval for future realized revenue. Cost excludes fixed overhead.
    """
    if not isinstance(cohort,int) or not isinstance(contacts,int) or not 0<=contacts<=cohort or cohort<=0:
        raise ValueError('invalid contact count')
    if not math.isfinite(cost) or cost<0 or not math.isfinite(margin) or not 0<=margin<=1:
        raise ValueError('invalid unit economics')
    budget=contacts*cost
    net=spend_per_contact.affine(contacts*margin,-budget)
    return {'cohort':cohort,'contacts':contacts,'coverage':contacts/cohort,'margin':margin,'unit_cost':cost,
            'marketing_cost':budget,'incremental_revenue':asdict(spend_per_contact.affine(contacts)),
            'incremental_net_contribution':asdict(net),'net_roi':asdict(net.affine(1/budget)) if budget>0 else None,
            'break_even_cost_per_contact':asdict(spend_per_contact.affine(margin)) if contacts else None,
            'economic_evidence': 'no_contacts' if not contacts else 'positive_lower_bound' if net.lo>0 else 'negative_upper_bound' if net.hi<0 else 'inconclusive'}


def from_locked_policy(record, cost, margin, cohort=10000):
    """Invert the locked AIPW net estimate to an effect per contacted unit.

    Only valid at the original policy and economics; changing cost/margin may change actions.
    """
    coverage=float(record['coverage']); n=int(record['n'])
    if not 0<=coverage<=1 or n<1 or not 0<margin<=1: raise ValueError('invalid coverage, sample size, or margin')
    if abs(coverage*n-round(coverage*n))>1e-6: raise ValueError('fractional observed contact')
    net=Estimate(*(float(record[k]) for k in ('aipw','lo','hi')))
    if coverage == 0:
        if net != Estimate(0,0,0): raise ValueError('nonzero value with no contacts')
        effect=net
    else:
        effect=net.affine(1/(margin*coverage),cost/margin)
    out=economics(effect,cohort,math.floor(coverage*cohort+1e-9),cost,margin)
    out.update({'holdout_n':n,'holdout_coverage':coverage,'holdout_net_per_eligible':asdict(net)})
    return out


def _rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def locked_readout(directory: Path, cohort=10000):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    mode=manifest['mode']
    if mode not in ('real','synthetic_fixture'):raise ValueError('unknown evaluation mode')
    if mode=='real':
        # Fail closed for partially written output or unsupported provenance.
        provenance=json.loads((directory/'provenance.json').read_text())
        digest=provenance['data']['sha256']
        if provenance.get('status')!='completed' or digest!=manifest['source_sha256'] or digest not in (LOCKED_SHA,FALLBACK_SHA):
            raise ValueError('unverified/incomplete evaluation')
        for key,path in [('protocol_sha256',ROOT/'docs/V43_PROTOCOL.md'),('benchmark_sha256',ROOT/'src/reality_v43.py')]:
            if provenance.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('protocol or benchmark fingerprint mismatch')
        if provenance.get('locked_v43_input') is not (digest==LOCKED_SHA):raise ValueError('input identity flag mismatch')
        if manifest.get('seed')!=431901 or list(manifest['primary'])!=[.25,.05,.3]:raise ValueError('locked protocol settings mismatch')
    cap,cost,margin=map(float,manifest['primary'])
    records={}
    for r in _rows(directory/'policy_results.csv'):
        key=(r['arm'],r['method'])
        if key in records: raise ValueError('duplicate policy row')
        if float(r['coverage'])>cap+1e-9: raise ValueError('policy exceeds budget')
        records[key]=r
    comparisons={}
    for c in _rows(directory/'comparison.csv'):
        if c['arm'] in comparisons:raise ValueError('duplicate comparison')
        comparisons[c['arm']]=c
    if set(comparisons)!=set(manifest['arms']): raise ValueError('missing arm comparison')
    out=[]
    for arm,a in manifest['arms'].items():
        c=comparisons[arm]
        if any(c[k]!=a[k] for k in ('baseline','futurelab')): raise ValueError('comparison does not match frozen validation selection')
        b,f=(from_locked_policy(records[arm,a[k]],cost,margin,cohort) for k in ('baseline','futurelab'))
        if b['holdout_n']!=a['n_test'] or f['holdout_n']!=a['n_test']:raise ValueError('holdout size mismatch')
        gap=Estimate(*(float(c[k]) for k in ('difference','lo','hi')))
        bv=b['holdout_net_per_eligible']['point']; fv=f['holdout_net_per_eligible']['point']
        if not math.isclose(gap.point,fv-bv,abs_tol=1e-9) or not math.isclose(bv,float(c['baseline_value']),abs_tol=1e-9) or not math.isclose(fv,float(c['futurelab_value']),abs_tol=1e-9): raise ValueError('inconsistent paired result')
        clears_protocol=gap.lo>0 and (bv<=0 or gap.point>=.10*bv)
        confirmatory=(mode=='real' and c['confirmatory'].lower()=='true' and provenance.get('locked_v43_input') is True and arm=='Mens E-Mail')
        # Evidence gate only; do not switch models based on holdout.
        eligible=confirmatory and clears_protocol and f['holdout_net_per_eligible']['lo']>0
        out.append({'arm':arm,'baseline_name':a['baseline'],'futurelab_name':a['futurelab'],'baseline':b,'futurelab':f,
                    'paired_gap_per_eligible':asdict(gap),'illustrative_paired_gap_at_cohort':asdict(gap.affine(cohort)), 'gap_scaling_note':'Paired gap uses exact fractional holdout coverage; separate economic budgets round contacts down to whole units.',
                    'protocol_pass':clears_protocol,'confirmatory':confirmatory,'eligible_for_prospective_trial':eligible})
    return {'mode':mode,'cohort_assumption':cohort,'source':str(directory),'primary_cap':cap,'cost':cost,'margin':margin,
            'caution':'Conditional historical valuation; scaled cohort is illustrative, not a future prediction interval. No holdout model selection or production approval.', 'comparisons':out}


def external_scenarios(path: Path, cohort=10000):
    source=json.loads(Path(path).read_text())
    if source['mode']!='external_published_aggregate':raise ValueError('wrong source type')
    out=[]
    for arm,summary in source['spend_effects'].items():
        for cap in (.1,.25,.5,1.):
            for margin in (.1,.3,.5):
                for cost in (.0,.05,.2):
                    r=economics(Estimate(**summary),cohort,math.floor(cohort*cap),cost,margin)
                    row={'arm':arm,'cap':cap,**r}
                    out.append(row)
    return {'mode':source['mode'],'source':source['source'],'note':'Third-party rounded, whole-arm estimates; random untargeted contacts assumed, no individual prediction or algorithm evaluation. Scenarios are descriptive; no simultaneous confidence coverage. Cost and margin hypothetical.', 'scenarios':out}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--locked',type=Path); p.add_argument('--external',type=Path); p.add_argument('--cohort',type=int,default=10000);p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if bool(a.locked)==bool(a.external): p.error('specify exactly one input')
    result=locked_readout(a.locked,a.cohort) if a.locked else external_scenarios(a.external,a.cohort)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
if __name__=='__main__':main()
