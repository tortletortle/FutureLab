"""Independent sequential Gaussian contrast audit, frozen actions, fixed horizon.
No outcomes used in the archive. All observations are paid on access.
"""
import numpy as np
from scipy.special import ndtr, ndtri, logsumexp
from v35 import C
from v38 import EPS, PROBE_COST

ROUND_VAR=.45

def potential_gain(b,candidate,reference):
    """Model-dependent upper bound E[max(-L,0)] on gross value of any binary audit."""
    d=(reference.astype(float)-candidate.astype(float))/candidate.shape[1]
    mu=np.einsum('bg,bhg->bh',d,b.mean-C)
    sd=np.sqrt(np.einsum('bg,bhg->bh',d*d,b.var))
    z=np.divide(mu,sd,out=np.zeros_like(mu),where=sd>0)
    gain=np.where(sd>0,sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)-mu*ndtr(-z),np.maximum(-mu,0))
    return np.maximum(0,(b.weights()*gain).sum(1))

def audit(candidate,reference,bank,variance,method='spend',max_rounds=3,alpha=.05,epsilon=EPS,unit_cost=PROBE_COST,value_bound=None,early_reject=True):
    """Rows are independent audit attempts. bank [N,G,M] is accessed only on demand.
    variance [N,G] fixed known per-round variance. One round samples every differing group.
    For spend: alpha allocation proportional to (1,...,M)^2; union bound for each tail.
    For mixture: fixed mixture of normal likelihood ratios, evaluated sequentially.
    Fixed method uses only the final look; peek is intentionally invalid negative control.
    Screening uses pre-audit information and does not enter the frequentist claim.
    """
    a=np.asarray(candidate);r=np.asarray(reference);v=np.asarray(variance,float);bank=np.asarray(bank)
    if a.ndim!=2 or a.shape!=r.shape or a.shape!=v.shape or bank.ndim!=3 or bank.shape[:2]!=a.shape or not np.isin(a,[0,1]).all() or not np.isin(r,[0,1]).all() or not np.isfinite(v).all() or np.any(v<=0):raise ValueError('invalid audit inputs')
    if method not in ('fixed','spend','mixture','peek') or type(max_rounds)!=int or not 1<=max_rounds<=bank.shape[2] or not 0<alpha<1 or not np.isfinite(epsilon) or not np.isfinite(unit_cost) or unit_cost<0:raise ValueError('invalid audit configuration')
    N,G=a.shape;d=(r.astype(float)-a.astype(float))/G;mask=d!=0;groups=mask.sum(1);s2=(d*d*v).sum(1);sd=np.sqrt(s2)
    accepted=groups==0;screened=np.zeros(N,bool);running=groups>0;count=np.zeros(N,int);total=np.zeros(N);stat=np.full(N,np.nan);reason=np.full(N,'same',dtype='<U12');reason[running]='budget'
    if value_bound is not None:
        vb=np.asarray(value_bound,float)
        if vb.shape!=(N,) or not np.isfinite(vb).all() or (vb<0).any():raise ValueError('invalid gain bound')
        screened=running&(vb<=unit_cost*groups);running&=~screened;reason[screened]='screened'
    weights=np.arange(1,max_rounds+1,dtype=float)**2;weights/=weights.sum()
    # standardized alternative distances; fixed conditional on frozen d and known variance.
    shifts=np.array([.25,.5,1.,2.,4.]);logw=-np.log(len(shifts))
    for k in range(1,max_rounds+1):
        ids=np.flatnonzero(running)
        if not len(ids):break
        ii,jj=np.nonzero(mask[ids]);samples=bank[ids[ii],jj,k-1]
        if not np.isfinite(samples).all():raise ValueError('missing selected audit sample')
        z=np.bincount(ii,weights=d[ids[ii],jj]*(samples-C),minlength=len(ids))
        total[ids]+=z;count[ids]+=1
        if method=='fixed' and k<max_rounds:continue
        mean=total[ids]/k
        if method=='mixture':
            t=(total[ids]-k*epsilon)/sd[ids];neg=logsumexp(-t[:,None]*shifts-k*shifts**2/2+logw,axis=1);pos=logsumexp(t[:,None]*shifts-k*shifts**2/2+logw,axis=1)
            stat[ids]=neg;yes=neg>=-np.log(alpha);no=pos>=-np.log(alpha)
        else:
            q=ndtri(1-alpha*(weights[k-1] if method=='spend' else 1.));upper=mean+q*sd[ids]/np.sqrt(k);lower=mean-q*sd[ids]/np.sqrt(k)
            stat[ids]=upper;yes=upper<=epsilon;no=lower>epsilon
        accepted[ids[yes]]=True;reason[ids[yes]]='accepted';running[ids[yes]]=False
        if early_reject and method!='fixed':
            no=no&~yes;reason[ids[no]]='rejected';running[ids[no]]=False
    return dict(accept=accepted,rounds=count,samples=count*groups,cost=count*groups*unit_cost,reason=reason,statistic=stat,screened=screened)
