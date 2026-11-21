"""Finite-horizon evidence gate for frozen candidate/reference contrasts.
Inputs are scalar projected block contrasts, already net of treatment cost.
A negative control is a zero-effect measurement on a matched channel; its
bias must match the target channel for subtraction to identify the contrast.
"""
import numpy as np
from scipy.stats import t, norm


def evidence_gate(target, control, reported_sd, groups, *, mode='robust', paired=True,
                  looks=(3,4,6), alpha=.05, epsilon=.035, unit_cost=.0015):
    """target/control [attempt, source, block]; source count is 1 or 3.
    robust: median of three per-source upper bounds; tolerates one arbitrary
    corrupted source when each other source has iid Gaussian unbiased blocks.
    No independence between sources is required. pooled: mean across sources
    per block; only valid if the pooled mean is unbiased and blocks iid normal.
    single: source 0. known: single with reported (known) per-block SD.
    No samples after a stop are read. Each target/control observation bills
    one contrast per differing group. The procedure makes no across-period claim.
    """
    x=np.asarray(target,float); c=None if control is None else np.asarray(control,float)
    g=np.asarray(groups); sd=np.asarray(reported_sd,float)
    if x.ndim!=3 or x.shape[1] not in (1,3) or g.shape!=(len(x),) or sd.shape!=(len(x),) or not np.isfinite(sd).all() or (sd<=0).any() or not np.isfinite(g).all() or (g<=0).any() or (g!=np.floor(g)).any(): raise ValueError('invalid dimensions or scale')
    if mode not in ('known','single','pooled','robust') or (mode in ('pooled','robust') and x.shape[1]!=3) or (paired and (c is None or c.shape!=x.shape)) or (mode=='known' and paired): raise ValueError('invalid source model')
    if not looks or any(type(k)!=int for k in looks) or tuple(sorted(set(looks)))!=tuple(looks) or looks[-1]>x.shape[2] or looks[0]<(1 if mode=='known' else 2) or not 0<alpha<1 or not np.isfinite(epsilon) or not np.isfinite(unit_cost) or unit_cost<0: raise ValueError('invalid audit configuration')
    n=len(x); s=3 if mode in ('pooled','robust') else 1
    weights=np.asarray(looks,float)**2; weights/=weights.sum()
    active=np.ones(n,bool); accepted=np.zeros(n,bool); count=np.zeros(n,int); upper=np.full(n,np.nan); lower=upper.copy(); disagreement=np.zeros(n,bool)
    prev=0; total=np.zeros((n,s)); ss=np.zeros((n,s))
    for look,w in zip(looks,weights):
        ix=np.flatnonzero(active)
        if len(ix)==0: break
        z=x[ix,:s,prev:look]
        if paired: z=z-c[ix,:s,prev:look]
        if not np.isfinite(z).all(): raise ValueError('nonfinite selected observation')
        if mode=='pooled': z=z.mean(1,keepdims=True)
        total[ix,:z.shape[1]]+=z.sum(2); ss[ix,:z.shape[1]]+=(z*z).sum(2); count[ix]=look
        m=total[ix,:z.shape[1]]/look
        if mode=='known': half=norm.ppf(1-alpha*w)*sd[ix,None]/np.sqrt(look)
        else:
            var=np.maximum(0,(ss[ix,:z.shape[1]]-look*m*m)/(look-1))
            # Zero variance is not trusted: exact repeated values may indicate sensor failure.
            half=np.where(var>0,t.ppf(1-alpha*w/(3 if mode=='robust' else 1),look-1)*np.sqrt(var/look),np.inf)
        u=m+half; l=m-half
        if mode=='robust':
            upper[ix]=np.median(u,axis=1); lower[ix]=np.median(l,axis=1)
            disagreement[ix]|=(l.max(1)>u.min(1))
        else: upper[ix]=u[:,0]; lower[ix]=l[:,0]
        yes=upper[ix]<=epsilon; no=lower[ix]>epsilon
        accepted[ix[yes]]=True;active[ix[yes|no]]=False;prev=look
    samples=count*g*s*(2 if paired else 1)
    return dict(accept=accepted,blocks=count,samples=samples,cost=samples*unit_cost,upper=upper,lower=lower,disagreement=disagreement)
