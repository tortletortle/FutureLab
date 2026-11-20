"""Dependency-aware, finite-horizon evidence routing for frozen contrasts.
Source declarations are external assumptions, never inferred safety certificates.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import t

@dataclass(frozen=True)
class Source:
    domain: str
    bias_bound: float | None  # residual absolute mean bias after paired subtraction
    cost: float = 2.0        # target + negative control, in sample-equivalent units


def route_gate(target, control, groups, sources, *, fault_budget=1,
               looks=(3,6,9), budget=36., alpha=.05, epsilon=.035,
               unit_cost=.0015, adaptive=True):
    """At most fault_budget distinct domains may be arbitrary. Each other
    declared source has iid normal blocks and residual bias <= bias_bound.
    One cheapest eligible representative per domain, fixed before observations.
    Union bound over all representative prefixes handles adaptive sampling.
    Budget is per differing group; costs multiplied by number of groups.
    No access to unpurchased observations; unknown assurance -> no audit.
    """
    x=np.asarray(target,float); c=np.asarray(control,float); g=np.asarray(groups)
    if x.ndim!=3 or c.shape!=x.shape or x.shape[1]!=len(sources) or g.shape!=(len(x),) or not np.isfinite(g).all() or (g<=0).any() or (g!=np.floor(g)).any(): raise ValueError('dimensions/groups')
    if not looks or any(type(k)!=int for k in looks) or tuple(sorted(set(looks)))!=tuple(looks) or looks[0]<2 or looks[-1]>x.shape[2] or not 0<alpha<1 or not np.isfinite([budget,epsilon,unit_cost]).all() or budget<0 or unit_cost<0: raise ValueError('configuration')
    if fault_budget is not None and (type(fault_budget)!=int or fault_budget<0): raise ValueError('fault budget')
    rep={}
    for j,s in enumerate(sources):
        if not isinstance(s,Source) or not isinstance(s.domain,str) or not s.domain or not np.isfinite(s.cost) or s.cost<=0 or (s.bias_bound is not None and (not np.isfinite(s.bias_bound) or s.bias_bound<0)): raise ValueError('source')
        if s.bias_bound is not None and (s.domain not in rep or s.cost<sources[rep[s.domain]].cost): rep[s.domain]=j
    ids=list(rep.values()); n=len(x); m=len(ids)
    accepted=np.zeros(n,bool); count=np.zeros((n,len(sources)),int); spent=np.zeros(n); upper=np.full((n,len(sources)),np.inf); calls=np.zeros(n,int)
    out=lambda:dict(accept=accepted,counts=count,samples=spent*g,cost=spent*g*unit_cost,upper=upper,calls=calls,representatives=ids,declined=np.full(n, fault_budget is None or m<((fault_budget or 0)+1)))
    if fault_budget is None or m<fault_budget+1: return out()
    need=fault_budget+1; kmax=looks[-1]; price=np.array([sources[j].cost for j in ids]); bias=np.array([sources[j].bias_bound for j in ids]); ct=np.zeros((n,m),int); total=np.zeros((n,m)); ss=total.copy(); u=np.full((n,m),np.inf); score=u.copy(); stage=np.zeros((n,m),int)
    inc=np.diff((0,*looks)); crit=t.ppf(1-alpha/(m*len(looks)),np.array(looks)-1)
    # At most m * len(looks) acquisitions. Previously certified domains need no more samples.
    for step in range(m*len(looks)):
        passmask=u<=epsilon; accepted|=passmask.sum(1)>=need
        possible=(stage<len(looks))&~passmask
        next_cost=price[None,:]*inc[np.minimum(stage,len(looks)-1)]
        possible &= (spent[:,None]+next_cost<=budget+1e-10)&~accepted[:,None]
        # Cheap unseen domains first. Then projected final-look upper bound per source.
        priority=np.where(stage==0,-1e9+price[None,:],np.nan_to_num(score,posinf=1e100,neginf=-1e100)) if adaptive else stage+price[None,:]*1e-3
        jchoice=np.argmin(np.where(possible,priority,np.inf),axis=1); active=possible.any(1)
        if not active.any(): break
        stage_before=stage.copy()
        for j,src in enumerate(ids):
            for h,k in enumerate(looks):
                ix=np.flatnonzero(active&(jchoice==j)&(stage_before[:,j]==h))
                if not len(ix):continue
                prev=0 if h==0 else looks[h-1]; z=x[ix,src,:k]-c[ix,src,:k]
                if not np.isfinite(z).all():raise ValueError('nonfinite selected observation')
                ct[ix,j]=k; count[ix,src]=k; stage[ix,j]+=1; calls[ix]+=1; spent[ix]+=price[j]*(k-prev)
                mean=z.mean(1); var=np.where(np.ptp(z,axis=1)>0,z.var(1,ddof=1),0.); half=np.where(var>0,crit[h]*np.sqrt(var/k),np.inf)
                u[ix,j]=mean+half+bias[j]; upper[ix,src]=u[ix,j]
                score[ix,j]=mean+np.where(var>0,crit[-1]*np.sqrt(var/kmax),np.inf)+bias[j]
    accepted|=(u<=epsilon).sum(1)>=need
    return out()
