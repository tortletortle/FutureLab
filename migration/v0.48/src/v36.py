"""Bounded online regime archive, approximate switching Gaussian filter.
Policies see randomized contrasts and sampling variances only, never regime labels.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from v35 import C, adaptive, log_normal

@dataclass(frozen=True)
class Config:
    slots: int=6
    hazard: float=.12
    birth: float=.05
    birth_gate: float=.55
    learn_gate: float=.65
    recall_gate: float=.70
    var_floor: float=.035
    fresh_var: float=2.25
    ensemble_share: float=.03
    def validate(self):
        if not (isinstance(self.slots,int) and 1<=self.slots<=32 and 0<self.birth<=self.hazard<1 and .5<=self.birth_gate<1 and .5<=self.learn_gate<1 and .5<=self.recall_gate<1 and 0<self.var_floor<=self.fresh_var and 0<self.ensemble_share<1):
            raise ValueError('invalid archive configuration')

def archive(y,v,cfg=Config()):
    """Return pre-observation predictions and post-observation memory events.
    State posterior is approximated with gated single-slot conjugate updates;
    unidentified novelty is discarded rather than injected into archived slots.
    Birth replaces least recently learned slot (excluding active if possible).
    """
    cfg.validate(); y=np.asarray(y,float);v=np.asarray(v,float)
    if y.ndim!=3 or y.shape!=v.shape or min(y.shape,default=0)<1 or not np.isfinite(y).all() or not np.isfinite(v).all() or np.any(v<=0):raise ValueError('invalid observation stream')
    B,T,G=y.shape;K=cfg.slots;b=np.arange(B)
    means=np.full((B,K,G),C);var=np.full_like(means,4.);active=np.zeros((B,K),bool);active[:,0]=True
    prob=np.zeros((B,K));prob[:,0]=1; born=np.full((B,K),-1,int);born[:,0]=0
    last=np.full((B,K),-1,int); last[:,0]=0
    pred=np.zeros_like(y);loss=np.zeros_like(y);events=np.zeros((B,T),np.int8);source=np.full((B,T),-1,int);age=np.zeros((B,T),int);count=np.zeros((B,T),int);confidence=np.zeros((B,T));accepted=np.zeros((B,T),bool)
    for t in range(T):
        # One broad candidate, centered at preceding filtered mean, is a possible new state.
        # Only past summaries enter the predictive mean; current variance is ancillary.
        old=(prob[:,:,None]*means).sum(1); winner_before=prob.argmax(1)
        bank_prior=(1-cfg.hazard)*prob+(cfg.hazard-cfg.birth)*active/active.sum(1,keepdims=True)
        weights=np.column_stack([bank_prior,np.full(B,cfg.birth)])
        m=np.concatenate([means,old[:,None,:]],axis=1)
        s=np.concatenate([np.maximum(var,cfg.var_floor),np.full((B,1,G),cfg.fresh_var)],axis=1)
        pred[:,t]=(weights[:,:,None]*m).sum(1)
        lw=np.full_like(weights,-np.inf);np.log(weights,out=lw,where=weights>0)
        ll=log_normal(y[:,t,None,:],m,s+v[:,t,None,:]);loss[:,t]=-logsumexp(lw[:,:,None]+ll,axis=1)
        score=lw+ll.sum(2);post=np.exp(score-logsumexp(score,axis=1,keepdims=True));best=post[:,:K].argmax(1)
        novelty=post[:,K]>=cfg.birth_gate
        # No new slot unless enough predictive evidence; unrelated slots remain immutable.
        rec=(~novelty)&(best!=winner_before)&(post[b,best]>=cfg.recall_gate)&(t-last[b,best]>=3)
        events[rec,t]=2;source[rec,t]=born[b[rec],best[rec]];age[rec,t]=t-last[b[rec],best[rec]]
        confidence[:,t]=post[b,best]
        learn=(~novelty)&(post[b,best]>=cfg.learn_gate)
        ids=b[learn];j=best[learn]
        pv=np.maximum(var[ids,j],cfg.var_floor);r=v[ids,t]
        means[ids,j]+=pv/(pv+r)*(y[ids,t]-means[ids,j]);var[ids,j]=pv*r/(pv+r);last[ids,j]=t
        # Posterior probability over allocated regimes conditional on no accepted birth.
        bank_log=score[:,:K];prob=np.exp(bank_log-logsumexp(bank_log,axis=1,keepdims=True))
        if novelty.any():
            ids=b[novelty];free=~active[ids];has_free=free.any(1)
            # LRU replacement, preserving the previously dominant slot where possible.
            lru=last[ids].copy();lru[np.arange(len(ids)),winner_before[ids]]=np.iinfo(np.int32).max if K>1 else lru[np.arange(len(ids)),winner_before[ids]]
            j=np.where(has_free,free.argmax(1),lru.argmin(1));r=v[ids,t];p=cfg.fresh_var
            means[ids,j]=old[ids]+p/(p+r)*(y[ids,t]-old[ids]);var[ids,j]=p*r/(p+r)
            active[ids,j]=True;born[ids,j]=t;last[ids,j]=t;prob[ids]=0;prob[ids,j]=1
            events[ids,t]=1;source[ids,t]=t;confidence[ids,t]=post[ids,K]
        accepted[:,t]=learn|novelty;count[:,t]=active.sum(1)
    return dict(pred=pred,loss=loss,event=events,source=source,age=age,count=count,confidence=confidence,learned=accepted)

def guarded(bank_pred,bank_loss,base_pred,base_loss,share=.03):
    """Fixed-share expert weighting using average marginal predictive log score.
    No hard guarantee of improved decisions; policy output precedes current score.
    """
    if not 0<share<1:raise ValueError('invalid share')
    B,T,G=bank_pred.shape;w=np.full((B,2),.5);pred=np.empty_like(bank_pred);trace=np.empty((B,T))
    for t in range(T):
        w=(1-share)*w+share/2;trace[:,t]=w[:,0];pred[:,t]=w[:,0,None]*bank_pred[:,t]+w[:,1,None]*base_pred[:,t]
        lp=np.log(w)-np.column_stack([bank_loss[:,t].mean(1),base_loss[:,t].mean(1)]);w=np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
    return pred,trace
