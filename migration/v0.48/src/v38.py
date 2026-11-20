"""Decision-equivalent archive reuse; online Gaussian-mixture working model.
Deploy after discovery and optional probes, then update bounded memory.
Risk is relative to the current posterior-optimal recomputation, not oracle.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr, logsumexp
from v35 import C, log_normal
from v36 import Config
from v37 import Belief, update_all, update_probe, choose

EPS=.035
PROBE_COST=.0015
DEFAULT_ARCHIVE=Config(hazard=.10,birth=.025,birth_gate=.75)


def relative_risk(b,actions,reference,eps=EPS):
    """Exact P(V(reference)-V(action)>eps | working mixture), equal population.
    b components have conditionally independent coordinates. Handles zero variance.
    actions [B,K,G], reference [B,G]. Returns [B,K] probabilities and means.
    """
    if not np.isfinite(eps) or eps<0:raise ValueError('invalid tolerance')
    d=(np.asarray(reference,float)[:,None]-np.asarray(actions,float))/b.mean.shape[2]
    mu=np.einsum('bkg,bhg->bkh',d,b.mean-C)
    var=np.einsum('bkg,bhg->bkh',d*d,b.var)
    z=np.divide(mu-eps,np.sqrt(var),out=np.zeros_like(mu),where=var>0)
    p=np.where(var>0,ndtr(z),mu>eps)
    w=b.weights();return np.einsum('bkh,bh->bk',p,w),np.einsum('bkh,bh->bk',mu,w)


def oracle_risk(b,actions):
    """Exact posterior mean missed net value vs per-group sign oracle."""
    net=b.mean-C;s=np.sqrt(b.var);z=net/s
    pos=s*np.exp(-z*z/2)/np.sqrt(2*np.pi)+net*ndtr(z)
    epos=np.einsum('bh,bhg->bg',b.weights(),pos)
    m=b.posterior_mean()-C
    return (epos[:,None]-actions*m[:,None]).mean(2)


def decision_gain(b,r):
    """Joint moment-matched Gaussian EVSI for unconstrained binary decisions.
    Between-component covariance allows probes to change decisions in other groups.
    Exact only under the Gaussian moment surrogate, not the original mixture.
    """
    w=b.weights();m=b.posterior_mean();delta=b.mean-m[:,None]
    cov=np.einsum('bh,bhi,bhj->bij',w,delta,delta)
    i=np.arange(m.shape[1]);cov[:,i,i]+=np.einsum('bh,bhg->bg',w,b.var)
    sd=np.abs(cov)/np.sqrt(cov[:,i,i]+r)[:,None,:]
    d=np.abs(m-C)[:,:,None];z=np.divide(d,sd,out=np.zeros_like(sd),where=sd>0)
    gain=np.where(sd>0,sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)-d*ndtr(-z),0.)
    return np.maximum(0,gain.mean(1))


def select_candidate(b,actions,active,mode='risk',eps=EPS):
    ref=b.posterior_mean()>C
    risk,excess=relative_risk(b,actions,ref,eps)
    if mode=='identity':
        scores=np.where(active,b.weights()[:,:active.shape[1]],-1)
        j=scores.argmax(1);confidence=scores[np.arange(len(j)),j]
    elif mode in ('risk','always'):
        # Minimize posterior expected loss; ties prefer lower tail risk, then index.
        score=np.where(active,excess+1e-10*risk,np.inf)
        j=score.argmin(1);confidence=1-risk[np.arange(len(j)),j]
    else:raise ValueError('invalid gate')
    return j,confidence,ref,risk[np.arange(len(j)),j]


def working_belief(means,var,active,prob,cfg):
    B,K,G=means.shape
    old=(prob[:,:,None]*means).sum(1)
    w=(1-cfg.hazard)*prob+(cfg.hazard-cfg.birth)*active/active.sum(1,keepdims=True)
    # Broad + near-regime uncertainty, splitting a fixed novelty prior.
    nm=np.concatenate([means,old[:,None],means],axis=1)
    nv=np.concatenate([np.maximum(var,cfg.var_floor),np.full((B,1,G),cfg.fresh_var),np.maximum(var,cfg.var_floor)+.55**2],axis=1)
    nw=np.concatenate([w,np.full((B,1),cfg.birth*.5),cfg.birth*.5*active/active.sum(1,keepdims=True)],axis=1)
    lp=np.full_like(nw,-np.inf);np.log(nw,out=lp,where=nw>0)
    return Belief(nm.copy(),nv.copy(),lp,K)


def simulate(y,v,bank,r,selector='evsi',gate='risk',threshold=.9,budget=1,interval=4,seed=1,cfg=DEFAULT_ARCHIVE,feedback=True):
    """[B,T,G] discovery; bank [B,T,G,budget] evaluator-owned, accessed only on choice.
    Candidate is a stored pre-discovery action. All arms deploy after discovery.
    Gate does not affect evidence collection, so thresholds can be swept offline.
    Feedback combines selected probes and discovery by precision for archive update.
    """
    cfg.validate();y=np.asarray(y,float);v=np.asarray(v,float);bank=np.asarray(bank,float);r=np.asarray(r,float)
    if y.ndim!=3 or y.shape!=v.shape or min(y.shape)<1 or not np.isfinite(y).all() or not np.isfinite(v).all() or (v<=0).any():raise ValueError('invalid discovery')
    B,T,G=y.shape;K=cfg.slots
    if type(budget)!=int or budget<0 or type(interval)!=int or interval<1 or bank.ndim!=4 or bank.shape[:3]!=y.shape or bank.shape[3]<budget or r.shape!=y.shape or not np.isfinite(r).all() or (r<=0).any():raise ValueError('invalid probes')
    if not 0<=threshold<=1.000001:raise ValueError('invalid threshold')
    rng=np.random.default_rng(seed);ix=np.arange(B)
    means=np.full((B,K,G),C);var=np.full_like(means,4.);active=np.zeros((B,K),bool);active[:,0]=True
    prob=np.zeros((B,K));prob[:,0]=1;last=np.full((B,K),-1,int);last[:,0]=0
    out={k:np.zeros((B,T)) for k in ('confidence','risk','reuse','old_reuse','match','probes','birth','slots','bayes_oracle_risk')}
    for k in ['decision','candidate','reference']:out[k]=np.zeros_like(y,dtype=bool)
    out['selected']=np.full((B,T,budget),-1,int)
    for t in range(T):
        old=(prob[:,:,None]*means).sum(1);winner=prob.argmax(1);acts=means>C
        b=working_belief(means,var,active,prob,cfg);update_all(b,y[:,t],v[:,t]);prec=1/v[:,t];suff=y[:,t]*prec
        if t>=12 and (t-12)%interval==0:
            visits=np.zeros((B,G),int)
            for q in range(budget):
                j=decision_gain(b,r[:,t]).argmax(1) if selector=='evsi' else choose(b,r[:,t],selector,rng)
                z=bank[ix,t,j,visits[ix,j]];visits[ix,j]+=1;rv=r[ix,t,j]
                update_probe(b,j,z,rv);prec[ix,j]+=1/rv;suff[ix,j]+=z/rv;out['selected'][:,t,q]=j
            out['probes'][:,t]=budget
        j,conf,ref,risk=select_candidate(b,acts,active,gate)
        candidate=acts[ix,j];pick=(conf>=threshold)&(t>=12)
        if gate=='always':pick=np.full(B,t>=12)
        out['decision'][:,t]=np.where(pick[:,None],candidate,ref);out['candidate'][:,t]=candidate;out['reference'][:,t]=ref
        out['confidence'][:,t]=conf;out['risk'][:,t]=risk;out['reuse'][:,t]=pick;out['old_reuse'][:,t]=pick&(t-last[ix,j]>=3);out['match'][:,t]=(candidate==ref).all(1)
        out['bayes_oracle_risk'][:,t]=oracle_risk(b,candidate[:,None])[:,0]
        # Existing V36-style hard archive update; all historical actions came from past evidence.
        # If feedback disabled, only discovery is committed to archive (ablation).
        obs=suff/prec if feedback else y[:,t];rv=1/prec if feedback else v[:,t]
        prior=(1-cfg.hazard)*prob+(cfg.hazard-cfg.birth)*active/active.sum(1,keepdims=True)
        mm=np.concatenate([means,old[:,None]],1);vv=np.concatenate([np.maximum(var,cfg.var_floor),np.full((B,1,G),cfg.fresh_var)],1)
        pw=np.column_stack([prior,np.full(B,cfg.birth)]);lp=np.full_like(pw,-np.inf);np.log(pw,out=lp,where=pw>0)
        score=lp+log_normal(obs[:,None],mm,vv+rv[:,None]).sum(2);post=np.exp(score-logsumexp(score,axis=1,keepdims=True));best=post[:,:K].argmax(1)
        novelty=post[:,K]>=cfg.birth_gate;learn=(~novelty)&(post[ix,best]>=cfg.learn_gate)
        ids=ix[learn];jj=best[learn];p=np.maximum(var[ids,jj],cfg.var_floor);rr=rv[ids]
        means[ids,jj]+=p/(p+rr)*(obs[ids]-means[ids,jj]);var[ids,jj]=p*rr/(p+rr);last[ids,jj]=t
        prob=np.exp(score[:,:K]-logsumexp(score[:,:K],axis=1,keepdims=True))
        if novelty.any():
            ids=ix[novelty];free=~active[ids];lru=last[ids].copy()
            if K>1:lru[np.arange(len(ids)),winner[ids]]=np.iinfo(np.int32).max
            jj=np.where(free.any(1),free.argmax(1),lru.argmin(1));p=cfg.fresh_var;rr=rv[ids]
            means[ids,jj]=old[ids]+p/(p+rr)*(obs[ids]-old[ids]);var[ids,jj]=p*rr/(p+rr);active[ids,jj]=True;last[ids,jj]=t;prob[ids]=0;prob[ids,jj]=1
        out['birth'][:,t]=novelty;out['slots'][:,t]=active.sum(1)
    return out
