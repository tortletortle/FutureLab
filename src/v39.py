"""V0.39 prequential risk adjustment, gated memory writes and independent audits."""
import numpy as np
from scipy.special import ndtr, logsumexp, ndtri
from v35 import C, log_normal
from v38 import EPS, PROBE_COST, DEFAULT_ARCHIVE, decision_gain, working_belief, select_candidate, relative_risk, oracle_risk
from v37 import update_all, update_probe, choose

AUDIT_VAR=.15
AUDIT_UNIT_COST=PROBE_COST*.45/AUDIT_VAR

def adjusted_risk(b, actions, ref, bias, scale, eps=EPS):
    """Heuristic stress envelope; not a calibrated frequentist probability."""
    d=(ref.astype(float)-actions.astype(float))/b.mean.shape[2]
    mu=np.einsum('bg,bhg->bh',d,b.mean-C)
    var=np.einsum('bg,bhg->bh',d*d,b.var)
    shift=np.abs(np.sum(d*bias,axis=1))[:,None]
    sd=np.sqrt(var*np.maximum(1,scale)[:,None])
    p=np.where(sd>0,ndtr(np.divide(mu+shift-eps,sd,out=np.zeros_like(mu),where=sd>0)),mu+shift>eps)
    exact=relative_risk(b,actions[:,None],ref,eps)[0][:,0]
    return np.maximum(exact,(p*b.weights()).sum(1))

def audit_upper(candidate,reference,obs,variance,alpha=.05):
    """One-sided independent Gaussian contrast upper bound. Known variance only."""
    if not 0<alpha<1: raise ValueError('invalid alpha')
    a=np.asarray(candidate,float); r=np.asarray(reference,float); y=np.asarray(obs,float); v=np.asarray(variance,float)
    if a.shape!=r.shape or y.shape!=a.shape or v.shape!=a.shape or a.ndim!=2 or not np.isfinite(v).all() or (v<=0).any(): raise ValueError('invalid audit shape/variance')
    d=(r-a)/a.shape[1]
    if not np.isfinite(y[d!=0]).all(): raise ValueError('missing selected audit outcome')
    return np.sum(d*np.where(d!=0,y-C,0),1)+ndtri(1-alpha)*np.sqrt(np.sum(d*d*v,1))

def simulate(y,v,bank,r,audit=None,audit_var=None,calibrate=False,quarantine=False,audit_gate=False,alpha=.05,selector='evsi',gate='risk',threshold=.9,budget=1,interval=4,seed=1,cfg=DEFAULT_ARCHIVE,feedback=True):
    """[B,T,G] discovery; bank [B,T,G,budget] evaluator-owned, accessed only on choice.
    Candidate is a stored pre-discovery action. All arms deploy after discovery.
    Gate changes audit collection/cost, but audit is never used for archive learning.
    Feedback combines selected EVSI probes and discovery by precision for archive update.
    Quarantine blocks a single write, not a persistent separate memory bank.
    """
    cfg.validate();y=np.asarray(y,float);v=np.asarray(v,float);bank=np.asarray(bank,float);r=np.asarray(r,float)
    if y.ndim!=3 or y.shape!=v.shape or min(y.shape)<1 or not np.isfinite(y).all() or not np.isfinite(v).all() or (v<=0).any():raise ValueError('invalid discovery')
    B,T,G=y.shape;K=cfg.slots
    if type(budget)!=int or budget<0 or type(interval)!=int or interval<1 or bank.ndim!=4 or bank.shape[:3]!=y.shape or bank.shape[3]<budget or r.shape!=y.shape or not np.isfinite(r).all() or (r<=0).any():raise ValueError('invalid probes')
    if not 0<=threshold<=1.000001:raise ValueError('invalid threshold')
    if (calibrate or audit_gate) and gate!='risk':raise ValueError('risk gate required for V39 extensions')
    if audit_gate:
        audit=np.asarray(audit,float);audit_var=np.asarray(audit_var,float)
        if audit.shape!=y.shape or audit_var.shape!=y.shape or not np.isfinite(audit_var).all() or (audit_var<=0).any() or not 0<alpha<1:raise ValueError('invalid audit inputs')
    rng=np.random.default_rng(seed);ix=np.arange(B)
    bias=np.zeros((B,G));scale=np.ones(B)
    means=np.full((B,K,G),C);var=np.full_like(means,4.);active=np.zeros((B,K),bool);active[:,0]=True
    prob=np.zeros((B,K));prob[:,0]=1;last=np.full((B,K),-1,int);last[:,0]=0
    out={k:np.zeros((B,T)) for k in ('confidence','risk','reuse','old_reuse','match','probes','birth','slots','bayes_oracle_risk','adjusted_risk','drift_scale','quarantine','audit_groups','audit_cost','audit_upper','model_accept')}
    for k in ['decision','candidate','reference']:out[k]=np.zeros_like(y,dtype=bool)
    out['selected']=np.full((B,T,budget),-1,int)
    for t in range(T):
        old=(prob[:,:,None]*means).sum(1);winner=prob.argmax(1);acts=means>C
        b=working_belief(means,var,active,prob,cfg)
        # Pre-discovery predictor; update stress statistics only after decision.
        pm=b.posterior_mean();pv=np.einsum('bh,bhg->bg',b.weights(),b.var+(b.mean-pm[:,None])**2)
        innovation=y[:,t]-pm;z2=(innovation**2/(pv+v[:,t])).mean(1)
        update_all(b,y[:,t],v[:,t]);prec=1/v[:,t];suff=y[:,t]*prec
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
        corrected=adjusted_risk(b,candidate,ref,bias,scale) if calibrate else risk
        if calibrate:pick=(corrected<=1-threshold)&(t>=12)
        out['model_accept'][:,t]=pick;out['adjusted_risk'][:,t]=corrected;out['drift_scale'][:,t]=scale
        if audit_gate:
            distinct=(candidate!=ref).any(1);ids=ix[pick&distinct]
            if len(ids):
                # Freeze both policies before reading independent outcomes. Only differing groups sampled.
                mask=candidate[ids]!=ref[ids];obs=np.where(mask,audit[ids,t],np.nan)
                upper=audit_upper(candidate[ids],ref[ids],obs,audit_var[ids,t],alpha)
                out['audit_upper'][ids,t]=upper;out['audit_groups'][ids,t]=mask.sum(1)
                out['audit_cost'][ids,t]=(mask*PROBE_COST*.45/audit_var[ids,t]).sum(1)
                pick[ids]=upper<=EPS
        bias=.80*bias+.20*innovation;scale=.85*scale+.15*z2
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
        if quarantine and t>=12:
            discrepancy=((obs-means[ix,best])**2/(np.maximum(var[ix,best],cfg.var_floor)+rv)).mean(1)
            block=learn&(discrepancy>1.75);learn=learn&~block;out['quarantine'][:,t]=block
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
