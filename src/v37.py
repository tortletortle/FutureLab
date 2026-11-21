"""Active discriminative verification of a bounded archive (independent episode API).
All estimators see observed Gaussian contrasts only. Novelty is a composite mixture.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.hermite import hermgauss
from v35 import C, log_normal

@dataclass
class Belief:
    mean: np.ndarray  # B,H,G
    var: np.ndarray
    logw: np.ndarray  # B,H
    known: int

    def weights(self):
        return np.exp(self.logw)

    def posterior_mean(self):
        return np.einsum('bh,bhg->bg',self.weights(),self.mean)

    def confidence(self):
        w=self.weights()[:,:self.known]
        return w.max(1),w.argmax(1),w.sum(1)


def build_belief(archive_mean,archive_var,structured=True,novel_prior=.5):
    m=np.asarray(archive_mean,float);v=np.asarray(archive_var,float)
    if m.ndim!=3 or m.shape!=v.shape or min(m.shape)<1 or not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v<=0) or not 0<novel_prior<1:
        raise ValueError('invalid archive')
    B,K,G=m.shape
    means=[m,np.full((B,1,G),C)];variances=[v,np.full((B,1,G),4.)]
    w=[np.full(K,(1-novel_prior)/K),np.array([novel_prior*(.5 if structured else 1)])]
    if structured:
        # Dense modest departure and sparse one-coordinate departure from any known regime.
        means += [m,np.repeat(m,G,axis=1)]
        variances += [v+.55**2,np.repeat(v,G,axis=1)+2.0**2*np.tile(np.eye(G),(K,1))[None]]
        w += [np.full(K,novel_prior*.25/K),np.full(K*G,novel_prior*.25/(K*G))]
    mean=np.concatenate(means,axis=1);var=np.concatenate(variances,axis=1);lp=np.log(np.concatenate(w));return Belief(mean.copy(),var.copy(),np.broadcast_to(lp,(B,len(lp))).copy(),K)


def update_all(b,y,r):
    y=np.asarray(y,float);r=np.asarray(r,float)
    if y.shape!=b.mean[:,0].shape or r.shape!=y.shape or not np.isfinite(y).all() or not np.isfinite(r).all() or (r<=0).any():raise ValueError('invalid discovery statistics')
    total=b.var+r[:,None];lp=b.logw+log_normal(y[:,None],b.mean,total).sum(2)
    b.logw=lp-logsumexp(lp,axis=1,keepdims=True);b.mean+=b.var/total*(y[:,None]-b.mean);b.var*=r[:,None]/total


def update_probe(b,j,y,r):
    j=np.asarray(j);y=np.asarray(y,float);r=np.asarray(r,float);B,H,G=b.mean.shape
    if j.shape!=(B,) or not np.issubdtype(j.dtype,np.integer) or np.any((j<0)|(j>=G)) or y.shape!=(B,) or r.shape!=(B,) or not np.isfinite(y).all() or not np.isfinite(r).all() or (r<=0).any():raise ValueError('invalid probe')
    ix=(np.arange(B)[:,None],np.arange(H)[None,:],j[:,None]);m=b.mean[ix];v=b.var[ix];total=v+r[:,None];lp=b.logw+log_normal(y[:,None],m,total)
    b.logw=lp-logsumexp(lp,axis=1,keepdims=True);b.mean[ix]=m+v/total*(y[:,None]-m);b.var[ix]=v*r[:,None]/total


def binary_moments(b,r):
    """Moment match two predictive mixtures, known versus novel, per candidate group."""
    r=np.asarray(r,float)
    if r.shape!=b.mean[:,0].shape or not np.isfinite(r).all() or np.any(r<=0):raise ValueError('invalid anticipated probe variance')
    mus=[];vs=[];lp=[]
    for sl in (slice(0,b.known),slice(b.known,None)):
        lw=b.logw[:,sl];z=logsumexp(lw,axis=1,keepdims=True);w=np.exp(lw-z);m=np.einsum('bh,bhg->bg',w,b.mean[:,sl]);v=np.einsum('bh,bhg->bg',w,b.var[:,sl]+(b.mean[:,sl]-m[:,None])**2)
        mus.append(m);vs.append(v+r);lp.append(z[:,0])
    return np.stack(mus,2),np.stack(vs,2),np.stack(lp,1)


def binary_gain(b,r,nodes=15):
    """Approximate binary novelty mutual information via moment-matched Gaussian
    predictive distributions and Gauss-Hermite quadrature, in nats. Actual belief
    updates remain full-mixture, not moment matched.
    """
    m,v,lp=binary_moments(b,r);x,q=hermgauss(nodes);q=q/np.sqrt(np.pi)
    y=m[:,:,:,None]+np.sqrt(2*v[:,:,:,None])*x
    ll=log_normal(y[:,:,:,:,None],m[:,:,None,None,:],v[:,:,None,None,:]);lpy=logsumexp(ll+lp[:,None,None,None,:],axis=-1)
    own=np.stack([ll[:,:,0,:,0],ll[:,:,1,:,1]],axis=2)
    return np.maximum(0,np.sum((own-lpy)*np.exp(lp)[:,None,:,None]*q,axis=(2,3)))


def choose(b,r,method,rng=None):
    if method=='random':
        if rng is None:raise ValueError('rng required')
        return rng.integers(b.mean.shape[2],size=b.mean.shape[0])
    if method=='active':s=binary_gain(b,r)
    elif method=='uncertainty':
        m=b.posterior_mean();s=np.einsum('bh,bhg->bg',b.weights(),b.var+(b.mean-m[:,None])**2)/r
    elif method=='disagreement':
        m=b.posterior_mean();s=np.einsum('bh,bhg->bg',b.weights(),(b.mean-m[:,None])**2)/r
    elif method=='decision':
        # Simple decision-boundary heuristic, not EVSI.
        m=b.posterior_mean();sd=np.sqrt(np.einsum('bh,bhg->bg',b.weights(),b.var+(b.mean-m[:,None])**2));s=sd*np.exp(-np.abs(m-C)/np.maximum(sd,1e-12))/np.sqrt(r)
    else:raise ValueError('unknown selector')
    return s.argmax(1)


def run_episode(archive_mean,archive_var,discovery,discovery_var,probe_bank,probe_var,method='active',budget=3,structured=True,seed=1):
    """Probe bank [B,G,budget] is evaluator-owned; policy chooses before selected
    value is read. Common random numbers by group/visit support paired evaluation.
    All probe variances are known design quantities; unselected outcomes never enter belief.
    """
    b=build_belief(archive_mean,archive_var,structured);update_all(b,discovery,discovery_var)
    B,H,G=b.mean.shape;bank=np.asarray(probe_bank);r=np.asarray(probe_var,float)
    if not isinstance(budget,int) or budget<0 or bank.ndim!=3 or bank.shape[:2]!=(B,G) or bank.shape[2]<budget or r.shape!=(B,G) or not np.isfinite(r).all() or (r<=0).any():raise ValueError('invalid probe design')
    rng=np.random.default_rng(seed);visits=np.zeros((B,G),int);trace=[];history=[];rows=np.arange(B)
    for t in range(budget+1):
        conf,slot,known=b.confidence();history.append(dict(conf=conf.copy(),slot=slot.copy(),known=known.copy(),mean=b.posterior_mean().copy()))
        if t==budget:break
        j=choose(b,r,method,rng);trace.append(j.copy());z=bank[rows,j,visits[rows,j]];visits[rows,j]+=1;update_probe(b,j,z,r[rows,j])
    return dict(history=history,choices=np.stack(trace,1) if trace else np.empty((B,0),int),visits=visits)
