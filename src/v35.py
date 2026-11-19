"""Adaptive causal memory: predictive expert weights and Bayesian covariance model averaging.
All inputs to policies are observation statistics, not simulator truth. NumPy batch axis B.
"""
import numpy as np
from scipy.special import logsumexp, ndtr
from numpy.polynomial.hermite import hermgauss
from functools import lru_cache
from v34 import C, G, T, generate

DISCOUNTS=np.array([1.,.97,.85,.65,.35])
SHARE=.035
DYNAMIC_SCENES=('stable','noise','global','cancel','local','repeated','mixed','gradual','recurrence','slow_local','smooth_noise','random_walk')
METHODS=('cumulative','rolling4','rolling8','exp85','exp65','adaptive_local','adaptive_global','adaptive_blend','hybrid_v34')

def extra_generate(seed,scene):
    if scene in DYNAMIC_SCENES[:8]: return generate(seed,scene)[:3]
    rng=np.random.default_rng(seed); base=C+rng.normal(0,1.4,G); truth=np.tile(base,(T,1)); sig=np.full((T,G),2.5)
    if scene=='recurrence':
        shift=np.r_[np.full(6,1.6),np.full(6,-1.6)];truth[12:24]+=shift;truth[36:]+=shift
    if scene=='slow_local': truth[16:,:2]+=np.linspace(0,3.2,T-16)[:,None]*[1,-1]
    if scene=='smooth_noise':
        truth[16:]+=np.linspace(0,1,T-16)[:,None]*np.r_[np.full(6,1.8),np.full(6,-1.8)];sig[16:]*=np.linspace(1,2.4,T-16)[:,None]
    if scene=='random_walk': truth[16:]+=np.cumsum(rng.normal(0,.28,(T-16,G)),0)
    y=truth+rng.normal(size=(T,G))*sig*np.sqrt(2/20)
    v=sig**2*rng.chisquare(38,size=(T,G))/38*2/20
    return truth,y,v

def log_normal(y,m,v): return -.5*(np.log(2*np.pi*v)+(y-m)**2/v)

def adaptive(y,v,share=SHARE,blend=.5,discounts=DISCOUNTS):
    """y,v [B,T,G]. Expert power-prior precision discount; fixed-share log score update.
    For each period output the mixture prediction BEFORE viewing the period.
    Local and global model probabilities share the same expert filters, no extra measurements.
    """
    y=np.asarray(y,float);v=np.asarray(v,float);d=np.asarray(discounts,float)
    if y.ndim!=3 or y.shape!=v.shape or not np.isfinite(y).all() or not np.isfinite(v).all() or (v<=0).any() or not 0<=share<1 or (d<=0).any() or (d>1).any():raise ValueError('invalid observations or filter parameters')
    B,H,K=y.shape;E=len(d); m=np.full((B,E,K),C);p=np.full_like(m,1/4.)
    local=np.full_like(m,1/E);glob=np.full((B,E,1),1/E)
    means={k:[] for k in ('adaptive_local','adaptive_global','adaptive_blend')}; losses={k:[] for k in means}; trace=[]
    for t in range(H):
        # transition and weight switching happen before seeing y_t
        pp=p*d[None,:,None]; variance=1/pp+v[:,t,None,:]
        l=(1-share)*local+share/E;g=(1-share)*glob+share/E
        ll=log_normal(y[:,t,None,:],m,variance)
        for name,w in [('adaptive_local',l),('adaptive_global',g),('adaptive_blend',blend*l+(1-blend)*g)]:
            means[name].append((w*m).sum(1));losses[name].append(-logsumexp(np.log(w)+ll,axis=1))
        local=np.exp(np.log(l)+ll-logsumexp(np.log(l)+ll,axis=1,keepdims=True))
        joint=np.log(g)+ll.sum(2,keepdims=True);glob=np.exp(joint-logsumexp(joint,axis=1,keepdims=True))
        p=pp+1/v[:,t,None,:];m=(pp*m+y[:,t,None,:]/v[:,t,None,:])/p
        trace.append((local*d[None,:,None]).sum(1))
    return {k:np.stack(a,1) for k,a in means.items()},{k:np.stack(a,1) for k,a in losses.items()},np.stack(trace,1)

def fixed(y,v,method):
    B,H,K=y.shape;mu=np.full((B,K),C);p=np.full_like(mu,.25); out=[]; loss=[]
    discount={'cumulative':1.,'exp85':.85,'exp65':.65}.get(method,1.)
    for t in range(H):
        p=p*discount;out.append(mu.copy());loss.append(-log_normal(y[:,t],mu,1/p+v[:,t]));p=p+1/v[:,t];mu=(mu*(p-1/v[:,t])+y[:,t]/v[:,t])/p
        if method.startswith('rolling'):
            start=max(0,t-int(method[7:])+1);p=(1/v[:,start:t+1]).sum(1);mu=(y[:,start:t+1]/v[:,start:t+1]).sum(1)/p
    return np.stack(out,1),np.stack(loss,1)

# Gaussian covariance hypothesis mixture: [model,group] posterior means,
# [model,group,group] covariance; categorical posterior weights given ALL sampled data.
RHOS=np.array([-.15,0.,.35,.65])
def covariance(sd,rho):
    s=np.asarray(sd);return rho*np.outer(s,s)+np.diag(s*s*(1-rho))

def update(m,S,w,j,y,r=4.):
    z=S[:,j,j]+r; lp=np.log(np.maximum(w,1e-300))+log_normal(y,m[:,j],z); nw=np.exp(lp-logsumexp(lp))
    k=S[:,:,j]/z[:,None];nm=m+k*(y-m[:,j])[:,None];ns=S-k[:,:,None]*S[:,None,j,:]
    return nm,ns,nw

@lru_cache(None)
def quadrature(nodes):
    x,q=hermgauss(nodes);return x*np.sqrt(2),q/np.sqrt(np.pi)

def single_gain(m,S,w,c=C,r=4.):
    sd=np.abs(S)/np.sqrt(np.diag(S)+r)[None,:];a=np.abs(m-c)[:,None];z=np.divide(a,sd,out=np.zeros_like(sd),where=sd>0)
    return (w[:,None]*np.maximum(0,sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)-a*ndtr(-z))).sum(0)

def mixture_gain(m,S,w,pop,r=4.,nodes=25):
    """Expected optimal binary terminal value over predictive Gaussian mixture.
    For each candidate j, integrate all originating hypotheses h; at each sample,
    recompute posterior weights for ALL hypotheses and take action using mixture mean.
    """
    if len(w)==1:return single_gain(m[0],S[0],pop,r=r)
    x,q=quadrature(nodes);base=pop@np.maximum(w@m-C,0);A=m.shape[1];out=np.zeros(A)
    for j in range(A):
        var=S[:,j,j]+r; k=S[:,:,j]/var[:,None]
        ys=(m[:,j,None]+np.sqrt(var[:,None])*x).ravel() # [origin*node]
        lp=log_normal(ys[:,None],m[None,:,j],var[None,:])+np.log(np.maximum(w,1e-300))[None,:]
        pw=np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
        new=m[None,:,:]+(ys[:,None]-m[None,:,j])[:,:,None]*k[None,:,:]
        mu=(pw[:,:,None]*new).sum(1);val=np.maximum(mu-C,0)@pop
        out[j]=max(0,(val.reshape(len(w),nodes)*w[:,None]*q).sum()-base)
    return out

def static_trial(seed,scene,budget=8):
    rng=np.random.default_rng(seed);g=6;base=C+rng.uniform(-2,2,g);sd=rng.uniform(.8,1.8,g);pop=rng.lognormal(0,.6,g);pop/=pop.sum()
    rho={'matched':.65,'independent':0.,'negative':-.15,'intermediate':.2,'strong':.9,'block':0.}[scene]
    true=covariance(sd,rho)
    if scene=='block':
        R=np.eye(g);R[:3,:3]=.85;R[3:,3:]=.85;np.fill_diagonal(R,1);true=R*np.outer(sd,sd)
    truth=rng.multivariate_normal(base,true);ys=truth[:,None]+rng.normal(0,2,(g,budget));rows=[]
    for name,rhos in [('independent',[0.]),('fixed_positive',[.65]),('shrinkage',[.25]),('mixture',RHOS),('mixture_random',RHOS)]:
        m=np.tile(base,(len(rhos),1));S=np.array([covariance(sd,rho) for rho in rhos]);w=np.full(len(rhos),1/len(rhos));visits=np.zeros(g,int);rg=np.random.default_rng(seed+9000000); gaps=[]; logloss=0
        for t in range(budget):
            if name=='mixture_random':j=int(rg.integers(g))
            else:
                gains=mixture_gain(m,S,w,pop);j=int(np.argmax(gains))
                if name=='mixture' and t==0:
                    refined=mixture_gain(m,S,w,pop,nodes=61);gaps.append((abs(refined-gains).max(),float(np.argmax(refined)!=j)))
            y=ys[j,visits[j]];visits[j]+=1;logloss-=logsumexp(np.log(np.maximum(w,1e-300))+log_normal(y,m[:,j],S[:,j,j]+4.));m,S,w=update(m,S,w,j,y)
        mean=w@m;value=pop@((mean>C)*(truth-C));regret=pop@np.maximum(truth-C,0)-value
        rows.append(dict(seed=seed,scene=scene,method=name,regret=regret,value=value,mse=((mean-truth)**2).mean(),sample_nll=logloss/budget,visits=int(visits.sum()),rho_post=w@np.array(rhos),quad_gap=gaps[0][0] if gaps else np.nan,quad_argmax_change=gaps[0][1] if gaps else np.nan))
    return rows
