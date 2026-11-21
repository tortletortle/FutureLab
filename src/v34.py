"""V0.34 research extension. All simulated truth restricted to evaluator.
Gaussian arm sufficient statistics emulate balanced randomized independent arms.
"""
import numpy as np
from scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss

G,T,N,C=12,48,20,1.5
WINDOWS=(2,4,8)
SCENES=('stable','noise','global','cancel','local','repeated','mixed','gradual')
METHODS=('cumulative','rolling4','rolling8','fixed_global_reset','local_reset','local_guard')

def generate(seed,scene):
    rng=np.random.default_rng(seed)
    truth=np.tile(C+rng.normal(0,1.4,G),(T,1)); sig=np.full((T,G),2.5)
    if scene in ('noise','mixed'): sig[16:32]*=2; sig[32:]*=.7
    if scene=='global': truth[16:]+=.9
    if scene in ('cancel','mixed','repeated'): truth[16:,:6]+=1.4;truth[16:,6:]-=1.4
    if scene=='repeated': truth[32:,:6]-=2.8;truth[32:,6:]+=2.8
    if scene=='local': truth[16:,0]+=2.4
    if scene=='gradual': truth[16:]+=np.linspace(0,1,T-16)[:,None]*np.r_[np.full(6,1.4),np.full(6,-1.4)]
    # Independent arm means and unbiased within-arm sample variances (normal model).
    def sample():
        y=truth+rng.normal(size=(T,G))*sig*np.sqrt(2/N)
        var=sig**2*rng.chisquare(2*(N-1),size=(T,G))/(2*(N-1))*2/N
        return y,var
    y,v=sample(); h,hv=sample()
    return truth,y,v,h,hv

def score(y,v,known=False):
    """Adjacent equal windows, only observations at or before current period."""
    z=np.zeros((T,G)); global_z=np.zeros(T); noise=np.zeros((T,G))
    for w in WINDOWS:
        for t in range(2*w-1,T):
            a=slice(t-2*w+1,t-w+1); b=slice(t-w+1,t+1)
            delta=y[b].mean(0)-y[a].mean(0)
            va=v[a].sum(0)/w**2; vb=v[b].sum(0)/w**2
            denom=np.full(G,2*(2.5**2*2/N)/w) if known else va+vb
            zz=delta/np.sqrt(denom)
            z[t]=np.maximum(z[t],abs(zz))
            global_z[t]=max(global_z[t],abs(delta.sum())/np.sqrt(denom.sum()))
            noise[t]=np.maximum(noise[t],abs(np.log(v[b].mean(0)/v[a].mean(0))))
    return z,global_z,noise

def rank_threshold(x,alpha=.05):
    x=np.sort(np.asarray(x)); k=int(np.ceil((len(x)+1)*(1-alpha)))
    return float(x[k-1]) if k<=len(x) else float('inf')

def calibrate(n=500):
    # Separate group-wise and global policies; noise alarm is separately calibrated.
    null={s:[] for s in ('stable','noise')}
    for scene in null:
        for i in range(n):
            _,y,v,_,_=generate(100000+i+(scene=='noise')*10000,scene)
            z,g,nz=score(y,v); old,og,_=score(y,v,True)
            null[scene].append([z.max(),g.max(),nz.max(),old.max(),og.max()])
    a,b=(np.array(null[s]) for s in ('stable','noise'))
    return dict(local=max(rank_threshold(a[:,0]),rank_threshold(b[:,0])),global_=max(rank_threshold(a[:,1]),rank_threshold(b[:,1])),noise=rank_threshold(a[:,2]),fixed_local=rank_threshold(a[:,3]),fixed_global=rank_threshold(a[:,4]))

def event_mask(z,threshold,cooldown=6):
    z=np.asarray(z); a=np.zeros_like(z,dtype=bool); last=np.full(z.shape[1:],-cooldown)
    for t in range(len(z)):
        a[t]=(z[t]>threshold)&(t-last>=cooldown);last=np.where(a[t],t,last)
    return a

def policy(y,v,h,hv,method,local,global_,guard_z=1.28):
    """Current decision strictly precedes current data; validation stream never fits candidate."""
    deployed=np.zeros(G,bool); starts=np.zeros(G,int); actions=[]; accepts=0; proposals=0
    a=np.vstack([np.zeros(G),np.cumsum(y/v,axis=0)]); b=np.vstack([np.zeros(G),np.cumsum(1/v,axis=0)]); jj=np.arange(G)
    for t in range(T):
        actions.append(deployed.copy())
        if method in ('rolling4','rolling8'): starts[:]=max(0,t-int(method[-1])+1)
        if method=='fixed_global_reset' and global_[t]: starts[:]=t
        if method in ('local_reset','local_guard'): starts[local[t]]=t
        # Match all arms on variance-weighted fitting and identical sample budgets.
        m=(a[t+1]-a[starts,jj])/(b[t+1]-b[starts,jj])
        candidate=m>C
        if method=='local_guard' and np.any(candidate!=deployed):
            # independent randomized validation contrasts, observed after fixing candidate.
            d=candidate.astype(int)-deployed.astype(int); proposals+=1
            est=np.sum(d*(h[t]-C))/G; se=np.sqrt(np.sum(d*d*hv[t]))/G
            if est>guard_z*se: deployed=candidate;accepts+=1
        else: deployed=candidate
    return np.array(actions),proposals,accepts

def metrics(actions,truth):
    net=truth-C; oracle=np.maximum(net,0).mean(1); value=(actions*net).mean(1)
    return dict(regret=float((oracle-value)[16:].mean()),value=float(value[16:].mean()),harm=float((actions*np.maximum(-net,0)).mean(1)[16:].mean()))

def correlated_update(m,S,j,y,r):
    k=S[:,j]/(S[j,j]+r); return m+k*(y-m[j]),S-np.outer(k,S[j,:])

def gain(m,S,w,c=C,r=4.):
    """Exact one-step expected terminal value improvement, including cross-group updates."""
    sd=np.abs(S)/np.sqrt(np.diag(S)+r)[None,:]; a=np.abs(m-c)[:,None]
    z=np.divide(a,sd,out=np.zeros_like(sd),where=sd>0)
    return (w[:,None]*np.where(sd>0,np.maximum(0,sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)-a*ndtr(-z)),0)).sum(0)

NODES,WEIGHTS=hermgauss(9);NODES=NODES*np.sqrt(2);WEIGHTS=WEIGHTS/np.sqrt(np.pi)
def two_step(m,S,w,r=4.):
    """One-step exact gain + quadrature expected best remaining one-step gain."""
    first=gain(m,S,w,r=r); out=first.copy()
    for j in range(len(m)):
        scale=S[:,j]/np.sqrt(S[j,j]+r); updated=S-np.outer(scale,scale)
        out[j]+=sum(q*gain(m+z*scale,updated,w,r=r).max() for z,q in zip(NODES,WEIGHTS))
    return out

def static_trial(seed,scene,budget=8):
    rng=np.random.default_rng(seed);g=6;m0=C+rng.uniform(-2,2,g); sd=rng.uniform(.8,1.8,g);w=rng.lognormal(0,.6,g);w/=w.sum()
    S=np.outer(sd,sd)*.65+np.diag(sd**2*.35); diag=np.diag(sd**2)
    true_cov=S if scene=='matched' else diag if scene=='independent' else np.outer(sd,sd)*(-.15)+np.diag(sd**2*1.15)
    truth=rng.multivariate_normal(m0,true_cov); ys=truth[:,None]+rng.normal(0,2,(g,budget)); rows=[]
    for method in ('random','independent_evsi','correlated_evsi','two_step'):
        m=m0.copy(); P=(diag if method=='independent_evsi' else S).copy(); visits=np.zeros(g,int); rg=np.random.default_rng(seed+9_000_000)
        for t in range(budget):
            j=int(rg.integers(g)) if method=='random' else int(np.argmax(two_step(m,P,w) if method=='two_step' and t+1<budget else gain(m,P,w)))
            y=ys[j,visits[j]]; visits[j]+=1;m,P=correlated_update(m,P,j,y,4.)
        val=np.sum(w*(m>C)*(truth-C));oracle=np.sum(w*np.maximum(truth-C,0));rows.append(dict(seed=seed,scene=scene,method=method,regret=oracle-val,value=val,experiments=int(visits.sum())))
    return rows

def dense_score(y,v):
    """Max over windows of sum of squared studentized stratum contrasts."""
    out=np.zeros(T)
    for w in WINDOWS:
        for t in range(2*w-1,T):
            a=slice(t-2*w+1,t-w+1); b=slice(t-w+1,t+1)
            delta=y[b].mean(0)-y[a].mean(0)
            out[t]=max(out[t],np.sum(delta**2/(v[a].sum(0)/w**2+v[b].sum(0)/w**2)))
    return out

def hybrid_policy(y,v,local,dense):
    """Exploratory dense reset, local reset, and bounded history after dense alert."""
    deployed=np.zeros(G,bool); starts=np.zeros(G,int); actions=[]; until=-1
    a=np.vstack([np.zeros(G),np.cumsum(y/v,axis=0)]); b=np.vstack([np.zeros(G),np.cumsum(1/v,axis=0)]); jj=np.arange(G)
    for t in range(T):
        actions.append(deployed.copy())
        if dense[t]: starts[:]=max(0,t-1);until=t+8
        if t<=until:starts=np.maximum(starts,max(0,t-3))
        starts[local[t]]=t
        m=(a[t+1]-a[starts,jj])/(b[t+1]-b[starts,jj]);deployed=m>C
    return np.array(actions)
