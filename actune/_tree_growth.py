"""Loss-directed tree growth and exact subtree pruning."""
from copy import deepcopy
import numpy as np

def grow(x,losses,max_depth=5,min_leaf=20):
    x=np.asarray(x,float);losses=np.asarray(losses,float)
    if x.ndim!=2 or losses.shape!=(len(x),4) or not np.isfinite(x).all() or not np.isfinite(losses).all():raise ValueError('Invalid fitting arrays')
    def node(indices,depth):
        total=losses[indices].sum(0);q=int(total.argmin());out=dict(profile_index=q,risk=float(total[q]),samples=len(indices),depth=depth)
        if depth>=max_depth or len(indices)<2*min_leaf:return out
        best=None
        for j in range(x.shape[1]):
            order=indices[np.argsort(x[indices,j],kind='stable')];values=x[order,j]
            cuts=np.arange(min_leaf,len(order)-min_leaf+1);cuts=cuts[values[cuts-1]<values[cuts]]
            if not len(cuts):continue
            sums=np.cumsum(losses[order],axis=0);cost=sums[cuts-1].min(1)+(total-sums[cuts-1]).min(1)
            z=int(cost.argmin());gain=out['risk']-float(cost[z])
            if gain>1e-12 and (best is None or gain>best[0]+1e-12):best=(gain,j,int(cuts[z]),order)
        if best is None:return out
        gain,j,cut,order=best;v0=x[order[cut-1],j];v1=x[order[cut],j];threshold=float(v0+(v1-v0)/2)
        out.update(feature=j,threshold=threshold,gain=gain,left=node(order[:cut],depth+1),right=node(order[cut:],depth+1))
        return out
    return node(np.arange(len(x)),0)

def candidates(tree,n,*,all_sizes=False):
    def dp(t):
        leaf={k:v for k,v in t.items() if k not in ['left','right','feature','threshold','gain']};choices={1:(t['risk'],leaf)}
        if 'left' in t:
            for kl,(rl,tl) in dp(t['left']).items():
                for kr,(rr,tr) in dp(t['right']).items():
                    k=kl+kr;r=rl+rr
                    if k not in choices or r<choices[k][0]-1e-12:
                        q={key:val for key,val in t.items() if key not in ['left','right']};q.update(left=tl,right=tr);choices[k]=(r,q)
        return choices
    alltrees=dp(tree);out=[]
    for k,(risk,t) in sorted(alltrees.items()):
        low=max([0.]+[(risk-rj)/n/(j-k) for j,(rj,_) in alltrees.items() if j>k])
        high=min([(rj-risk)/n/(k-j) for j,(rj,_) in alltrees.items() if j<k],default=float('inf'))
        on_path=high>=low-1e-12
        if on_path or all_sizes:out.append(dict(leaves=k,fitting_loss=risk/n,lambda_min=low if on_path else None,lambda_max=high if on_path and np.isfinite(high) else None,tree=deepcopy(t)))
    return out

def predict(tree,x):
    out=[]
    for row in x:
        t=tree
        while 'left' in t:t=t['left'] if row[t['feature']]<=t['threshold'] else t['right']
        out.append(t['profile_index'])
    return np.asarray(out,int)
