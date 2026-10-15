#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from decision_tasks import weighted_channel
T=ROOT/'outputs/tables'; cfg=json.loads((ROOT/'config.json').read_text()); prior=float(cfg['channel_prior']); seed=int(cfg['seed'])
eta=np.linspace(0,1,int(cfg['eta_grid_n'])); reps=int(cfg['decision_bootstrap_replicates'])
assign=pd.read_csv(T/'crossfitted_decision_assignments.csv')

def fast_summary(ch,phi):
    m=phi@ch
    post=np.divide(phi[1]*ch[1],m,out=np.zeros_like(m),where=m>1e-12)
    E=eta[:,None]
    value=np.sum(m[None,:]*np.maximum(post[None,:]-E,0),axis=1)
    perfect=phi[1]*(1-eta)
    midl=np.maximum(perfect-value,0)
    action=post[None,:]>=E
    missed=phi[1]*np.sum(ch[1][None,:]*(~action),axis=1)
    over=phi[0]*np.sum(ch[0][None,:]*action,axis=1)
    return float(np.trapz(midl,eta)),float(np.max(midl)),float(np.trapz(missed,eta)),float(np.trapz(over,eta))

subjects=np.array(sorted(assign.Subject.unique())); keys=[]; contrib={}
for key,g in assign.groupby(['task','model']):
    mats=[]; states=[]
    for s in subjects:
        h=g[g.Subject.eq(s)]; m=np.zeros((2,2)); st=np.zeros(2)
        if len(h):
            w=1/len(h)
            for x,o in zip(h.X.astype(int),h.O.astype(int)):m[x,o]+=w;st[x]+=w
        mats.append(m);states.append(st)
    keys.append(key);contrib[key]=(np.stack(mats),np.stack(states))
rng=np.random.default_rng(seed+1); rows=[]
for rep in range(reps):
    idx=rng.integers(0,len(subjects),size=len(subjects))
    for task,model in keys:
        mats,states=contrib[(task,model)]; counts=mats[idx].sum(0); ch=counts+prior;ch/=ch.sum(1,keepdims=True); st=states[idx].sum(0)+prior;phi=st/st.sum()
        iv,mv,im,io=fast_summary(ch,phi)
        rows.append((rep,task,model,iv,mv,im,io))
b=pd.DataFrame(rows,columns=['replicate','task','model','integrated_midl','max_midl','integrated_missed_treatment','integrated_overtreatment'])
b.to_csv(T/'decision_bootstrap_long.csv',index=False)
b.groupby(['task','model'],as_index=False).agg(
 integrated_midl_median=('integrated_midl','median'),integrated_midl_ci_low=('integrated_midl',lambda x:x.quantile(.025)),integrated_midl_ci_high=('integrated_midl',lambda x:x.quantile(.975)),
 integrated_missed_median=('integrated_missed_treatment','median'),integrated_missed_ci_low=('integrated_missed_treatment',lambda x:x.quantile(.025)),integrated_missed_ci_high=('integrated_missed_treatment',lambda x:x.quantile(.975)),
 integrated_overtreatment_median=('integrated_overtreatment','median'),integrated_overtreatment_ci_low=('integrated_overtreatment',lambda x:x.quantile(.025)),integrated_overtreatment_ci_high=('integrated_overtreatment',lambda x:x.quantile(.975))
).to_csv(T/'decision_bootstrap_summary.csv',index=False)
print('done',len(b), 'rows')
