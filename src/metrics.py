import pandas as pd, numpy as np, json
CAP=2450          # default capacity-stress threshold (≈ 97th pct of 2025 post-surge-regime load)
MODELS=['Naive','MovingAvg7','ARIMA','SARIMA','ETS','RandomForest','GradientBoosting']
def load(t,tag):
    r=pd.read_csv(f'results/bt_{t}_{tag}.csv',parse_dates=['origin','target_date']); return r[r.observed==1].copy()
def metr(r):
    r=r.assign(e=r.y_true-r.y_pred); r['ae']=r.e.abs(); r['se']=r.e**2; r['ape']=r.ae/r.y_true.replace(0,np.nan)*100
    g=r.groupby(['model','h']).agg(MAE=('ae','mean'),RMSE=('se',lambda s:np.sqrt(s.mean())),MAPE=('ape','mean'),n=('ae','size')).reset_index(); return g
out={}
for t in ('hhs_care','discharged'):
    for tag in ('main','stress'):
        m=metr(load(t,tag)); m.to_csv(f'results/metrics_{t}_{tag}.csv',index=False)
        s=m[m.h.isin([1,3,7,14])].pivot(index='model',columns='h',values='MAE').reindex(MODELS); print(t,tag,'MAE by horizon\n',s.round(1).to_string(),'\n')
# --- best model (mean MAE over h=1..14 on main window) -------------------------
best={}
for t in ('hhs_care','discharged'):
    m=pd.read_csv(f'results/metrics_{t}_main.csv'); best[t]=m.groupby('model').MAE.mean().sort_values().index[0]
    print(t,'best:',best[t],m.groupby('model').MAE.mean().round(1).to_dict())
# --- empirical prediction intervals per model/h (from main-window residuals) ---
for t in ('hhs_care','discharged'):
    r=load(t,'main'); r['e']=r.y_true-r.y_pred
    q=r.groupby(['model','h']).e.quantile([.025,.10,.90,.975]).unstack().reset_index(); q.columns=['model','h','q025','q10','q90','q975']
    q.to_csv(f'results/intervals_{t}.csv',index=False)
    # out-of-sample coverage: calibrate on first half of origins, test on second half
    cut=r.origin.quantile(.5); a=r[r.origin<=cut]; b=r[r.origin>cut]
    qa=a.groupby(['model','h']).e.quantile([.025,.975]).unstack().reset_index(); qa.columns=['model','h','lo','hi']
    bb=b.merge(qa,on=['model','h']); bb['cov']=((bb.e>=bb.lo)&(bb.e<=bb.hi)); cov=bb.groupby('model')['cov'].mean()
    cov.to_csv(f'results/coverage95_{t}.csv'); print(t,'95% PI out-of-sample coverage (calibrate H1 -> test H2):\n',cov.round(3).to_dict())
# --- KPIs ----------------------------------------------------------------------
r=pd.read_csv('results/bt_hhs_care_main.csv',parse_dates=['origin','target_date'])
iv=pd.read_csv('results/intervals_hhs_care.csv')
d=pd.read_csv('results/data_clean.csv',index_col=0,parse_dates=True)
act=d.hhs_care
# breach events: first day load >= CAP after >=7 days below
ev=[]; below=0
for dt,v in act.loc['2025-04-01':].items():
    if v>=CAP and below>=7: ev.append(dt)
    below = below+1 if v<CAP else 0
print('breach events',ev)
def breach_stats(model):
    x=r[r.model==model].set_index(['origin','h'])
    leads=[]
    for e in ev:
        L=0
        for h in range(14,0,-1):          # earliest origin (largest h) that already forecast >= CAP
            o=e-pd.Timedelta(days=h)
            if (o,h) in x.index and x.loc[(o,h),'y_pred']>=CAP: L=h; break
        leads.append(L)
    # h=7 classification of "load >= CAP" on days observed
    z=r[(r.model==model)&(r.h==7)&(r.observed==1)]; yt=z.y_true>=CAP; yp=z.y_pred>=CAP
    tp=(yt&yp).sum(); fp=(~yt&yp).sum(); fn=(yt&~yp).sum()
    # probability via empirical residual dist
    e=(z.y_true-z.y_pred).values; p=[(( (c)>=CAP-0).mean()) for c in [z.y_pred.values[i]+e for i in range(0)]] if False else None
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1)
    return dict(model=model,events=len(ev),mean_lead_days=float(np.mean(leads)) if leads else np.nan,detected=int(sum(l>0 for l in leads)),precision_h7=float(prec),recall_h7=float(rec),fp=int(fp),fn=int(fn))
bs=pd.DataFrame([breach_stats(m) for m in MODELS]); bs.to_csv('results/kpi_breach.csv',index=False); print(bs.round(2).to_string())
# Brier score of breach probability at h=7 (probabilities from empirical residuals of other half)
z=r[(r.h==7)&(r.observed==1)].copy(); z['e']=z.y_true-z.y_pred; brier={}
for m in MODELS:
    zm=z[z.model==m].sort_values('origin'); E=zm.e.values; P=zm.y_pred.values; Y=(zm.y_true.values>=CAP).astype(float)
    p=np.array([ (P[i]+E>=CAP).mean() for i in range(len(P))]); brier[m]=float(((p-Y)**2).mean())
print('Brier h=7',brier)
# Forecast stability index: 1 - CV of monthly MAE (h=7) ; robustness = skill vs naive in stress window
FSI={}; SK={}
for m in MODELS:
    zm=r[(r.model==m)&(r.h==7)&(r.observed==1)].copy(); zm['ae']=(zm.y_true-zm.y_pred).abs(); mm=zm.groupby(zm.target_date.dt.to_period('M')).ae.mean()
    FSI[m]=float(max(0,1-mm.std()/mm.mean()))
ms=pd.read_csv('results/metrics_hhs_care_stress.csv'); mm_=pd.read_csv('results/metrics_hhs_care_main.csv')
for m in MODELS:
    s=ms[(ms.model==m)&(ms.h==7)].MAE.iloc[0]; n=ms[(ms.model=='Naive')&(ms.h==7)].MAE.iloc[0]; s2=mm_[(mm_.model==m)&(mm_.h==7)].MAE.iloc[0]; n2=mm_[(mm_.model=='Naive')&(mm_.h==7)].MAE.iloc[0]
    SK[m]=dict(skill_main=float(1-s2/n2),skill_stress=float(1-s/n))
kp=[]
for m in MODELS:
    mr=mm_[(mm_.model==m)&(mm_.h==7)].iloc[0]
    kp.append(dict(model=m,accuracy_h7=100-mr.MAPE,mae_h7=mr.MAE,FSI=FSI[m],brier_h7=brier[m],skill_main=SK[m]['skill_main'],skill_stress=SK[m]['skill_stress']))
kp=pd.DataFrame(kp).merge(bs[['model','mean_lead_days','detected','events','precision_h7','recall_h7']],on='model'); kp.to_csv('results/kpis.csv',index=False); print(kp.round(3).to_string())
json.dump({'best':best,'CAP':CAP,'breach_events':[str(e.date()) for e in ev]},open('results/summary.json','w'),indent=1)
