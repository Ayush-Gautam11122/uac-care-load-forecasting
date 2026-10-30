import pandas as pd, numpy as np, matplotlib, json; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sys; sys.path.insert(0,'src'); from forecast import features
import statsmodels.api as sm
plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
COL={"Naive":"#9e9e9e","MovingAvg7":"#b59f3b","ARIMA":"#2a9d8f","SARIMA":"#1b6f66","ETS":"#8e6bbf","RandomForest":"#e07a1f","GradientBoosting":"#c0392b"}
d=pd.read_csv('results/data_clean.csv',index_col=0,parse_dates=True)
# fig5 MAE by horizon, main+stress
fig,ax=plt.subplots(1,2,figsize=(8.5,3.4))
for a,tag,t in zip(ax,['main','stress'],['Main test (Apr–Dec 2025)','Stress test (Dec 2024–Mar 2025)']):
    m=pd.read_csv(f'results/metrics_hhs_care_{tag}.csv')
    for k,g in m.groupby('model'): a.plot(g.h,g.MAE,color=COL[k],label=k,lw=2.2 if k in('ETS','Naive') else 1.2)
    a.set_title(t); a.set_xlabel('Horizon (days)'); a.set_ylabel('MAE (children)')
ax[0].legend(frameon=False,fontsize=7); plt.tight_layout(); plt.savefig('figures/fig5_mae_horizon.png'); plt.close()
# fig6 h=7 backtest actual vs ETS vs naive
r=pd.read_csv('results/bt_hhs_care_main.csv',parse_dates=['origin','target_date']); r=r[(r.h==7)]
fig,ax=plt.subplots(figsize=(8,3.3)); a=r[r.model=='Naive'].set_index('target_date')
ax.plot(a.y_true,color='#1f4e79',label='Actual',lw=1.8)
for m in('ETS','Naive'): x=r[r.model==m].set_index('target_date'); ax.plot(x.y_pred,color=COL[m],label=f'{m} (7-day-ahead)',lw=1.2)
ax.axhline(2450,color='#c0392b',ls='--',lw=.8); ax.text(a.index[2],2455,'capacity threshold 2,450',color='#c0392b',fontsize=7)
ax.legend(frameon=False,fontsize=8); ax.set_ylabel('Children in HHS care'); ax.set_title('Walk-forward 7-day-ahead forecasts vs actual'); plt.tight_layout(); plt.savefig('figures/fig6_backtest_h7.png'); plt.close()
# fig7 final fan chart ETS
f=pd.read_csv('results/final_hhs_care.csv',parse_dates=['date']); iv=pd.read_csv('results/intervals_hhs_care.csv')
e=f[f.model=='ETS'].set_index('h'); q=iv[iv.model=='ETS'].set_index('h')
fig,ax=plt.subplots(figsize=(8,3.3)); h=d.hhs_care.iloc[-90:]; ax.plot(h,color='#1f4e79',label='Actual')
ax.fill_between(e.date,e.forecast+q.q025,e.forecast+q.q975,color='#8e6bbf',alpha=.18,label='95% interval'); ax.fill_between(e.date,e.forecast+q.q10,e.forecast+q.q90,color='#8e6bbf',alpha=.3,label='80% interval')
ax.plot(e.date,e.forecast,color='#8e6bbf',lw=2,label='ETS forecast'); ax.axhline(2450,color='#c0392b',ls='--',lw=.8)
ax.legend(frameon=False,fontsize=8,loc='upper left'); ax.set_title('14-day-ahead care-load forecast from 21 Dec 2025 (ETS, empirical intervals)'); ax.set_ylabel('Children in HHS care'); plt.tight_layout(); plt.savefig('figures/fig7_forecast.png'); plt.close()
print(e.forecast.round(0).tolist(),(e.forecast+q.q025).round(0).tolist(),(e.forecast+q.q975).round(0).tolist())
# RF importance h=7 on 2025 regime
F=features(d); y=(d.hhs_care.shift(-7)-d.hhs_care); X=F.dropna(); idx=X.index.intersection(y.dropna().index); 
rf=RandomForestRegressor(300,min_samples_leaf=3,max_features=.6,random_state=0,n_jobs=-1).fit(X.loc[idx].values,y.loc[idx].values)
imp=pd.Series(rf.feature_importances_,X.columns).sort_values(ascending=False); imp.to_csv('results/rf_importance.csv'); print(imp.head(8).round(3))
fig,ax=plt.subplots(figsize=(6,3)); imp.head(8)[::-1].plot.barh(ax=ax,color='#e07a1f'); ax.set_title('Random Forest feature importance (7-day change in care load)'); plt.tight_layout(); plt.savefig('figures/fig8_importance.png'); plt.close()
# pass-through: 7-day change in care vs 7-day sum of net pressure (2025 regime, observed-only gaps irrelevant: use interpolated)
s=d.loc['2025-03-15':]; x=s.net_pressure.rolling(7).sum().shift(7); yy=s.hhs_care.diff(7); k=pd.concat([x,yy],axis=1).dropna(); k.columns=['x','y']
res=sm.OLS(k.y,sm.add_constant(k.x)).fit(); print('passthrough slope',res.params.x,res.rsquared)
json.dump({'passthrough_slope':float(res.params.x),'r2':float(res.rsquared)},open('results/passthrough.json','w'))
# discharges: fraction of variance explained, naive CV
print('discharge mean/std 2025',d.loc['2025-04-01':].discharged.mean(),d.loc['2025-04-01':].discharged.std())
# test-window facts
t=d.loc['2025-04-01':]; print('test care range',t.hhs_care.min(),t.hhs_care.max())
