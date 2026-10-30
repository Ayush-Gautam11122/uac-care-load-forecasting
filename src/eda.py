import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, json
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
plt.rcParams.update({'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False,'font.size':9})
d=pd.read_csv('results/data_clean.csv',index_col=0,parse_dates=True); o=d[d.observed==1]
C='#1f4e79'; A='#c0504d'
fig,ax=plt.subplots(2,1,figsize=(8,5.5),sharex=True)
ax[0].plot(d.hhs_care,color=C); ax[0].set_ylabel('Children in HHS care'); ax[0].set_title('HHS care load, Jan 2023 – Dec 2025')
ax[1].plot(d.discharged.rolling(7).mean(),color=A,label='Discharged (7d avg)'); ax[1].plot(d.transferred.rolling(7).mean(),color=C,label='Transferred in (7d avg)')
ax[1].legend(frameon=False); ax[1].set_ylabel('Children / day'); plt.tight_layout(); plt.savefig('figures/fig1_overview.png'); plt.close()
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
ax[0].plot(d.cbp_custody.rolling(7).mean(),color=C,label='In CBP custody'); ax[0].plot(d.apprehended.rolling(7).mean(),color=A,label='Apprehended'); ax[0].legend(frameon=False); ax[0].set_title('CBP intake side (7d avg)')
ax[1].plot(d.net_pressure.rolling(14).mean(),color=C); ax[1].axhline(0,color='k',lw=.6); ax[1].set_title('Net pressure: transfers − discharges (14d avg)')
plt.tight_layout(); plt.savefig('figures/fig2_flows.png'); plt.close()
stl=STL(d.hhs_care,period=7,robust=True).fit()
fig,ax=plt.subplots(4,1,figsize=(8,6),sharex=True)
for a,s,t in zip(ax,[d.hhs_care,stl.trend,stl.seasonal,stl.resid],['Observed','Trend','Weekly seasonal','Residual']): a.plot(s,color=C,lw=.9); a.set_ylabel(t)
plt.tight_layout(); plt.savefig('figures/fig3_stl_care.png'); plt.close()
stl2=STL(d.discharged,period=7,robust=True).fit()
dow=d.discharged.groupby(d.index.dayofweek).mean(); dowo=o.discharged.groupby(o.index.dayofweek).mean().reindex(range(7)); print('report days by dow',o.index.dayofweek.value_counts().sort_index().to_dict())
fig,ax=plt.subplots(1,2,figsize=(8,3))
ax[0].bar(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],dowo.fillna(0).values,color=C); ax[0].set_title('Mean discharges by weekday (reported days)')
ax[1].plot(d.discharged.rolling(7).mean(),color=C); ax[1].set_title('Discharges, 7d avg'); plt.tight_layout(); plt.savefig('figures/fig4_dow.png'); plt.close()
# regimes
reg={'2023-01-12':'2023-12-31','2024-01-01':'2024-12-31','2025-01-01':'2025-12-21'}
tab=[]
for a,b in reg.items():
    s=d.loc[a:b]; tab.append(dict(period=f'{a[:4]}',mean_care=s.hhs_care.mean(),min_care=s.hhs_care.min(),max_care=s.hhs_care.max(),mean_discharge=s.discharged.mean(),mean_transfer=s.transferred.mean(),mean_intake=s.apprehended.mean()))
pd.DataFrame(tab).round(1).to_csv('results/regime_table.csv',index=False)
# stats
dd=d.hhs_care.diff().dropna()
st={'adf_level_p':adfuller(d.hhs_care)[1],'adf_diff_p':adfuller(dd)[1],
 'pct_interpolated':float(1-d.observed.mean()),'n_obs':int(d.observed.sum()),'n_days':len(d),
 'corr_care_dis':float(d[['hhs_care','discharged']].corr().iloc[0,1]),
 'corr_net_dcare':float(d[['net_pressure','care_change']].corr().iloc[0,1]),
 'peak':[str(d.hhs_care.idxmax().date()),float(d.hhs_care.max())],'trough':[str(d.hhs_care.idxmin().date()),float(d.hhs_care.min())],
 'weekly_seasonal_amp_care':float(stl.seasonal.abs().max()),'seasonal_strength_dis':float(max(0,1-stl2.resid.var()/(stl2.resid+stl2.seasonal).var())),
 'seasonal_strength_care':float(max(0,1-stl.resid.var()/(stl.resid+stl.seasonal).var()))}
# lag corr of net_pressure with future care change
lc={l:float(d.net_pressure.rolling(7).mean().corr(d.hhs_care.diff(7).shift(-l))) for l in (0,3,7)}
st['lagcorr_net7_vs_future_dcare7']=lc
# duplicates
raw=pd.read_csv('data_raw.csv').dropna(how='all'); st['dup_flow_rows']=int(raw.iloc[:,[1,2,3]].duplicated(keep=False).sum())
json.dump(st,open('results/eda_stats.json','w'),indent=1,default=float); print(json.dumps(st,indent=1,default=float)); print(pd.DataFrame(tab).round(1).to_string())
