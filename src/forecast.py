import pandas as pd, numpy as np, warnings, sys
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas.tseries.holiday import USFederalHolidayCalendar
warnings.filterwarnings('ignore')
H=14
HZ=list(range(1,H+1))

def features(d):
    f=pd.DataFrame(index=d.index)
    c=d.hhs_care; x=d.discharged
    f['care']=c
    for l in (1,7,14): f[f'care_d{l}']=c-c.shift(l)
    for l in (1,7,14): f[f'dis_l{l}']=x.shift(l)
    for w in (7,14):
        f[f'dis_m{w}']=x.rolling(w).mean(); f[f'dis_v{w}']=x.rolling(w).var()
        f[f'care_chg_m{w}']=c.diff().rolling(w).mean(); f[f'care_chg_v{w}']=c.diff().rolling(w).var()
        f[f'net_m{w}']=d.net_pressure.rolling(w).mean()
    f['net_l0']=d.net_pressure; f['trans_m7']=d.transferred.rolling(7).mean()
    f['intake_m7']=d.apprehended.rolling(7).mean(); f['cbp_m7']=d.cbp_custody.rolling(7).mean()
    f['dow']=d.index.dayofweek; f['month']=d.index.month
    hol=USFederalHolidayCalendar().holidays(d.index.min(),d.index.max()+pd.Timedelta(days=30))
    f['holiday']=d.index.isin(hol).astype(int)
    return f

MODELS=['Naive','MovingAvg7','ARIMA','SARIMA','ETS','RandomForest','GradientBoosting']

def ml(name):
    if name=='RandomForest': return RandomForestRegressor(n_estimators=120,min_samples_leaf=3,max_features=0.6,n_jobs=-1,random_state=0)
    return GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0)

def stat_fit(name,y):
    if name=='ARIMA': return SARIMAX(y,order=(1,1,1),trend='n').fit(disp=False)
    if name=='SARIMA': return SARIMAX(y,order=(1,1,1),seasonal_order=(1,0,1,7),trend='n').fit(disp=False,maxiter=60)

def stat_fc(name,fit,y_hist,n=H):
    if name in('ARIMA','SARIMA'):
        return np.asarray(fit.apply(y_hist,refit=False).forecast(n))   # fixed params, new data
    if name=='ETS':
        m=ExponentialSmoothing(y_hist.iloc[-120:],trend='add',damped_trend=True,seasonal=None).fit()
        return np.asarray(m.forecast(n))

def walk_forward(d,target,test_start,test_end,refit=21):
    """Expanding-window walk-forward. Rows: one per (origin, h, model)."""
    f=features(d); y=d[target]; rows=[]
    origins=d.index[(d.index>=pd.Timestamp(test_start)-pd.Timedelta(days=1))&(d.index<pd.Timestamp(test_end))]
    obs=d.observed
    is_level=(target=='hhs_care')
    fitted={}; last_refit=None
    for k,t in enumerate(origins):
        if t+pd.Timedelta(days=1)>d.index[-1]: break
        if last_refit is None or (t-last_refit).days>=refit:
            last_refit=t; hist=y.loc[:t]
            for nm in ('ARIMA','SARIMA'): fitted[nm]=stat_fit(nm,hist)
            for nm in ('RandomForest','GradientBoosting'):
                fitted[nm]={}
                for h in HZ:
                    # training pairs whose target date <= t  (no leakage)
                    idx=d.index[(d.index<=t-pd.Timedelta(days=h))]
                    X=f.loc[idx].dropna(); 
                    yt=(y.shift(-h)-(y if is_level else 0)).loc[X.index]
                    m=ml(nm).fit(X.values,yt.values); fitted[nm][h]=m
        hist=y.loc[:t]; base=y.loc[t]
        preds={'Naive':np.repeat(base,H),'MovingAvg7':np.repeat(hist.iloc[-7:].mean(),H)}
        preds['ARIMA']=stat_fc('ARIMA',fitted['ARIMA'],hist); preds['SARIMA']=stat_fc('SARIMA',fitted['SARIMA'],hist)
        preds['ETS']=stat_fc('ETS',None,hist)
        xr=f.loc[[t]].values
        for nm in ('RandomForest','GradientBoosting'):
            preds[nm]=np.array([fitted[nm][h].predict(xr)[0]+(base if is_level else 0) for h in HZ])
        for nm,p in preds.items():
            for h in HZ:
                td=t+pd.Timedelta(days=h)
                if td>d.index[-1]: continue
                rows.append((t,td,h,nm,y.loc[td],p[h-1],obs.loc[td],base))
    return pd.DataFrame(rows,columns=['origin','target_date','h','model','y_true','y_pred','observed','y_origin'])

def final_forecast(d,target,H=H):
    f=features(d); y=d[target]; t=d.index[-1]; is_level=(target=='hhs_care'); base=y.iloc[-1]; out={}
    out['Naive']=np.repeat(base,H); out['MovingAvg7']=np.repeat(y.iloc[-7:].mean(),H)
    for nm in('ARIMA','SARIMA'): out[nm]=stat_fc(nm,stat_fit(nm,y),y)
    out['ETS']=stat_fc('ETS',None,y)
    xr=f.loc[[t]].values
    for nm in('RandomForest','GradientBoosting'):
        p=[]
        for h in HZ:
            X=f.loc[d.index[d.index<=t-pd.Timedelta(days=h)]].dropna(); yt=(y.shift(-h)-(y if is_level else 0)).loc[X.index]
            p.append(ml(nm).fit(X.values,yt.values).predict(xr)[0]+(base if is_level else 0))
        out[nm]=np.array(p)
    rows=[(t+pd.Timedelta(days=h),h,nm,float(v[h-1])) for nm,v in out.items() for h in HZ]
    return pd.DataFrame(rows,columns=['date','h','model','forecast'])

if __name__=='__main__':
    d=pd.read_csv('results/data_clean.csv',index_col=0,parse_dates=True)
    for target in ('hhs_care','discharged'):
        for tag,(a,b) in {'main':('2025-04-01','2025-12-21'),'stress':('2024-12-15','2025-03-31')}.items():
            r=walk_forward(d,target,a,b); r.to_csv(f'results/bt_{target}_{tag}.csv',index=False); print(target,tag,len(r),flush=True)
        final_forecast(d,target).to_csv(f'results/final_{target}.csv',index=False)
    print('done')
