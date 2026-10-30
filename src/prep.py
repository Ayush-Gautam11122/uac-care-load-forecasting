import pandas as pd, numpy as np
def load_clean(path='data_raw.csv'):
    d=pd.read_csv(path).dropna(how='all')
    d.columns=['date','apprehended','cbp_custody','transferred','hhs_care','discharged']
    d['date']=pd.to_datetime(d['date'])
    for c in d.columns[1:]: d[c]=pd.to_numeric(d[c].astype(str).str.replace(',',''))
    d=d.sort_values('date').drop_duplicates('date').set_index('date')
    full=pd.date_range(d.index.min(),d.index.max(),freq='D')
    d['observed']=1
    d=d.reindex(full); d.index.name='date'
    d['observed']=d['observed'].fillna(0).astype(int)
    for c in ['apprehended','cbp_custody','transferred','hhs_care','discharged']:
        d[c]=d[c].interpolate('linear').round(1)
    d['net_pressure']=d['transferred']-d['discharged']
    d['care_change']=d['hhs_care'].diff()
    return d
if __name__=='__main__':
    d=load_clean(); d.to_csv('results/data_clean.csv')
    print(d.shape, d.observed.sum(), (1-d.observed.mean()))
    o=d[d.observed==1]
    # reconciliation: does 1-day change in stock ~ transferred - discharged?
    ch=o.hhs_care.diff(); gap=o.index.to_series().diff().dt.days
    m=(gap==1)
    x=(o.transferred-o.discharged)[m]; y=ch[m]
    print('corr net vs change (1-day gaps):',np.corrcoef(x[1:],y[1:])[0,1] if len(x)>2 else None, 'n',m.sum())
    print(pd.DataFrame({'x':x,'y':y}).dropna().corr())
