from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,ConfusionMatrixDisplay,RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.costs import optimal_threshold

ROOT=Path(__file__).resolve().parent; DATA=ROOT/'data'/'application_train.csv'; SEED=42
def engineer(d):
    d=d.copy(); d.loc[d.DAYS_EMPLOYED==365243,'DAYS_EMPLOYED']=np.nan; d['CREDIT_INCOME_RATIO']=d.AMT_CREDIT/d.AMT_INCOME_TOTAL.replace(0,np.nan); d['ANNUITY_INCOME_RATIO']=d.AMT_ANNUITY/d.AMT_INCOME_TOTAL.replace(0,np.nan); d['EMPLOYED_AGE_RATIO']=d.DAYS_EMPLOYED.abs()/d.DAYS_BIRTH.abs(); return d
def metrics(y,p,t):
    pred=p>=t; return {'roc_auc':roc_auc_score(y,p),'average_precision':average_precision_score(y,p),'f1':f1_score(y,pred),'threshold':t,'false_negatives':int(((y==1)&~pred).sum()),'false_positives':int(((y==0)&pred).sum())}
def main():
    out=ROOT/'outputs'; (out/'figures').mkdir(parents=True,exist_ok=True); (out/'metrics').mkdir(parents=True,exist_ok=True)
    d=engineer(pd.read_csv(DATA)); y=d.pop('TARGET').astype(int); X=d.drop(columns=['SK_ID_CURR'])
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.4,stratify=y,random_state=SEED); Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.5,stratify=ytmp,random_state=SEED)
    cats=X.select_dtypes('object').columns.tolist(); nums=X.columns.difference(cats).tolist()
    prep=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',min_frequency=100))]),cats)])
    logistic=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=SEED))]).fit(Xtr,ytr)
    def cat_ready(z):
        z=z.copy()
        for c in cats:z[c]=z[c].fillna('Missing').astype(str)
        return z
    cb=CatBoostClassifier(iterations=450,depth=7,learning_rate=.06,loss_function='Logloss',eval_metric='AUC',auto_class_weights='Balanced',random_seed=SEED,verbose=False,thread_count=-1)
    cb.fit(cat_ready(Xtr),ytr,cat_features=cats,eval_set=(cat_ready(Xv),yv),early_stopping_rounds=60,verbose=False)
    models={'Logistic Regression':(logistic,Xv,Xte),'CatBoost':(cb,cat_ready(Xv),cat_ready(Xte))}; results={}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for name,(m,xv,xte) in models.items():
        pv=m.predict_proba(xv)[:,1]; pt=m.predict_proba(xte)[:,1]; threshold,table=optimal_threshold(yv,pv,10,1); table.to_csv(out/'metrics'/f'{name.lower().replace(" ","_")}_threshold_costs.csv',index=False); results[name]={'validation':metrics(yv,pv,threshold),'test':metrics(yte,pt,threshold)}; RocCurveDisplay.from_predictions(yte,pt,name=name,ax=axes[0])
    best=max(results,key=lambda n:results[n]['validation']['roc_auc']); m,xv,xte=models[best]; t=results[best]['validation']['threshold']; ConfusionMatrixDisplay.from_predictions(yte,m.predict_proba(xte)[:,1]>=t,cmap='Blues',ax=axes[1]); axes[1].set_title(f'{best} at cost-optimal threshold {t:.2f}'); fig.tight_layout(); fig.savefig(out/'figures'/'model_evaluation.png',dpi=160); plt.close(fig)
    imp=pd.Series(cb.get_feature_importance(),index=X.columns).nlargest(20); imp.to_csv(out/'metrics'/'catboost_feature_importance.csv'); fig,ax=plt.subplots(figsize=(9,7)); imp.sort_values().plot.barh(ax=ax,color='#2563eb'); ax.set_title('CatBoost feature importance'); fig.tight_layout(); fig.savefig(out/'figures'/'feature_importance.png',dpi=160); plt.close(fig)
    payload={'rows':len(d),'target_rate':float(y.mean()),'costs':{'false_negative':10,'false_positive':1},'selected_model':best,'results':results}; (out/'metrics'/'metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2)); return payload
if __name__=='__main__':main()
