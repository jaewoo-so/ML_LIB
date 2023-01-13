from sklearn.metrics import roc_curve, auc
import numpy as np

def Find_Optimal_Cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def to_soft_labling(df):
    xs = df.drop(['BMI_range','target'], axis=1).values
    ys = df['BMI_range']
    ys = np.where( ys.values > 0 , 0.9, 0.01)
    return xs, ys
    
def to_soft_labling_binary(df):
    xs = df.drop(['BMI_range','target'], axis=1).values
    ys = df['BMI_range']
    ys = np.where( ys.values > 0 , 0.9, 0.01)
    return xs, ys



###f1_score method for boost
from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.where(y_hat < 0.5, 0, 1)  
    return 'f1', f1_score(y_true, y_hat), True


def xgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_true =  np.where(y_true < 0.5, 0, 1) 
    y_hat = np.where(y_hat < 0.5, 0, 1) 
    return 'f1', f1_score(y_true, y_hat)