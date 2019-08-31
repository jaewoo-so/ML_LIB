from sklearn.metrics import roc_curve, auc

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
    return xs , ys