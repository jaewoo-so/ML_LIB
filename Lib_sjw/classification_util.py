from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve ,auc
import numpy as np

# metric function 을 여기에 정의한다. 
def aucpr(y , pred ):
    y_onehot = y.copy()
    pred_onehot = pred.copy()
    if len(y.shape) == 1:
        y_onehot = label_binarize(y , np.unique(y) )

    if len(pred.shape) == 1:
        pred_onehot = label_binarize(pred , np.unique(pred) )
    
    precision, recall, _ = precision_recall_curve(y_onehot.ravel(),pred_onehot.ravel()) # 평균까지 같이 내준다. 
    score_aucpr = auc(recall , precision) # recall is x axis , precision is y axis
    return score_aucpr





