import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.metrics import accuracy_score , recall_score
from sklearn import preprocessing
from matplotlib import pyplot as plt

#Boxplot
def box_plot(df_res , save_path):
    top = df_res.max().max() + df_res.max().max()*0.05
    bot = df_res.min().min() - df_res.min().min()*0.05
    
    fig , ax = plt.subplots(1,1,figsize = (14,8))
    sns.boxplot( width = 0.3, data = df_res, ax = ax )
    ax.set_title('AUC of each K-Fold' , fontsize = 24)
    ax.set_xlabel('K-Fold' , fontsize = 18)
    ax.set_ylabel('Average AUC' , fontsize = 18)
    ax.set_ylim(bottom = bot , top = top)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()



def test_fold_performance(result_list , y_true , score_func):
    '''
    test_fold_index , predictions , test_fold_models = tr.training_Testfold('regression' , model_list[name] , param_list[name] , fitpm_list[name] ,  metric_func , X , y , 5 , 5 ) 
    result_list[name] = [test_fold_index , predictions, test_fold_models] # 모든 데이터에 대해 예측값이 oof에 저장되어 있다. 
    '''
    # 각 테스트 폴드 별로 각 모델의 CV스코어를 계산하는것 
    for k , v in result_list.items():
        print(' Test Fold {}'.format(k))
        model_res = []
        for i, pred in enumerate(v[1].T):
            score = score_func(y_true,pred)
            print('Val Fold : {:.4f}%'.format(score))
            model_res.append(score)
        print('Average : {:.4f}%'.format( np.array(model_res).mean()))
        print()
        # 현재 테스트 폴드에서
        
def roc_auc_multi_plot(y_test, y_score,n_classes , save_path = None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def roc_auc_binary_plot(y_test, y_score , title = None,save_path = None):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title is not None:
        plt.title(title)
    else:
        plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


# Todo : shap imp 에서 특정 피쳐만 빼고 플롯하기

def tree_summary_abs_plot( shap_values , X , exclude_list=None):
    '''
    explainer = shap.TreeExplainer(res_best_model)
    shap_values = explainer.shap_values(X)
    '''
    shap_abs = np.abs(shap_values)
    shap_abs_sum = shap_abs.sum(axis = 0)
    df_shap = pd.DataFrame(np.expand_dims(shap_abs_sum , 0) , columns = X.columns)
   

# ----------


def ImpFunc(model):
    t = model.coef_[0]
    feature_importance = abs(model.coef_[0])
    feature_importance = feature_importance / feature_importance.max()
    sorted_idx = np.argsort(feature_importance)
    imp =  feature_importance[sorted_idx]
    return imp

def ImpFunc_tree(model):
    feature_importance = model.feature_importances_
    feature_importance = feature_importance / feature_importance.max()
    sorted_idx = np.argsort(feature_importance)
    imp =  feature_importance[sorted_idx]
    return imp

def GetImportance(model, importance_func, xtrain, ytrain, xtest, ytest, xs_df):
    '''
    example
    implist , score = GetImportance(model,ImpFunc_tree,x_train,y_train,x_test,y_test,xs_df)
    PlotSaveImp(implist,'feature_importance_xg_v8.png','XGBoost : {:.2f}%'.format(score*100))
    '''
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    pred_round = pred.round()
    result = recall_score(ytest, pred_round )
    imp = importance_func(model)[2:]
    cols = xs_df.columns.values[2:]
    impdf = pd.DataFrame( {'colname' : cols , 'importance' : imp} )
    sortedimp_all = impdf.sort_values(["importance"], ascending=[False])
    sortedimp = sortedimp_all.iloc[:44,:]
    return sortedimp , result

def PlotSaveImp(df, path, title):
    '''
    example
    implist , score = GetImportance(model,ImpFunc_tree,x_train,y_train,x_test,y_test,xs_df)
    PlotSaveImp(implist,'feature_importance_xg_v8.png','XGBoost : {:.2f}%'.format(score*100))
    '''
    #plot 
    import seaborn as sns
    import matplotlib.pyplot as plt
    #sns.set(style="whitegrid")
    sns.set(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))
    sns.set_color_codes("pastel")
    print(df.shape)
    sns.barplot(x="importance" , y = 'colname', data=df, label="Importance", color="b")
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, df['importance'].max()*1.1), ylabel="Feature Name", xlabel="Feature Importance")
    ax.set_title(" Feature Importance : XGBoost")
    sns.despine(left=True, bottom=True)
    #plt.figure(facecolor='w')
    f.tight_layout()
    plt.savefig(path)