import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#Boxplot
def box_plot(df_res , save_path):
    top = df_res.max().max() + 0.01
    bot = df_res.min().min() - 0.01
    
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