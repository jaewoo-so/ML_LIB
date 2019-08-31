import Lib_CDSS.training as tr
import Lib_CDSS.model_interface as mi
import Lib_CDSS.model_parmas as mp


from sklearn.datasets import load_boston , load_iris , load_breast_cancer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


'''
1. 데이터 만들기
2. 모델 제너레이터 만들기 
3. 모델 생성 파라미터, 피팅 파라미터 만들기 
4. 성능평가 메트릭 정의하기 :  f : y , pred -> value
5. training_fixedTest 실행하기 
6. 유틸에 테스트 데이터에 대한 성능평가, 또는 저장 등을 하기 
'''

def Test_Regression():
    data = load_boston()
    X = data.data
    y = data.target
    
    xtrain , xtest , ytrain , ytest = train_test_split(X , y , test_size = 0.2 )
    xgb = mi.myXGBRegressor()
    p_xgb = mp.param_xg('regression' , use_gpu= False)
    fit_xgb = {}
    # metric func
    metric_func = mean_squared_error

    # xgboost complete
    fold_predict , fold_oof , fold_metric = tr.training_fixedTest('regression' , xgb , p_xgb , fit_xgb ,  metric_func , xtrain , ytrain , xtest , 5  ) 
    print('done')

Test_Regression()