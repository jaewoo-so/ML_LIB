import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV



X_use = None
y_use = None


# sklearn ANN parameter search 
mlp_grid={
    'learning_rate_init': [0.001 , 0.002 , 0.02 , 0.1 ,0.05],
    'activation': ['relu'],
    'solver': ['adam'],
    'max_iter' : [180  , 190 , 200 , 210 , 220 , 230 , 240 , ],
    'hidden_layer_sizes': [ (56,4), (58,6), (64,8) , (54,8) , (54,4) , (64,16,8)]}

estimator = MLPClassifier(random_state = 517)
parameter_grid = mlp_grid

model_gridsearch = GridSearchCV(estimator,
                                    parameter_grid,
                                    n_jobs=-1,
                                    cv = 10,
                                    verbose=True,
                                    scoring='roc_auc')

model_gridsearch.fit(X_use, y_use)
best_params = model_gridsearch.best_params_
model_best = model_gridsearch.best_estimator_

print(model_gridsearch.best_score_)
print(model_best)
print(best_params)