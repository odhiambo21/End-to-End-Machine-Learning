import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
import dill

from src.exception import CustomeException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomeException(e, sys)

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param,
                # n_iter=20,                # number of random combinations to try
                # scoring='accuracy',       # metric to optimize
                cv=3,                     # 5-fold cross-validation
                random_state=42,
                n_jobs=3,                # use all CPU cores
                verbose=1,
            )

            # model.fit(X_train,y_train)
            random_search.fit(X_train,y_train)

            model.set_params(**random_search.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        raise CustomeException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomeException(e, sys)