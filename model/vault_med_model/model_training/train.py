from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict,GridSearchCV
import numpy as np
import joblib as jl
import pandas as pd

BASE_DIR=Path(__file__).parent.resolve()
BASE_PARENT_DIR=BASE_DIR.parent

np.set_printoptions(suppress=True,precision=4)

train_features=np.load(f"{BASE_PARENT_DIR}/feature_extraction/data/train_features.npy")
train_labels=np.load(f"{BASE_PARENT_DIR}/feature_extraction/data/train_labels.npy")

"""
This files contains the code for training of the logistic regression model for the updated VaultMed System

"""

sk=StratifiedKFold(shuffle=True)

# TESTING - Grid Search Cross Validation

"""
grid={
    "C":[0.001,0.0001,0.002,0.025,0.0015,0.01],
    "class_weight":["balanced",{0:1.75,1:1},{0:2,1:1},{0:1,1:1},{0:1.25,1:1},{0:1.5,1:1},{0:1.85,1:1}],
}

clf_test=LogisticRegression(max_iter=1000)

grid_search=GridSearchCV(clf_test,grid,cv=sk,scoring="balanced_accuracy")
grid_search.fit(train_features,train_labels)

data = pd.DataFrame(grid_search.cv_results_)
data.sort_values(by="rank_test_score",inplace=True)
data.to_csv(f"{BASE_DIR}/test.csv")

print(grid_search.best_params_) # eg- {'C': 0.05, 'class_weight': {0: 1.75, 1: 1}}
print(grid_search.best_index_) # eg- 37
print(grid_search.best_score_) #eg- 0.9873644608129093

clf = grid_search.best_estimator_

"""

clf=LogisticRegression(max_iter=1000,C=0.000075,class_weight="balanced")
clf.fit(train_features,train_labels)

accuracy_cross_fold=cross_val_score(clf,train_features,train_labels,cv=sk)
print(f"Mean Accuracy of the model is : {accuracy_cross_fold.mean() : 0.2f}")
print(f"Standard deviation of the accuracy is : {np.std(accuracy_cross_fold) : 0.4f}")

weights=clf.coef_ #shape-(1,1024)
bias=clf.intercept_ #shape-(1,)

print(f"Mean of weights : {np.abs(weights).mean() : 0.4f}")
print(f"Max of weights : {weights.max() : 0.4f}")
print(f"Min of weights : {weights.min() : 0.4f}")

flattened_weights=weights.flatten() #This is to make it compatible

# np.save(f"{BASE_DIR}/vault_weights",flattened_weights)
# np.save(f"{BASE_DIR}/vault_bias",bias)
# jl.dump(clf,f"{BASE_DIR}/vault_med_model",0)