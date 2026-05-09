from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict
import numpy as np
import joblib as jl
from pathlib import Path

BASE_DIR=Path(__file__).parent.resolve()

train_features=np.load(f"{BASE_DIR}/train_features.npy")
train_labels=np.load(f"{BASE_DIR}/train_labels.npy")

test_features=np.load(f"{BASE_DIR}/test_features.npy")
test_labels=np.load(f"{BASE_DIR}/test_labels.npy")

"""
This files contains the code for training and testing of the logistic regression model for the updated VaultMed System , along with its predictions for the given test data.

"""

clf=LogisticRegression(max_iter=1000,class_weight={0:2,1:1},C=1)

clf.fit(train_features,train_labels)

sk=StratifiedKFold(shuffle=True)

accuracy_cross_fold=cross_val_score(clf,test_features,test_labels,cv=sk)
print(f"Mean Accuracy of the model is : {accuracy_cross_fold.mean():0.2f}")
print(f"Standard deviation of the accuracy is : {np.std(accuracy_cross_fold):0.4f}")

# jl.dump(clf,f"{BASE_DIR}/vault_med_model",0)