from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_val_predict
from sentence_embedding import embedding,user_embed,test_cases_embed
from vault_dataset import train_labels
import numpy as np
import joblib as jl

#----------------------------------------------------------------------------Model Training-A-----------------------------------------------------------------#

# - Earlier used method

"""
x_train,x_test,y_train,y_test=train_test_split(embedding,train_labels,test_size=0.2,train_size=0.8,random_state=67,stratify=train_labels) 
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)

"""

# Currently used method

clf=LogisticRegression(max_iter=1000,class_weight="balanced",C=1)

#-----------------------------------------------------------------Cross_validation using StratifiedKFold--------------------------------------------------#

sk=StratifiedKFold(shuffle=True)

accuracy_cross_fold=cross_val_score(clf,embedding,train_labels,cv=sk)
print(accuracy_cross_fold.mean(),np.std(accuracy_cross_fold))

prediction_cross_fold=cross_val_predict(clf,embedding,train_labels,cv=sk)
prediction_probability_cross_fold=cross_val_predict(clf,embedding,train_labels,cv=sk,method='predict_proba')

# print(prediction_cross_fold)
# print(prediction_probability_cross_fold)

#----------------------------------------------------------------------------Model Training-B-----------------------------------------------------------------#

clf.fit(embedding,train_labels)

#----------------------------------------------------------------------------Model Predictions-----------------------------------------------------------------#

print(clf.score(embedding,train_labels))

# User input prediction

prediction=clf.predict(user_embed.reshape(1,384))
probability=clf.predict_proba(user_embed.reshape(1,384))

print(prediction)
print(probability)

# Test cases prediction

"""
prediction=clf.predict(test_cases_embed)
probability=clf.predict_proba(test_cases_embed)

print(prediction)
print(probability)

"""

#----------------------------------------------------------------SAVING MODEL,WEIGHTS AND BIAS TO RESPECTIVE FILES------------------------------------#

# weights=clf.coef_ #shape-(1,384)
# bias=clf.intercept_ #shape-(1,)

# flattened_weights=weights.flatten()

# np.save("./VaultInfer/Sentence_Classifier_model/New/vault_weights",flattened_weights)
# np.save("./VaultInfer/Sentence_Classifier_model/New/vault_bias",bias)
# jl.dump(clf,"./VaultInfer/Sentence_Classifier_model/New/vault_model",0)

#----------------------------------------------------------------LOADING MODEL,WEIGHTS AND BIAS FROM RESPECTIVE FILES (TEST)------------------------------------#

weights=np.load("./VaultInfer/Sentence_Classifier_model/New/vault_weights.npy")
bias=np.load("./VaultInfer/Sentence_Classifier_model/New/vault_bias.npy")

model=jl.load("./VaultInfer/Sentence_Classifier_model/New/vault_model")

print(weights[:4],weights.shape,bias)
prediction=model.predict(user_embed.reshape(1,384))
probability=model.predict_proba(user_embed.reshape(1,384))

print(prediction,probability)

#----------------------------------------------------------------MANUAL IMPLEMENTATION OF FORWARD PASS------------------------------------------------#

# Numpy version

"""
weighted_sum=np.dot(weights,test_embed) + bias

score=(0.5) + (weighted_sum/4) - (pow(weighted_sum,3)/48) + (pow(weighted_sum,5)/480) #-[0.81565414]-5th degree polynomial

"""

# Tenseal version

# To be completed

#--------------------------------------------------------------------------TEST CODE A---------------------------------------------------------------#

"""
predictions_all = clf.predict(embedding)
print(predictions_all)
print(type(predictions_all))


print(f"Predicted ALERT: {np.sum(predictions_all == 1)}")
print(f"Predicted NORMAL: {np.sum(predictions_all == 0)}")

test_sentences = [
    "Hydraulic pressure critical, shutdown imminent",
    "The meeting is at 3pm",
    "Unauthorized access detected at the server",
    "I'm making a cup of tea"
]
predictions = clf.predict(em)
probs=clf.predict_proba(em)
print(type(probs),probs)
print(type(predictions),predictions)

for sentence, prob in zip(test_sentences, probs):
    print(f"{prob[1]:.4f} — {sentence}")

prediction=clf.predict_proba(x_test)
for pre,actual in zip(prediction,y_test):
    print(pre,actual)

"""

#--------------------------------------------------------------------------TEST CODE B---------------------------------------------------------------#

"""
 
weighted_sum=np.dot(test_embed,weights) + bias #error because shapes don't meet requirements

print(np.dot(weights,test_embed)) # 1D Array

score_2=1/(1+np.exp(-weighted_sum)) #This uses sigmoid function-[0.81308656]
score_3=0.5+(0.15012*weighted_sum)-(0.001593*pow(weighted_sum,3)) #-[0.71564303] -3rd degree
score_4=0.5 + 0.197 * weighted_sum - 0.004 * weighted_sum **3 #-[ 0.777 ] -3rd degree


print(score,score_2,score_3,score_4)

all_weighted_sum=np.dot(weights,embedding.T) + bias
print(f"Min value is {all_weighted_sum.min():0.2f}")
print(f"Max value is {all_weighted_sum.max():0.2f}")
print(f"Mean value is {all_weighted_sum.mean():0.2f}")
    
"""

#--------------------------------------------------------------------------TEST CODE C---------------------------------------------------------------#

"""
Cross_validation using StratifiedKFold

index_list=[train_labels.index(true_label) for true_label,predict_label in zip(train_labels,prediction_cross_fold) if true_label != predict_label] - incorrect because it returns first appearance

index_list=[i for i,(actual_label,predict_label) in enumerate(zip(train_labels,prediction_cross_fold)) if actual_label!=predict_label]

wrongly_guessed_pairs=[(train_sentences[index],prediction_probability_cross_fold[index][1].item()) for index in index_list]
print(wrongly_guessed_pairs)

Testing which sentences are near the margin and which are way off

for i,predict_proba_tuple in enumerate(prediction_probability_cross_fold):
    if 0.4 <= predict_proba_tuple[1] <= 0.6:
        print(f"{train_sentences[i]} - {predict_proba_tuple[1]}")
    
    elif abs(predict_proba_tuple[1] - train_labels[i]) > 0.6:
        print(f"{train_sentences[i]} - {predict_proba_tuple[1]}")

"""


