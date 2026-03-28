from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_embedding import embedding,train_labels,test_embed
import numpy as np

x_train,x_test,y_train,y_test=train_test_split(embedding,train_labels,test_size=0.2,train_size=0.8,random_state=42)
clf=LogisticRegression(max_iter=1000,class_weight={0:1,1:1.6})
clf.fit(x_train,y_train)

accuracy=clf.score(x_test,y_test)
# print(accuracy)

prediction=clf.predict(test_embed.reshape(1,384))
probability=clf.predict_proba(test_embed.reshape(1,384))

# print(prediction)
# print(probability)

#------------------------------------------------------------MANUAL IMPLEMENTATION OF FORWARD PASS---------------------------------------------#

weights=clf.coef_ #shape-(1,384)
bias=clf.intercept_ #shape-(1,)

weighted_sum=np.dot(weights,test_embed) + bias
score=(0.5) + (weighted_sum/4) - (pow(weighted_sum,3)/48) + (pow(weighted_sum,5)/480) #-[0.81565414]
score_3=0.5+(0.15012*weighted_sum)-(0.001593*pow(weighted_sum,3))
print(score_3)

score_2=1/(1+np.exp(-weighted_sum)) #This uses sigmoid function-[0.81308656]
print(score,score_2)

#----------------------------------------------------------------------TEST CODE---------------------------------------------------------------#

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