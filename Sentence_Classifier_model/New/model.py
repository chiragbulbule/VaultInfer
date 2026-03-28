from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_embedding import embedding,train_labels,em
import numpy as np

x_train,x_test,y_train,y_test=train_test_split(embedding,train_labels,test_size=0.2,train_size=0.8,random_state=42)
clf=LogisticRegression(max_iter=1000,class_weight={0:1,1:1.6})
clf.fit(x_train,y_train)

accuracy=clf.score(x_test,y_test)
print(accuracy)

# predictions_all = clf.predict(embedding)
# print(predictions_all)
# print(type(predictions_all))
# print(f"Predicted ALERT: {np.sum(predictions_all == 1)}")
# print(f"Predicted NORMAL: {np.sum(predictions_all == 0)}")

# test_sentences = [
#     "Hydraulic pressure critical, shutdown imminent",
#     "The meeting is at 3pm",
#     "Unauthorized access detected at the server",
#     "I'm making a cup of tea"
# ]
# predictions = clf.predict(em)
# probs=clf.predict_proba(em)
# print(type(probs),probs)
# print(type(predictions),predictions)

# for sentence, prob in zip(test_sentences, probs):
#     print(f"{prob[1]:.4f} — {sentence}")

prediction=clf.predict_proba(x_test)
for pre,actual in zip(prediction,y_test):
    print(pre,actual)