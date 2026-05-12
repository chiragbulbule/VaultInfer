from pathlib import Path
from PIL import Image
from torchvision import models,transforms
from torch import device,nn,load
import numpy as np
import joblib as jl
import re

BASE_DIR=Path(__file__).parent.resolve()
BASE_PARENT_DIR=BASE_DIR.parent

np.set_printoptions(suppress=True,precision=4)
model=jl.load(f"{BASE_PARENT_DIR}/model_training/vault_med_model")

#--------------------------------------------------------------------------------------TESTING-Range of Weights ,Weighted Sum and Features--------------------------------------------------------------------------------------#

"""
weights=np.load(f"{BASE_PARENT_DIR}/model_training/vault_weights.npy")
bias=np.load(f"{BASE_PARENT_DIR}/model_training/vault_bias.npy")
features=np.load(f"{BASE_PARENT_DIR}/feature_extraction/data/train_features.npy")
weighted_sum=np.dot(features,weights) + bias[0]

print(f"Weights")
print(f"Mean of weights : {np.abs(weights).mean() :0.4f}")
print(f"Max of weights : {weights.max() :0.4f}")
print(f"Min of weights : {weights.min() :0.4f}\n")


print(f"Features")
print(f"Shape of features : {features.shape}")
print(f"Std of features : {features.std() : 0.4f}")
print(f"Mean of features : {features.mean() : 0.4f}")
print(f"Max of features : {features.max() : 0.4f}")
print(f"Min of features : {features.min() : 0.4f}\n")


print(f"Weighted Sum")
print(f"Max of weighted sum : {weighted_sum.max() : 0.4f}")
print(f"Min of weighted sum : {weighted_sum.min(): 0.4f}\n")

"""

# Loading the model

densenet = models.densenet121()

weights_dict=load(f"{BASE_PARENT_DIR}/feature_extraction/data/weights.pth.tar")
scaler=jl.load(f"{BASE_PARENT_DIR}/feature_extraction/data/vault_med_r_scaler.joblib")
clipper=jl.load(f"{BASE_PARENT_DIR}/feature_extraction/data/vault_med_clipper.joblib")

# print(weights_dict.keys())
# print(weights_dict["state_dict"].keys())

weights_dict_updated={}

for key,value in weights_dict["state_dict"].items():
    weights_dict_updated[re.sub(r'([a-z]+)\.(\d+)', r'\1\2', key.replace("module.densenet121.",""))] = value

densenet.load_state_dict(weights_dict_updated,strict=False)

# Replacing the classification layer with an identity layer that outputs a 1024 dim vector of features

densenet.classifier = nn.Identity()
densenet.eval()

device=device("cuda")
densenet=densenet.to(device=device)

transformation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

image_1=Image.open(f"{BASE_DIR}/pneumonia.png").convert("RGB")
# print(transformation(img=image_1).shape) - 3,224,224

result_1 = densenet(transformation(img=image_1).to(device=device).unsqueeze(0)).numpy(force=True)
# print(result_1.shape)

result_1=np.clip(result_1,0,clipper)
result_1=scaler.transform(result_1)

print(model.predict_proba(result_1))
print(model.predict(result_1),"\n")

print(f"Probability of Pneumonia is : {model.predict_proba(result_1)[0][1] * 100 : 0.2f} %\n")

#--------------------------------------------------------------------------------------TESTING-Confusion Matrix--------------------------------------------------------------------------------------#

"""
from sklearn.metrics import confusion_matrix,classification_report

test_features=np.load(f"{BASE_PARENT_DIR}/feature_extraction/data/test_features.npy")
test_labels=np.load(f"{BASE_PARENT_DIR}/feature_extraction/data/test_labels.npy")

prediction=model.predict(test_features)
print(classification_report(test_labels,prediction,target_names=["NORMAL","PNEUMONIA"]))

"""


