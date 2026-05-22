from pathlib import Path
from torch import load,nn,device,no_grad
from torchvision import transforms,models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import numpy as np
import joblib as jb
import re

BASE_DIR=Path(__file__).parent.resolve()
np.set_printoptions(suppress=True,precision=4)

# Loading the model with random weights

densenet = models.densenet121()
weights_dict=load(f"{BASE_DIR}/data/weights.pth.tar")

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

training_images=ImageFolder(root=f"{BASE_DIR}/image_dataset/train",transform=transformation)
training_dataloader=DataLoader(dataset=training_images,batch_size=32,pin_memory=True)

testing_images=ImageFolder(root=f"{BASE_DIR}/image_dataset/test",transform=transformation)
testing_dataloader=DataLoader(dataset=testing_images,batch_size=32,pin_memory=True)

"""
from collections import Counter
print(Counter(training_images.targets),Counter(testing_images.targets)) 
print(len(training_images),len(testing_images))

# Counter({1: 3875, 0: 1341}) Counter({1: 390, 0: 234})
# 5216 624

"""

train_features_list=[]
train_labels_list=[]

test_features_list=[]
test_labels_list=[]

with no_grad():
    
    for batch in training_dataloader:
        # print(densenet(batch[0].to(device=device)).shape) - 32,1024
        # print((batch[0].to(device=device)).shape) - 32,3,224,224
        train_features_list.append(densenet(batch[0].to(device=device)).numpy(force=True))
        train_labels_list.append(batch[1])

    for batch in testing_dataloader:
        test_features_list.append(densenet(batch[0].to(device=device)).numpy(force=True))
        test_labels_list.append(batch[1])
        
print("Train : ", len(train_features_list),len(train_labels_list))
print("Test : ", len(test_features_list),len(test_labels_list))

train_features=np.concatenate(train_features_list)
train_labels=np.concatenate(train_labels_list)

test_features=np.concatenate(test_features_list)
test_labels=np.concatenate(test_labels_list)

# TESTING with PCA,RBF,Scaler

"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler

pca=PCA(n_components=256)
pca.fit(train_features)

print(pca.explained_variance_ratio_.sum())
train_features=pca.transform(train_features)
test_features=pca.transform(test_features)

rbf=RBFSampler(gamma=0.001,n_components=2048,random_state=42)

train_features=rbf.fit_transform(train_features)
test_features=rbf.transform(test_features)

print(train_features.mean(),train_features.std())

s_scaler = StandardScaler()
s_scaler.fit(train_features)

train_features=s_scaler.transform(train_features)
test_features=s_scaler.transform(test_features)

# jb.dump(pca,f"{BASE_DIR}/data/vault_med_pca.joblib")
# jb.dump(rbf,f"{BASE_DIR}/data/vault_med_rbf.joblib")
# jb.dump(s_scaler,f"{BASE_DIR}/data/vault_med_s_scaler.joblib")

"""

p95=np.percentile(train_features,95,axis=0)
train_features = np.clip(train_features,0,p95)
test_features=np.clip(test_features,0,p95)

r_scaler = RobustScaler()  # uses median and IQR, ignores tail outliers
r_scaler.fit(train_features)

train_features=r_scaler.transform(train_features)
test_features=r_scaler.transform(test_features)

jb.dump(p95,f"{BASE_DIR}/data/vault_med_clipper.joblib")
jb.dump(r_scaler,f"{BASE_DIR}/data/vault_med_r_scaler.joblib")

print("Train-Shape : ",train_features.shape,train_labels.shape)
print("Test-Shape : ",test_features.shape,test_labels.shape)

np.save(file=f"{BASE_DIR}/data/train_features",arr=train_features)
np.save(file=f"{BASE_DIR}/data/train_labels",arr=train_labels)

np.save(file=f"{BASE_DIR}/data/test_features",arr=test_features)
np.save(file=f"{BASE_DIR}/data/test_labels",arr=test_labels)


