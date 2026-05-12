import kagglehub
from torch import nn,no_grad,utils
from torchvision import transforms,models,datasets
from PIL import Image
from pathlib import Path
import torch 
import numpy as np
from collections import Counter

BASE_DIR=Path(__file__).parent.resolve()

# Loading the model

densenet = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')

# Replacing the classification layer with an identity layer that outputs a 1024 dim vector of features

densenet.classifier = nn.Identity()
densenet.eval()

device=torch.device("cuda")
densenet=densenet.to(device=device)

transformation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])

training_images=datasets.ImageFolder(root=f"{BASE_DIR}/kaggle_dataset/chest_xray/train",transform=transformation)
training_dataloader=utils.data.DataLoader(dataset=training_images,batch_size=32,pin_memory=True)

testing_images=datasets.ImageFolder(root=f"{BASE_DIR}/kaggle_dataset/chest_xray/test",transform=transformation)
testing_dataloader=utils.data.DataLoader(dataset=testing_images,batch_size=32,pin_memory=True)

# print(Counter(training_images.targets),Counter(testing_images.targets)) 
# print(len(training_images),len(testing_images))

# Counter({1: 3875, 0: 1341}) Counter({1: 390, 0: 234})
# 5216 624

train_features_list=[]
train_labels_list=[]

test_features_list=[]
test_labels_list=[]

with torch.no_grad():
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

print("Train : ",train_features.shape,train_labels.shape)
print("Test : ",test_features.shape,test_labels.shape)

# np.save(file=f"{BASE_DIR}/train_features",arr=train_features)
# np.save(file=f"{BASE_DIR}/train_labels",arr=train_labels)

# np.save(file=f"{BASE_DIR}/test_features",arr=test_features)
# np.save(file=f"{BASE_DIR}/test_labels",arr=test_labels)


