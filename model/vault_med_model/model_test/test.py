import numpy as np
import joblib as jl
from pathlib import Path
from torch import nn,no_grad,utils
from torchvision import transforms,models,datasets
from PIL import Image
from pathlib import Path
import torch 
from collections import Counter

BASE_DIR=Path(__file__).parent.resolve()
np.set_printoptions(suppress=True)
model=jl.load(f"{BASE_DIR}/vault_med_model")

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

image_1=Image.open(f"{BASE_DIR}/pneumonia.png").convert("RGB")

# print(transformation(img=image_1).shape) - 3,224,224

result_1 = densenet(transformation(img=image_1).to(device=device).unsqueeze(0)).numpy(force=True)

print(model.predict_proba(result_1))
print(model.predict(result_1))

print(f"Probability of Pneumonia is : {model.predict_proba(result_1)[0][1] * 100 : 0.2f} %")