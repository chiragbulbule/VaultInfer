from torchvision import transforms,models
from PIL import Image
from torch import nn



image=Image.open("VaultInfer/Sentence_Classifier_model/Final/image.png")
transformation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
# print(transformation(img=image))
model=models.densenet121(weights="DEFAULT")

# model.classifier=nn.Identity()
model.eval()
model.zero_grad()

result = model(transformation(img=image).unsqueeze(0))
print(result.shape)
print(result)
