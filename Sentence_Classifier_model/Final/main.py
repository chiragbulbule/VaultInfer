import kagglehub
import torchvision.models as models
from torch import nn,no_grad

# path=kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
# print(path)

# Loading the model

densenet = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')

# Replacing the classification layer with an identity layer that outputs a 1024 dim vector of features

# densenet.classifier = nn.Identity()

densenet.eval()

with no_grad():
    result=densenet("Sentence_Classifier_model/Final/kaggle_dataset/test/NORMAL/IM-0001-0001.jpeg")

print(result)