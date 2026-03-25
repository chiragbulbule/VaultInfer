import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.network(x)

model = DigitClassifier()
model.load_state_dict(torch.load(r'D:\rvce\main el vs code folder\mnsit_practice\mnist_model.pth'))
model.eval()

def predict(image_path):
    # load and preprocess
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    # show what network sees ← moved INSIDE predict, after img is created
    plt.imshow(img, cmap='gray')
    plt.title("What the network sees")
    plt.show()

    # predict
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    with torch.no_grad():
        output    = model(tensor)
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted].item() * 100

    print(f"Predicted : {predicted}")
    print(f"Confidence: {confidence:.1f}%")

# save digit as PNG not JPG
predict(r'D:\rvce\main el vs code folder\mnsit_practice\digit.png')