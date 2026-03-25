import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os

# ── 1. LOAD DATA ─────────────────────────────────────────────

print("Loading dataset...")

# better transforms — adds variety during training
train_data = torchvision.datasets.MNIST(
    root='./data', train=True, download=True,
    transform=transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
)
test_data = torchvision.datasets.MNIST(
    root='./data', train=False,
    download=True, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"Training images : {len(train_data)}")
print(f"Test images     : {len(test_data)}")


# ── 2. BUILD THE NETWORK ──────────────────────────────────────
#
#  Input:    784 numbers  (28x28 pixels flattened)
#  Hidden 1: 128 neurons  + ReLU
#  Hidden 2: 64 neurons   + ReLU
#  Output:   10 neurons   (one score per digit 0-9)

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


model     = DigitClassifier()
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}")


# ── 3. TRAINING LOOP ─────────────────────────────────────────

def train_one_epoch():
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        preds   = model(images)
        loss    = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (preds.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(train_loader), correct / total * 100


# ── 4. TEST LOOP ─────────────────────────────────────────────

def test():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds    = model(images)
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total * 100


# ── 5. TRAIN FOR 15 EPOCHS ────────────────────────────────────

print("\n" + "=" * 50)
print("  TRAINING")
print("=" * 50)
print(f"\n{'Epoch':>6}  {'Loss':>10}  {'Train Acc':>10}  {'Test Acc':>10}")
print("-" * 42)

for epoch in range(1, 16):      # 15 epochs instead of 5
    train_loss, train_acc = train_one_epoch()
    test_acc              = test()
    print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  {test_acc:>9.2f}%")

print("Saving to:", os.getcwd())
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved!")