from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torch import device, nn, load
import tenseal as ts
import numpy as np
import joblib as jl
import requests
import re
import math
import sys

BASE_DIR        = Path(__file__).parent.resolve()
VAULT_MED       = BASE_DIR.parent.parent
MODEL_DATA      = VAULT_MED / "feature_extraction" / "data"
SECRET_PATH     = BASE_DIR / "secret.tenseal"

# ── Load secret context ────────────────────────────────────────────────────
with open(SECRET_PATH, "rb") as f:
    context = ts.context_from(f.read())

# ── Load DenseNet ──────────────────────────────────────────────────────────
print("Loading DenseNet model...")
densenet = models.densenet121()
weights_dict = load(MODEL_DATA / "weights.pth.tar", map_location="cpu")

weights_dict_updated = {}
for key, value in weights_dict["state_dict"].items():
    weights_dict_updated[re.sub(r'([a-z]+)\.(\d+)', r'\1\2',
        key.replace("module.densenet121.", ""))] = value

densenet.load_state_dict(weights_dict_updated, strict=False)
densenet.classifier = nn.Identity()
densenet.eval()

dev = device("cpu")
densenet = densenet.to(device=dev)

scaler  = jl.load(MODEL_DATA / "vault_med_r_scaler.joblib")
clipper = jl.load(MODEL_DATA / "vault_med_clipper.joblib")

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("DenseNet loaded.\n")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(image_path: str, server_url: str = "http://localhost:8000"):
    # 1. Extract features
    print("Extracting features from image...")
    img = Image.open(image_path).convert("RGB")
    with __import__("torch").no_grad():
        features = densenet(transformation(img).to(device=dev).unsqueeze(0)).numpy(force=True)

    # 2. Preprocess
    features = np.clip(features, 0, clipper)
    features = scaler.transform(features)
    print("Preprocessing done.\n")

    # 3. Encrypt
    print("Encrypting feature vector...")
    enc_features = ts.ckks_vector(context, features.flatten().tolist())
    encrypted_bytes = enc_features.serialize()
    print(f"Encryption done. Payload size : {len(encrypted_bytes) / 1024:.2f} KB\n")

    # 4. Send to server
    print("Sending encrypted data to server...")
    response = requests.post(
        f"{server_url}/predict",
        files={"file": ("features.bin", encrypted_bytes, "application/octet-stream")}
    )
    response.raise_for_status()
    print("Received encrypted score from server.")

    # 5. Decrypt score
    print("Decrypting score...\n")
    enc_score_bytes = bytes.fromhex(response.json()["encrypted_score"])
    enc_score = ts.ckks_vector_from(context, enc_score_bytes)
    score = enc_score.decrypt()[0]

    # 6. Sigmoid → probability
    prob = sigmoid(score)

    if prob >= 0.7:
        print("Result:         PNEUMONIA DETECTED")
        print("Confidence:     High")

    elif prob <= 0.45:
        print("Result:         NORMAL")
        print("Confidence:     High")

    else:
        print("Result:         INCONCLUSIVE")
        print("Recommendation: Refer to radiologist for review")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_image>")
        sys.exit(1)
    predict(sys.argv[1])