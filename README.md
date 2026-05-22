# VaultMed

VaultMed is a privacy-preserving encrypted medical diagnosis system built on Fully Homomorphic Encryption. Chest X-rays are processed entirely on the client — CheXNet extracts a 1024-dimensional feature vector locally, which is encrypted using CKKS before leaving the device. The diagnostic server computes classification directly on the ciphertext and returns an encrypted result. Decryption happens only on the client. The server sees nothing but ciphertext at every stage.

Built by second-semester students at RV College of Engineering, Bengaluru.

---

## How It Works

```
Chest X-ray → CheXNet (local) → 1024-dim features
→ clip + RobustScaler → CKKS encrypt
→ encrypted dot product + bias (server)
→ decrypt (client) → sigmoid → PNEUMONIA / NORMAL / INCONCLUSIVE
```

The server performs classification entirely on ciphertext using the CKKS homomorphic encryption scheme via TenSEAL. It never sees the raw features, the weighted sum, or the result.

---

## Performance

| Metric             | Value                                                     |
| ------------------ | --------------------------------------------------------- |
| Accuracy           | 91%                                                       |
| PNEUMONIA Recall   | 0.96                                                      |
| NORMAL Recall      | 0.82                                                      |
| Weighted Sum Range | −4.3 to +4.5                                              |
| Encrypted Payload  | ~326 KB                                                   |
| Encryption Scheme  | CKKS (poly_modulus_degree=8192)                           |
| Feature Extractor  | CheXNet (DenseNet-121, chest X-ray pretrained)            |
| Classifier         | Logistic Regression (C=0.000075, class_weight='balanced') |

---

## Key Technical Decisions

**Why Logistic Regression?**

```
FHE inference must reduce to a dot product — logistic regression satisfies this directly. Weights are extractable, inference is a single encrypted dot product, and CheXNet features are rich enough that a linear classifier achieves strong performance without a more complex model.
```

**Why client-side sigmoid?**

```
A 1024-dim encrypted dot product consumes most of the available CKKS noise budget. Applying a degree-5 Taylor polynomial sigmoid on top exhausts it entirely, producing garbage output. Instead, the server returns the encrypted weighted sum and the client applies sigmoid after decryption. The privacy guarantee is unchanged — the server never sees any plaintext value at any point.
```

**Why the inconclusive band?**

```
Forcing every prediction into a binary label is inappropriate for a medical screening tool. The model's score distribution shows natural separation — NORMAL cases cluster between 0.0 and 0.4, PNEUMONIA cases between 0.7 and 1.0. Cases in the overlap region are flagged as INCONCLUSIVE and referred to a radiologist rather than assigned a potentially wrong label.
```

---

## Research Prototype

VaultInfer is the NLP classifier that preceded VaultMed, validating the FHE inference approach on text before applying it to medical imaging. It classifies sentences as ALERT or NORMAL using the same CKKS encrypted dot product architecture.

---

## Setup

→ [VaultMed Setup Guide](docs/setup-vault_med.md)

→ [VaultInfer Setup Guide](docs/setup-vault_infer.md)

---

## Repository Structure

```
vaultmed/                               <-- Main Repository Root
├── docs/                               <-- Project Documentation
│   ├── Literature and Information
│   ├── Roles and Tasks
│   ├── setup-vault_infer.md            <-- VaultInfer setup guide
│   └── setup-vault_med.md              <-- VaultMed setup guide
│
├── vault_infer/                        <-- A privacy-preserving NLP security classifier (Project-1)
│   ├── client_and_server/
│   │   ├── client/
│   │   │   └── client.py
│   │   ├── server/
│   │   │   ├── server.py
│   │   │   └── vault_inference.py
│   │   └── requirements.txt
│   ├── dataset.py
│   ├── model.py
│   ├── sentence_embedding.py
│   ├── vault_bias.npy
│   ├── vault_model
│   └── vault_weights.npy
│
├── vault_med/                          <-- A privacy-preserving encrypted medical diagnostic system (Project-2)
│   ├── client_and_server/
│   │   ├── client/
│   │   │   ├── client.py
│   │   │   ├── context_setup.py
│   │   │   └── secret.tenseal
│   │   ├── logs/
│   │   │   └── audit.log
│   │   ├── server/
│   │   │   └── server.py
│   │   ├── shared/
│   │   │   └── public.tenseal
│   │   └── requirements.txt
│   │
│   ├── feature_extraction/
│   │   ├── data/
│   │   │   ├── test_features.npy
│   │   │   ├── test_labels.npy
│   │   │   ├── train_features.npy
│   │   │   ├── train_labels.npy
│   │   │   ├── vault_med_clipper.joblib
│   │   │   ├── vault_med_r_scaler.joblib
│   │   │   └── weights.pth.tar
│   │   ├── image_dataset/
│   │   └── feature_extraction.py
│   │
│   ├── model_test_and_inference/
│   │   ├── inconclusive.jpg
│   │   ├── normal.jpg
│   │   ├── pneumonia.jpg
│   │   └── test.py
│   │
│   └── model_training/
│       ├── FinalTenSEALModel.ipynb
│       ├── train.py
│       ├── vault_bias.npy
│       ├── vault_med_model
│       └── vault_weights.npy
|
├── .gitignore
├── README.md
├── requirements.txt

---

```

## Team

| Name           | Role          | Responsibilities                                                                          |
| -------------- | ------------- | ----------------------------------------------------------------------------------------- |
| Chirag Bulbule | Cryptographer | CKKS parameter tuning, TenSEAL integration, key management, server-client crypto protocol |
| Dhaksh S Kumar | AI Engineer   | Pipeline design, feature extraction, model training, FHE compatibility                    |
| Ayush S P      | Backend       | FastAPI server, encrypted inference endpoint, audit logging                               |
| Ram Nitish D   | UX/UI         | Frontend interface, privacy visualization, server perspective view                        |
| Dhyan M        | Research      | Accuracy evaluation, benchmarking, documentation, research literature                     |

RV College of Engineering, Bengaluru — Experiential Learning Project, Semester 2, 2025–26

---

## Future Work

- Slot-packed batch inference — CKKS with poly_modulus_degree=8192 provides 4096 slots; a 1024-dim feature vector occupies 1024, allowing up to 4 parallel encrypted classifications per ciphertext at near-identical latency
- Multi-label classification — CheXNet was pretrained on 14 chest pathologies; extending VaultMed to detect multiple conditions simultaneously within the same encrypted inference pipeline
- Formal noise analysis — quantify CKKS approximation error against plaintext inference across all 624 test samples, measuring mean error, maximum error, and whether noise ever shifts a prediction across the decision boundary
- Adaptive inconclusive thresholds — the current 0.4/0.7 band is calibrated on the Kaggle test distribution; a clinical deployment requires thresholds tuned to local patient population, condition prevalence, and acceptable false negative rate
