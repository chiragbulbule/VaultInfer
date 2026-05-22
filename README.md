# VaultMed

VaultMed is a privacy-preserving encrypted medical diagnosis system built on Fully Homomorphic Encryption. A chest X-ray is never transmitted in plaintext — CheXNet extracts features locally, those features are encrypted on the client, classified on a diagnostic server without decryption, and the encrypted result is returned and decrypted only on the client.

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

```
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

```

---

## Key Technical Decisions

```
**Why Logistic Regression?**
FHE requires a linear classifier — weights must be extractable and the inference must reduce to a dot product. Logistic regression satisfies both while achieving strong performance on CheXNet features.

**Why client-side sigmoid?**
Applying the degree-5 Taylor polynomial sigmoid on encrypted vectors after a 1024-dim dot product exhausts the CKKS noise budget regardless of parameter choice. The server returns the encrypted weighted sum; the client decrypts and applies sigmoid in plaintext. The privacy guarantee is identical — the server never sees any plaintext value.

**Why the inconclusive band?**
The model's probability distribution shows clear separation: NORMAL cases cluster between 0.0–0.4, PNEUMONIA cases between 0.7–1.0, with genuine overlap in between. Cases in the 0.4–0.7 range are flagged as INCONCLUSIVE and referred to a radiologist rather than forced into a binary label.

```

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

```
| Name           | Role          | Responsibilities                                                                          |
| -------------- | ------------- | ----------------------------------------------------------------------------------------- |
| Chirag Bulbule | Cryptographer | CKKS parameter tuning, TenSEAL integration, key management, server-client crypto protocol |
| Dhaksh S Kumar | AI Engineer   | Pipeline design, feature extraction, model training, FHE compatibility                    |
| Ayush S P      | Backend       | FastAPI server, encrypted inference endpoint, audit logging                               |
| Ram Nitish D   | UX/UI         | Frontend interface, privacy visualization, server perspective view                        |
| Dhyan M        | Research      | Accuracy evaluation, benchmarking, documentation, research literature                     |

RV College of Engineering, Bengaluru — Experiential Learning Project, Semester 2, 2025–26

```

---

## Future Work

```
- Slot-packed batch inference — CKKS with poly_modulus_degree=8192 provides 4096 slots; a 1024-dim vector uses 1024, leaving room for 4 parallel encrypted classifications per ciphertext while maintaining around the same processing time.
- Extension to other CheXNet pathologies — the feature extractor supports 14 chest conditions beyond pneumonia.
- Formal noise analysis — quantify CKKS approximation error against plaintext dot product across all 624 test samples, measuring mean error, maximum error, and whether noise ever crosses the decision boundary to change a prediction.
- Adaptive inconclusive thresholds — current 0.4/0.7 thresholds are calibrated on the Kaggle test distribution; a clinical deployment would require adaptive thresholds based on local patient population, pneumonia prevalence, and acceptable false negative rate.

```
