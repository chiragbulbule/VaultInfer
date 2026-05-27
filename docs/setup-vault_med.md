## Step by Step Process to Run VaultMed

### To Install Required Dependencies

> [!NOTE]
> TenSEAL supports Python 3.8 through 3.13. Use Python 3.13 for the entire project to avoid dependancy mismatches.

```bash
cd vault_med/client_and_server
pip install -r requirements.txt
```

---

### To Generate Keys

1. Open a new terminal

```bash
cd vault_med/client_and_server/client
python context_setup.py
```

> [!NOTE]
> This generates `secret.tenseal` in the client folder and `public.tenseal` in the shared folder. Keep `secret.tenseal` private — it is the decryption key.

---

### To Start the Server

1. Open a new terminal

```bash
cd vault_med/client_and_server/server
uvicorn server:app --reload
```

> [!NOTE]
> By default the server will run on `http://127.0.0.1:8000`. Keep the server running.

---

### To Run the Client

1. Open a new terminal

```bash
cd vault_med/client_and_server/client
python client.py <path_to_image>
```

> [!NOTE]
> Accepted formats: PNG, JPG. The X-ray is never transmitted in plaintext — features are encrypted on the client before being sent to the server.

---

## To Reproduce the Model from Scratch

### To Download the Dataset

1. Download the Chest X-Ray Images dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
2. Place it at `vault_med/feature_extraction/image_dataset/` with the following structure:

```
image_dataset/
    train/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

---

### To Extract Features

1. Open a new terminal

```bash
cd vault_med/feature_extraction
python feature_extraction.py
```

> [!NOTE]
> This uses the committed `weights.pth.tar` CheXNet weights to extract 1024-dim features from each image. Outputs `train_features.npy`, `test_features.npy`, `train_labels.npy`, `test_labels.npy`, `vault_med_clipper.joblib`, and `vault_med_r_scaler.joblib` into `feature_extraction/data/`.

---

### To Train the Model

1. Open a new terminal

```bash
cd vault_med/model_training
python train.py
```

> [!NOTE]
> Outputs `vault_med_model`, `vault_weights.npy`, and `vault_bias.npy` into `model_training/`. Then follow the Quick Start steps above from key generation onwards.

> [!WARNING]
> Retraining will overwrite the existing committed weights. The committed weights achieve 91% accuracy, 0.96 Pneumonia recall, and 0.82 Normal recall on the Kaggle test set.
