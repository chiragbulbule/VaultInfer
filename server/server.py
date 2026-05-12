from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import tenseal as ts
import joblib as jl

BASE_DIR     = Path(__file__).parent.resolve()                                  # VaultInfer/server/
VAULT_MED    = BASE_DIR.parent / "model" / "vault_med_model"                   # VaultInfer/model/vault_med_model/
MODEL_PATH   = VAULT_MED / "model_training" / "vault_med_model"                # joblib model
CONTEXT_PATH = BASE_DIR.parent / "shared" / "public.tenseal"                   # VaultInfer/shared/public.tenseal

app = FastAPI()

# ── Load model weights ─────────────────────────────────────────────────────
model   = jl.load(MODEL_PATH)
weights = model.coef_.flatten().tolist()    # 1024 floats
bias    = float(model.intercept_[0])

# ── Load public context (no secret key) ───────────────────────────────────
with open(CONTEXT_PATH, "rb") as f:
    public_context = ts.context_from(f.read())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    encrypted_bytes = await file.read()

    # Deserialize encrypted feature vector
    enc_features = ts.ckks_vector_from(public_context, encrypted_bytes)

    # Homomorphic dot product + bias
    enc_score = enc_features.dot(weights) + bias

    return {"encrypted_score": enc_score.serialize().hex()}