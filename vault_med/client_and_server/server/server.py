from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import tenseal as ts
import joblib as jl
import hashlib
from datetime import datetime
import numpy as np
import json

BASE_DIR     = Path(__file__).parent.resolve()                                 
VAULT_MED_CORE    = BASE_DIR.parent.parent                                          
MODEL_DATA   = VAULT_MED_CORE   / "model_training"              
CONTEXT_PATH = BASE_DIR.parent / "shared" / "public.tenseal"                  

LOG_PATH = BASE_DIR.parent / "logs" / "audit.log"
LOG_PATH.parent.mkdir(exist_ok=True)

app = FastAPI()

# ── Load model data ─────────────────────────────────────────────────────
model   = jl.load(f"{MODEL_DATA}/vault_med_model")
weights = np.load(f"{MODEL_DATA}/vault_weights.npy")  # 1024 floats
bias    = np.load(f"{MODEL_DATA}/vault_bias.npy")

# ── Load public context (no secret key) ───────────────────────────────────
with open(CONTEXT_PATH, "rb") as f:
    public_context = ts.context_from(f.read())


# ── Audit Logging  ───────────────────────────────────
def log_request(encrypted_bytes: bytes, response_hex: str):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "payload_hash": hashlib.sha256(encrypted_bytes).hexdigest(),
        "payload_size_kb": round(len(encrypted_bytes) / 1024, 2),
        "response_hash": hashlib.sha256(bytes.fromhex(response_hex)).hexdigest()
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    encrypted_bytes = await file.read()

    # Deserialize encrypted feature vector
    enc_features = ts.ckks_vector_from(public_context, encrypted_bytes)

    # Homomorphic dot product + bias
    enc_score = enc_features.dot(weights) + bias
    response_hex = enc_score.serialize().hex()

    # Audit logging
    log_request(encrypted_bytes,response_hex)

    return {"encrypted_score": response_hex}