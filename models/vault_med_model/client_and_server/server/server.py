from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import tenseal as ts
import joblib as jl
import hashlib
from datetime import datetime
import json

BASE_DIR     = Path(__file__).parent.resolve()                                 # vault_med_model/client_and_server/server/
VAULT_MED    = BASE_DIR.parent.parent                                          # vault_med_model/
MODEL_PATH   = VAULT_MED / "model_training" / "vault_med_model"                # model
CONTEXT_PATH = BASE_DIR.parent / "shared" / "public.tenseal"                   # vault_med_model/client_and_server/shared/public.tenseal

LOG_PATH = BASE_DIR.parent / "logs" / "audit.log"
LOG_PATH.parent.mkdir(exist_ok=True)

app = FastAPI()

# ── Load model weights ─────────────────────────────────────────────────────
model   = jl.load(MODEL_PATH)
weights = model.coef_.flatten().tolist()    # 1024 floats
bias    = float(model.intercept_[0])

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