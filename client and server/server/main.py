import tenseal as ts
import numpy as np
import os
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vault_inference import encrypted_forward_pass

app = FastAPI(title="Vault-LLM Secure Gateway")


# In-memory store for computed results (keyed by job_id)
result_store: dict = {}

# Load trained weights and bias
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights = np.load(os.path.join(BASE_DIR, "vault_weights.npy"))
bias = np.load(os.path.join(BASE_DIR, "vault_bias.npy"))


     

class EncryptedPayload(BaseModel):
    context: list
    data: list

@app.post("/compute")
async def compute(payload: EncryptedPayload):
    try:
        client_context = ts.context_from(bytes(payload.context))
        client_context.make_context_public()  # ← ADD THIS

        enc_vector = ts.ckks_vector_from(client_context, bytes(payload.data))

        score=encrypted_forward_pass(enc_vector)

        job_id = str(uuid.uuid4())
        result_store[job_id] = list(score.serialize())
        return {"job_id": job_id}

    except Exception as e:
        print(f"❌ Compute Error: {e}")
        return {"error": str(e)}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """
    Client GETs the encrypted result using the job_id.
    """
    if job_id not in result_store:
        raise HTTPException(status_code=404, detail="Result not found. Invalid or expired job_id.")

    result = result_store.pop(job_id)  # pop to free memory after retrieval
    return {"result": result}