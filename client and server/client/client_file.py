import tenseal as ts
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

print("🔄 Loading Sentence Transformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_client_context():
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60] 
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

context = create_client_context()

def run_vault_inference():
    user_text = input("\nEnter sentence to analyze: ")
    if not user_text:
        return

    # Step 1: Encode text → vector
    embedding = model.encode(user_text).tolist()

    # Step 2: Encrypt the embedding
    print("🔒 Encrypting embedding...")
    enc_vector = ts.ckks_vector(context, embedding)

    # Step 3: POST encrypted vector + context to server
    payload = {
        "context": list(context.serialize()),
        "data": list(enc_vector.serialize())
    }
    print("📡 POST → Sending encrypted vector to server...")
    try:
        post_response = requests.post("http://127.0.0.1:8000/compute", json=payload)
        post_json = post_response.json()

        if "error" in post_json:
            print(f"❌ Server Error: {post_json['error']}")
            return

        job_id = post_json["job_id"]
        print(f"✅ Job accepted. ID: {job_id}")

    except Exception as e:
        print(f"❌ POST failed: {e}")
        return

    # Step 4: GET the encrypted result using job_id
    print("📥 GET → Fetching encrypted result from server...")
    try:
        get_response = requests.get(f"http://127.0.0.1:8000/result/{job_id}")
        get_json = get_response.json()

        if "error" in get_json:
            print(f"❌ Server Error: {get_json['error']}")
            return

        # Step 5: Decrypt the result on client side
        enc_res_bytes = bytes(get_json["result"])
        enc_result = ts.ckks_vector_from(context, enc_res_bytes)
        score = enc_result.decrypt()[0]

        # Clamp to [0, 1] since polynomial is an approximation of sigmoid
        score = max(0.0, min(1.0, score))
        decision = 1 if score >= 0.5 else 0

        print("\n" + "=" * 40)
        print(f"SENTENCE  : {user_text}")
        print(f"SCORE     : {score:.6f}")
        print(f"DECISION  : {decision}  →  {'🚨 ALERT' if decision == 1 else '✅ NORMAL'}")
        print("=" * 40)

    except Exception as e:
        print(f"❌ GET failed: {e}")

if __name__ == "__main__":
    while True:
        run_vault_inference()
        if input("\nContinue? (y/n): ").lower() != 'y':
            break