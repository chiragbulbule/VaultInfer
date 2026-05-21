from pathlib import Path
import tenseal as ts

BASE_DIR = Path(__file__).parent.resolve()          # vault_med_model/client_and_server/client/
SHARED_DIR = BASE_DIR.parent / "shared"             # vault_med_model/client_and_server/shared/
SHARED_DIR.mkdir(exist_ok=True)

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40

with open(BASE_DIR / "secret.tenseal", "wb") as f:
    f.write(context.serialize(save_secret_key=True))

with open(SHARED_DIR / "public.tenseal", "wb") as f:
    f.write(context.serialize(save_secret_key=False))

print("Keys generated.")
print(f"Secret key saved to : {BASE_DIR / 'secret.tenseal'}")
print(f"Public key saved to : {SHARED_DIR / 'public.tenseal'}")