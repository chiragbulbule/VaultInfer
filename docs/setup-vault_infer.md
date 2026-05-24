**Step by Step Process to Run VaultInfer**

To Install Required Dependancies

1. run `cd vault_infer/client_and_server`
2. run `pip install -r requirements.txt`

Environment Configuration (Optional but Recommended)

To disable the Hugging Face authentication warning and enable higher rate limits for downloading embedding weights:

1. Go to huggingface.co and create a free account.
2. Navigate to Settings -> Access Tokens and create a new token with Read access.
3. Create a file named .env preferably inside the data/ folder (vault_infer/data/.env) and add your token without spaces:

   HF_TOKEN=your_actual_token_here

[!WARNING]

Windows Symlinks Warning: When downloading the weights for the very first time, Hugging Face may throw a warning regarding file symlinks. To resolve or disable this, either run your IDE/Terminal as Administrator for the first run, or turn on Developer Mode in your Windows system settings.

To Start the Server

1. Open a new terminal
2. run `cd vault_infer/client_and_server/server`
3. run `uvicorn server:app --reload`

> [!NOTE]
> By default the server will run on `http://127.0.0.1:8000`

To Start the Client

1. Open another new terminal
2. run `cd vault_infer/client_and_server/client`
3. run `python client.py`

**To Reproduce the Model from Scratch**

To Retrain the Classifier

1. Run `cd vault_infer`
2. Run `python model.py`

> [!NOTE]
> `model.py` imports from `dataset.py` and `sentence_embedding.py` automatically. Running it alone is sufficient.

> [!WARNING]
> Retraining will overwrite the existing committed weights. The committed weights achieve 99.4% accuracy with 0.007 SD across stratified 5-fold cross validation.
