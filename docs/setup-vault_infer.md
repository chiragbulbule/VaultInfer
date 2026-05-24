## Step by Step Process to Run VaultInfer

### To Install Required Dependancies

```bash
cd vault_infer/client_and_server
pip install -r requirements.txt
```

---

### Environment Configuration _(Optional but Recommended)_

To disable the Hugging Face authentication warning and enable higher rate limits for downloading embedding weights:

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Navigate to **Settings -> Access Tokens** and create a new token with **Read** access.
3. Create a `.env` file preferably inside the `data/` folder (vault_infer/data/.env) and add your token without spaces:

   `HF_TOKEN=your_actual_token_here`

---

> [!WARNING]
> **Windows Symlinks Warning:** When downloading the weights for the very first time, Hugging Face may throw a warning regarding file symlinks. To fix or suppress this completely on Windows, choose one of the following options:

<details>
<summary><strong>Option A: Enable Windows Developer Mode (Recommended)</strong></summary>
<br>

1. Open **Windows Settings**.
2. Go to **Privacy & security -> For developers** (or search "Developer settings").
3. Toggle **Developer Mode** to On and accept the prompt.

</details>

<details>
<summary><strong>Option B: Add a Global System Environment Variable</strong></summary>
<br>

If you cannot enable Developer Mode, you can tell Hugging Face to bypass symlinks globally:

1. Search for **"Edit the system environment variables"** in the Start Menu and open it.
2. Click **Environment Variables...** at the bottom right.
3. Under **User variables**, click **New...** and set:
   - **Variable name:** `HF_HUB_DISABLE_SYMLINKS_WARNING`
   - **Variable value:** `1`
4. Click **OK** on all windows to save.

> **Note:** Restart your terminal or IDE for the change to take effect.

</details>

<details>
<summary><strong>Option C: One-Time Administrator Run</strong></summary>
<br>

1. Simply run your terminal or VS Code as Administrator the very first time you launch `client.py`. After this, it can be run normally.

</details>

> [!Note]
> During the first run, the sentence transformer model is downloaded and the weights are cached.

---

### To Start the Server

1. Open a new terminal

```bash
cd vault_infer/client_and_server/server
uvicorn server:app --reload
```

> [!NOTE]
> By default the server will run on `http://127.0.0.1:8000`

---

### To Start the Client

1. Open a new terminal

```bash
cd vault_infer/client_and_server/client
python client.py
```

---

## To Reproduce the Model from Scratch

### To Retrain the Classifier

```bash
cd vault_infer
python model.py
```

---

> [!NOTE]
> `model.py` imports from `dataset.py` and `sentence_embedding.py` automatically. Running it alone is sufficient.

> [!WARNING]
> Retraining will overwrite the existing committed weights. The committed weights achieve 99.4% accuracy with 0.007 SD across stratified 5-fold cross validation.
