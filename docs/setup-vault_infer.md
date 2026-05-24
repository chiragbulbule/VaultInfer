## Running VaultInfer

### Installation

```
cd vault_infer/client_and_server
pip install -r requirements.txt
```

---

### Environment Configuration _(Optional but Recommended)_

Creating a Hugging Face token disables authentication warnings and unlocks higher download rate limits for embedding weights.

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Go to **Settings → Access Tokens** and create a token with **Read** access.
3. Create a `.env` file inside `vault_infer/data/` and add your token:
   HF_TOKEN=your_actual_token_here

> [!WARNING]
> **Windows Symlink Warning:** When downloading weights for the first time, Hugging Face may warn about file symlinks. Fix this with one of the following options:

<details>
<summary><strong>Option A — Enable Developer Mode (Recommended)</strong></summary>

1. Open **Windows Settings**.
2. Go to **Privacy & security → For developers** (or search "Developer settings").
3. Toggle **Developer Mode** on and accept the prompt.

</details>

<details>
<summary><strong>Option B — Set a System Environment Variable</strong></summary>

1. Search for **"Edit the system environment variables"** in the Start Menu and open it.
2. Click **Environment Variables...** at the bottom right.
3. Under **User variables**, click **New...** and set:
   - **Variable name:** `HF_HUB_DISABLE_SYMLINKS_WARNING`
   - **Variable value:** `1`
4. Click **OK** on all windows to save.

> **Note:** Restart your terminal or IDE for the change to take effect.

</details>

<details>
<summary><strong>Option C — One-Time Administrator Run</strong></summary>

Run your terminal or VS Code as **Administrator** on the first launch of `server.py`. Once weights are downloaded and cached, normal permissions work fine.

</details>

---

### Starting the Server

```bash
cd vault_infer/client_and_server/server
uvicorn server:app --reload
```

> [!NOTE]
> The server runs at `http://127.0.0.1:8000` by default.

---

### Starting the Client

Open a **new terminal**, then:

```bash
cd vault_infer/client_and_server/client
python client.py
```

---

## Reproducing the Model from Scratch

### Retraining the Classifier

```bash
cd vault_infer
python model.py
```

> [!NOTE]
> `model.py` imports from `dataset.py` and `sentence_embedding.py` automatically — running it alone is sufficient.

> [!WARNING]
> Retraining will **overwrite** the committed weights. The committed weights achieve **99.4% accuracy** with **0.007 SD** across stratified 5-fold cross-validation.
