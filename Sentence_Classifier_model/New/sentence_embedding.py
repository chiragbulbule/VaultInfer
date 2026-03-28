from vault_dataset import train_sentences,train_labels
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

embedding=model.encode(train_sentences)

# em=model.encode(["Hydraulic pressure critical, shutdown imminent",
# "The meeting is at 3pm",
# "Unauthorized access detected at the server",
# "I'm making a cup of tea"])

train_labels=train_labels

# embedding.shape = (370,384)