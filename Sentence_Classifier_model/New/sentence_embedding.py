from vault_dataset import train_sentences,train_labels
from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

embedding=model.encode(train_sentences) # embedding.shape = (137,384)
train_labels=train_labels

#----------------------------------------------TEST CODE-----------------------------------------------#

"""em=model.encode(["Hydraulic pressure critical, shutdown imminent",
"The meeting is at 3pm",
"Unauthorized access detected at the server",
"I'm making a cup of tea"])"""

test_embed=model.encode("Hydraulic pressure critical, shutdown imminent") #shape-(384,)

