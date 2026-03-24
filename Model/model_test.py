from sentence_transformers import SentenceTransformer
import torch,torch.nn as nn

class VaultClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer=nn.Linear(384,1)

    def forward(self,x):
        return self.linear_layer(x)

my_model=VaultClassifier()
my_model.load_state_dict(torch.load("./VaultLLM/Model/vault_model.pth"))
my_model.eval()


model=SentenceTransformer("all-MiniLM-L6-v2")

test_normal=["Critical: Satellite link lost.", "Warning: Low battery on emergency lights.",
    "Alert: Unlocked terminal in open office.", "Emergency! Bomb threat received.",
    "Security alert: Tampering with smoke detector.", "Warning: High CO2 levels in conference room.",
    "Alert: Suspicious file added to startup.", "Emergency: Elevator fire detected.",
    "Critical: Main power line cut.", "Warning: Overload on circuit breaker 12."
    ]

test_abnormal=["The phone is ringing.", "I’m looking at the screen.",
    "The paper is white.", "I’m thinking about the future.",
    "The music is playing softly.", "I’m holding a book.",
    "The pen is on the table.", "I’m breathing deeply.",
    "The day is just beginning.", "I’m ending the work day."
]

embedding=model.encode(test_normal+test_abnormal)

user_tensor=torch.tensor(embedding).float()
score_list=my_model(user_tensor).flatten().tolist()

for score in score_list:
    print(f"Score:{score} : Alert") if score > 0 else print(f"Score:{score} : Normal")

