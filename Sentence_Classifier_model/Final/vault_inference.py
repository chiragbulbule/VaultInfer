import tenseal as ts
import numpy as np

weights=np.load("./Sentence_Classifier_model/Final/vault_weights.npy").tolist()
bias=np.load("./Sentence_Classifier_model/Final/vault_bias.npy").tolist()

def encrypted_forward_pass(encrypted_vector):
  ws = encrypted_vector.dot(weights) + float(bias[0])
  ws2 = ws * ws
  ws3 = ws2 * ws
  ws5 = ws2 * ws3

  score = 0.5 + (ws * 0.25) - (ws3 * (1/48)) + (ws5 * (1/480))
  return score