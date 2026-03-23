from sentence_transformers import SentenceTransformer
import numpy

model=SentenceTransformer('all-MiniLM-L6-v2')

sentence="This is my sentence"

embedding=model.encode(sentence)

print(f"Number of values are : {len(embedding)}")
print(f"First five values are : {embedding[:5]}")

weights=numpy.random.randn(384)
bias=-0.05

score=numpy.dot(embedding,weights) + bias

prediction="Alert" if score > 0 else "Normal"

print(f"Final score is  : {score:.4f}")
print(f"The prediction is : {prediction}")

