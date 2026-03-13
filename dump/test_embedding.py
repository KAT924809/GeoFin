from model import get_embedding
import numpy as np
import os

# change this to any image you want to test
image_path = os.path.join(os.path.dirname(__file__), "test.jpg")

embedding = get_embedding(image_path)

print("Embedding type:", type(embedding))
print("Embedding shape:", embedding.shape)

print("First 10 values:")
print(embedding[:10])