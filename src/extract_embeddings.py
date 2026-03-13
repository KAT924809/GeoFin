import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import get_embedding


DATASET_DIR = "data/raw/gldv2_micro"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "gldv2_micro.csv")


OUTPUT_DIR = "embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(CSV_PATH)

embeddings = []
image_paths = []
landmark_ids = []

print("Starting embedding extraction...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    filename = row["filename"]
    landmark_id = row["landmark_id"]

    image_path = os.path.join(IMAGE_DIR, filename)

    try:
        emb = get_embedding(image_path)

        embeddings.append(emb)
        image_paths.append(filename)
        landmark_ids.append(landmark_id)

    except Exception as e:
        print(f"Skipping {filename}: {e}")


embeddings = np.array(embeddings)
image_paths = np.array(image_paths)
landmark_ids = np.array(landmark_ids)


np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(OUTPUT_DIR, "image_paths.npy"), image_paths)
np.save(os.path.join(OUTPUT_DIR, "landmark_ids.npy"), landmark_ids)

print("Embedding extraction complete.")
print("Embeddings shape:", embeddings.shape)