import faiss
import numpy as np
from collections import Counter
from model import get_embedding

INDEX_PATH = "index/landmark.index"
IMAGE_PATHS = "embeddings/image_paths.npy"
LANDMARK_IDS = "embeddings/landmark_ids.npy"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
image_paths = np.load(IMAGE_PATHS)
landmark_ids = np.load(LANDMARK_IDS) 

def query_image(image_path, k=5):

    print("Generating embedding...")
    query_vector = get_embedding(image_path).astype("float32")

    query_vector = np.expand_dims(query_vector, axis=0)

    print("Searching FAISS index...")

    distances, indices = index.search(query_vector, k)

    neighbors = indices[0]

    print("\nNearest images:")

    neighbor_landmarks = []

    for idx in neighbors:
        img = image_paths[idx]
        landmark = landmark_ids[idx]

        print(f"{img} → landmark {landmark}")

        neighbor_landmarks.append(landmark)

    vote = Counter(neighbor_landmarks)

    prediction = vote.most_common(1)[0][0]
    confidence = vote[prediction] / k

    print("\nPrediction:")
    print("Landmark ID:", prediction)
    print("Confidence:", confidence)

    return prediction, confidence

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/query.py <image_path>")
        exit()

    image_path = sys.argv[1]

    query_image(image_path)