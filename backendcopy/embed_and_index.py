from load_data import load_and_clean_data
from vectorizer import load_model, build_faiss_index

# Load data
df = load_and_clean_data()

# Load Hugging Face model
model = load_model()

# Build and save FAISS index
build_faiss_index(df, model)