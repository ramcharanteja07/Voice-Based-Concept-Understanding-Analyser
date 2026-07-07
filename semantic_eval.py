from sentence_transformers import SentenceTransformer, util
import streamlit as st


@st.cache_resource
def load_semantic_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)


def generate_embeddings(reference_text, transcribed_text, model=None):
    if model is None:
        model = load_semantic_model()
    return model.encode([reference_text, transcribed_text], convert_to_tensor=True)


def compute_raw_similarity(embeddings):
    return util.cos_sim(embeddings[0], embeddings[1]).item()


def normalize_similarity(raw_score):
    return round(max(0.0, min(1.0, raw_score)), 2)


def compute_similarity(reference_text, transcribed_text, model=None):
    embeddings = generate_embeddings(reference_text, transcribed_text, model)
    raw_score = compute_raw_similarity(embeddings)
    return normalize_similarity(raw_score)