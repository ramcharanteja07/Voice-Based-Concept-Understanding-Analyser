import whisper
import streamlit as st


@st.cache_resource
def load_whisper_model(name="base"):
    return whisper.load_model(name)


def transcribe(path, model=None):
    if model is None:
        model = load_whisper_model()
    result = model.transcribe(path)
    return result["text"]


def transcribe_with_confidence(path, model=None):
    if model is None:
        model = load_whisper_model()
    result = model.transcribe(path)
    segments = result.get("segments", [])
    if segments:
        avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
    else:
        avg_logprob = 0
    return result["text"], round(avg_logprob, 3)