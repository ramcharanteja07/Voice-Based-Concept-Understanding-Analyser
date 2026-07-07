def compute_concept_score(text, keywords):
    matched = sum(1 for word in keywords if word in text.lower())
    return round((matched / len(keywords)) * 10, 2)


def compute_filler_counts(text, filler_words):
    return {word: text.lower().count(word) for word in filler_words}


def compute_filler_ratio(text, filler_count):
    total_fillers = sum(filler_count.values())
    word_count = len(text.split())
    return round(total_fillers / max(word_count, 1), 2)


def compute_fluency_score(filler_ratio):
    fluency_score = max(0, 10 - (filler_ratio * 10))
    return round(fluency_score, 2)


def compute_audio_confidence_score(rms_energy, pause_ratio):
    energy_component = min(rms_energy * 100, 10)
    pause_penalty = pause_ratio * 10
    confidence_score = max(0, energy_component - pause_penalty)
    return round(min(confidence_score, 10), 2)


def compute_final_score(similarity, fluency_score, audio_confidence_score):
    weighted = (similarity * 10 * 0.5) + (fluency_score * 0.3) + (audio_confidence_score * 0.2)
    return round(weighted * 10, 2)


def classify_understanding(final_score):
    if final_score >= 70:
        return "Strong"
    elif final_score >= 40:
        return "Moderate"
    else:
        return "Poor"