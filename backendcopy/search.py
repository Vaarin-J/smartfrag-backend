import faiss
import json
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Paths
INDEX_PATH  = "backendcopy/data/perfumes1.faiss"
ID_MAP_PATH = "backendcopy/data/id_map1.json"

# Load FAISS index and ID map
index = faiss.read_index(INDEX_PATH)
with open(ID_MAP_PATH, "r") as f:
    id_map = json.load(f)

# Boost maps based on structured preferences
SEASON_NOTE_BOOST = {
    "Fall": {"spicy": 0.1, "amber": 0.08, "woody": 0.06},
    "Summer": {"citrus": 0.1, "aquatic": 0.08, "green": 0.06},
    "Spring": {"floral": 0.1, "green": 0.08, "fruity": 0.06},
    "Winter": {"sweet": 0.1, "resinous": 0.08, "leather": 0.06}
}

STRENGTH_BOOST = {
    "Light": {"citrus": 0.1, "green": 0.08, "fresh": 0.06},
    "Moderate": {"floral": 0.08, "woody": 0.06},
    "Strong": {"oud": 0.1, "leather": 0.08, "amber": 0.06, "musk": 0.05}
}

OCCASION_BOOST = {
    "Date Night": {"sweet": 0.1, "musky": 0.08, "amber": 0.06},
    "Work": {"clean": 0.1, "fresh": 0.08, "green": 0.06},
    "Everyday": {"citrus": 0.08, "woody": 0.06},
    "Special Events": {"oriental": 0.1, "gourmand": 0.08, "oud": 0.06}
}

def classify_gender(gender_raw: str):
    g = str(gender_raw or "").lower()
    has_men = "men" in g or "male" in g
    has_women = "women" in g or "female" in g

    if has_men and has_women:
        return "unisex"
    if has_men:
        return "men"
    if has_women:
        return "women"
    return "unisex"

def get_similar_perfumes(name: str, top_k: int = 5):
    match = next((item for item in id_map if item["name"].lower() == name.lower()), None)
    if not match:
        return []
    embedding = model.encode([match["name"]])
    embedding = np.array(embedding).astype("float32")
    D, I = index.search(embedding, top_k + 1)
    similar_items = []
    for idx in I[0]:
        sim_item = id_map[idx]
        if sim_item["name"].lower() != name.lower():
            similar_items.append(sim_item)
    return similar_items

def search_perfumes(user_input: str, per_category: int = 5, fetch_factor: int = 10, survey_context: dict = None):
    embedding = model.encode([user_input])
    initial_n = per_category * fetch_factor
    distances, indices = index.search(
        np.array(embedding).astype("float32"),
        initial_n
    )

    query_tokens = set(re.findall(r"\w+", user_input.lower()))

    season = survey_context.get("season") if survey_context else None
    strength = survey_context.get("strength") if survey_context else None
    occasion = survey_context.get("occasion") if survey_context else None

    candidates = []
    for idx, dist in zip(indices[0], distances[0]):
        perfume = id_map[idx]
        score_norm = 1.0 / (1.0 + dist)

        raw_gender = perfume.get("gender", "unisex")
        gender_str = str(raw_gender).lower()
        category = classify_gender(raw_gender)

        accords_set = {a.lower() for a in perfume.get("main_accords", [])}
        keyword_matches = len(query_tokens & accords_set)
        match_boost = 0.05 * keyword_matches

        seasonal_boost = sum([SEASON_NOTE_BOOST.get(season, {}).get(a, 0) for a in accords_set])
        strength_boost = sum([STRENGTH_BOOST.get(strength, {}).get(a, 0) for a in accords_set])
        occasion_boost = sum([OCCASION_BOOST.get(occasion, {}).get(a, 0) for a in accords_set])

        final_score = score_norm + match_boost + seasonal_boost + strength_boost + occasion_boost

        candidates.append({
            "name": perfume.get("name", ""),
            "main_accords": perfume.get("main_accords", []),
            "rating_value": perfume.get("rating_value"),
            "rating_count": perfume.get("rating_count"),
            "rating_score": perfume.get("rating_score", 0),
            "gender_str": gender_str,
            "category": category,
            "url": perfume.get("url", ""),
            "score_norm": final_score,
            "keyword_matches": keyword_matches
        })

    candidates.sort(key=lambda x: (x["keyword_matches"], x["score_norm"]), reverse=True)

    top_men, top_women, top_unisex = [], [], []

    for cand in candidates:
        cat = cand["category"]
        gender_str = cand["gender_str"]
        if cat == "men" and len(top_men) < per_category:
            top_men.append(cand)
        elif cat == "women" and len(top_women) < per_category:
            top_women.append(cand)
        elif cat == "unisex" and "women" in gender_str and len(top_women) < per_category:
            top_women.append(cand)
        elif cat == "unisex" and len(top_unisex) < per_category:
            top_unisex.append(cand)
        if len(top_men) >= per_category and len(top_women) >= per_category and len(top_unisex) >= per_category:
            break

    def fallback_fill(bucket, target_category):
        if len(bucket) >= per_category:
            return
        fallback_candidates = [
            {
                "name": p.get("name", ""),
                "main_accords": p.get("main_accords", []),
                "rating_value": p.get("rating_value"),
                "rating_count": p.get("rating_count"),
                "rating_score": p.get("rating_score", 0),
                "gender_str": str(p.get("gender", "")).lower(),
                "category": classify_gender(p.get("gender", "")),
                "url": p.get("url", ""),
                "keyword_matches": len(query_tokens & set(map(str.lower, p.get("main_accords", []))))
            }
            for p in id_map
            if classify_gender(p.get("gender", "")) == target_category
        ]
        fallback_candidates.sort(key=lambda x: (x["keyword_matches"], x["rating_score"]), reverse=True)
        existing = {perf["name"] for perf in bucket}
        for f in fallback_candidates:
            if len(bucket) >= per_category:
                break
            if f["name"] not in existing:
                bucket.append(f)
                existing.add(f["name"])

    fallback_fill(top_men, "men")
    fallback_fill(top_women, "women")
    fallback_fill(top_unisex, "unisex")

    return {
        "top_men": top_men[:per_category],
        "top_women": top_women[:per_category],
        "top_unisex": top_unisex[:per_category],
    }
