from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi.responses import JSONResponse
import openai
import os
import json
import faiss
import math
from sentence_transformers import SentenceTransformer
from openai import OpenAIError
from fastapi.middleware.cors import CORSMiddleware
from .search import search_perfumes

# ========== FastAPI app setup ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

# ========== Load Model & Index ==========
model = SentenceTransformer("all-mpnet-base-v2")
INDEX_PATH = "backendcopy/data/perfumes1.faiss"
ID_MAP_PATH = "backendcopy/data/id_map1.json"
index = faiss.read_index(INDEX_PATH)
with open(ID_MAP_PATH, "r") as f:
    id_map = json.load(f)

# ========== Pydantic Models ==========
class PerfumeResult(BaseModel):
    name: str
    main_accords: List[str]
    keyword_matches: int
    similarity_pct: Optional[float]
    rating_value: Optional[float]
    rating_count: Optional[float]
    gender_str: str
    url: str

class SearchResponse(BaseModel):
    men: List[PerfumeResult]
    women: List[PerfumeResult]
    unisex: List[PerfumeResult]

class GuidedInput(BaseModel):
    mood: str
    occasion: str
    notes: List[str]
    gender: str

class RewordInput(BaseModel):
    query: str

class UserProfile(BaseModel):
    gender: str
    liked_notes: List[str]
    disliked_notes: Optional[List[str]] = []

class SurveyInput(BaseModel):
    user_id: str
    scent_families: List[str]
    strength: str
    occasion: str
    season: str

# ========== Helpers ==========
user_survey_data: Dict[str, dict] = {}

def safe_float(val):
    return float(val) if val is not None and not math.isnan(val) else None

def to_model(item: dict) -> PerfumeResult:
    pct = item.get("score_norm")
    return PerfumeResult(
        name=item["name"],
        main_accords=item["main_accords"],
        keyword_matches=item["keyword_matches"],
        similarity_pct=safe_float(pct * 100) if pct is not None else None,
        rating_value=safe_float(item.get("rating_value")),
        rating_count=safe_float(item.get("rating_count")),
        gender_str=item["gender_str"],
        url=item["url"],
    )

def get_similar_perfumes(name: str, top_k: int = 5):
    match = next((item for item in id_map if item["name"].lower() == name.lower()), None)
    if not match:
        raise ValueError(f"Perfume '{name}' not found in index.")
    embedding = model.encode([match["name"]])
    embedding = embedding.astype("float32")
    D, I = index.search(embedding, top_k + 1)
    results = []
    for idx in I[0]:
        item = id_map[idx]
        if item["name"].lower() != name.lower():
            results.append(item)
    return results

# ========== Routes ==========

@app.get("/search", response_model=SearchResponse)
def perfume_search(
    q: str = Query(..., min_length=3),
    per_category: int = Query(5, ge=1, le=20),
    fetch_factor: int = Query(5, ge=1, le=20),
):
    raw = search_perfumes(q, per_category=per_category, fetch_factor=fetch_factor)
    return {
        "men": [to_model(r) for r in raw["top_men"]],
        "women": [to_model(r) for r in raw["top_women"]],
        "unisex": [to_model(r) for r in raw["top_unisex"]],
    }

@app.post("/guided-search", response_model=SearchResponse)
def guided_search(input: GuidedInput):
    prompt = (
        f"A user is looking for a fragrance. Mood: {input.mood}. "
        f"Occasion: {input.occasion}. Preferred notes: {', '.join(input.notes)}. "
        f"Gender preference: {input.gender}. "
        f"Suggest search keywords for a fragrance recommendation system."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    keywords = response['choices'][0]['message']['content']
    raw = search_perfumes(keywords)
    return {
        "men": [to_model(r) for r in raw["top_men"]],
        "women": [to_model(r) for r in raw["top_women"]],
        "unisex": [to_model(r) for r in raw["top_unisex"]],
    }

@app.post("/reword")
def reword_query(data: RewordInput):
    prompt = (
        f"The user entered a vague fragrance query: '{data.query}'. "
        "Reword and expand it into a more descriptive search query using perfume-related vocabulary."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    refined_query = response['choices'][0]['message']['content']
    return {"refined_query": refined_query}

@app.post("/recommend", response_model=SearchResponse)
def recommend_from_profile(profile: UserProfile):
    note_str = ', '.join(profile.liked_notes)
    prompt = (
        f"A {profile.gender} user likes the notes: {note_str}. "
        f"Disliked notes: {', '.join(profile.disliked_notes)}. "
        "Suggest keywords for fragrance search."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    keywords = response['choices'][0]['message']['content']
    raw = search_perfumes(keywords)
    return {
        "men": [to_model(r) for r in raw["top_men"]],
        "women": [to_model(r) for r in raw["top_women"]],
        "unisex": [to_model(r) for r in raw["top_unisex"]],
    }

@app.get("/similar", response_model=List[PerfumeResult])
def similar_perfumes(name: str = Query(..., min_length=2), top_k: int = 5):
    try:
        results = get_similar_perfumes(name, top_k)
        return [to_model(item) for item in results]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/submit-survey")
def submit_survey(data: SurveyInput):
    user_survey_data[data.user_id] = data.dict()
    return {"success": True}

@app.get("/recommend-from-survey/{user_id}", response_model=SearchResponse)
def recommend_from_survey(user_id: str):
    profile = user_survey_data.get(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Survey not found")

    prompt = (
        f"The user prefers {', '.join(profile['scent_families'])} scents, "
        f"{profile['strength'].lower()} strength, for {profile['occasion'].lower()} occasions, "
        f"and is shopping for the {profile['season'].lower()} season. "
        "Recommend fragrance search keywords."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    keywords = response['choices'][0]['message']['content']
    raw = search_perfumes(keywords)
    return {
        "men": [to_model(r) for r in raw["top_men"]],
        "women": [to_model(r) for r in raw["top_women"]],
        "unisex": [to_model(r) for r in raw["top_unisex"]],
    }

@app.exception_handler(OpenAIError)
async def openai_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "OpenAI API error", "details": str(exc)},
    )
