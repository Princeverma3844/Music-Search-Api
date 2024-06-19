from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.clap_model import get_clap_model
from app.search import check_similarity, get_music_address
from app.database import music_cap_collection

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    index: list

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": 50}

@app.get("/predict", response_model=PredictionOut)
def predict(text: str = Query(..., description="The text input for prediction"), clap_model=Depends(get_clap_model)):
    prompt_embeddings = clap_model.get_text_embeddings([text])
    prompt_embeddings = prompt_embeddings.cpu().numpy()

    vec1 = check_similarity(music_cap_collection, prompt_embeddings, size=128, given_index=None)
    vec2 = check_similarity(music_cap_collection, prompt_embeddings, size=256, given_index=vec1)
    vec3 = check_similarity(music_cap_collection, prompt_embeddings, size=512, given_index=vec2)
    vec4 = check_similarity(music_cap_collection, prompt_embeddings, size=1024, given_index=vec3)

    return {"index": get_music_address(music_cap_collection, vec4)[:40]}

# uvicorn app.main:app --reload