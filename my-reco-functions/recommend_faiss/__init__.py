import logging
import json
import azure.functions as func
import pickle
import numpy as np
import pandas as pd
import faiss
import os

# LOAD MODELS 
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "models")

meta = pd.read_csv(os.path.join(MODEL_DIR, "articles_metadata.csv"))

clicks = pd.read_parquet(os.path.join(MODEL_DIR, "clicks.parquet"))

emb = np.load(os.path.join(MODEL_DIR, "embeddings.npy")).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)


from .code.faiss_model import recommend_for_user_faiss

# MAIN FUNCTION
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Recommender Azure Function called.")

    try:
        body = req.get_json()
        user_id = int(body["user_id"])
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": "Invalid request: missing user_id"}),
            status_code=400
        )

    # RUN ALL 3 MODELS
    try:
        faiss_results = recommend_for_user_faiss(
            user_id=user_id,
            clicks_df=clicks,
            emb=emb,
            index=index,
            meta=meta,
            n=5
        )
    except Exception as e:
        faiss_results = []



    # FORMAT OUTPUT JSON
    response = {
        "user_id": user_id,
        "faiss": [{"article_id": art, "score": float(score)} for (art, score) in faiss_results],
    }

    return func.HttpResponse(
        json.dumps(response),
        status_code=200,
        mimetype="application/json"
    )
