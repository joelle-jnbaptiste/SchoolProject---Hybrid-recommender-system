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

with open(os.path.join(MODEL_DIR, "knn.pkl"), "rb") as f:
    knn_model = pickle.load(f)


with open(os.path.join(MODEL_DIR, "anti_testset.pkl"), "rb") as f:
    anti_testset = pickle.load(f)

with open(os.path.join(MODEL_DIR, "trainset.pkl"), "rb") as f:
    trainset = pickle.load(f)

meta = pd.read_csv(os.path.join(MODEL_DIR, "articles_metadata.csv"))

clicks = pd.read_parquet(os.path.join(MODEL_DIR, "clicks.parquet"))

emb = np.load(os.path.join(MODEL_DIR, "embeddings.npy")).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

from .code.knn_model import recommend_cf_knn_means
from .code.faiss_model import recommend_for_user_faiss
from .code.hybrid_model import hybrid_recommend

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
        knn_results = recommend_cf_knn_means(
            user_id=user_id,
            trainset=trainset,
            anti_testset=anti_testset,
            model=knn_model,
            clicks_df=clicks,
            meta_df=meta,
            n=5
        )


    except Exception as e:
        knn_results = []

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

    try:
        hybrid_results = hybrid_recommend(
            cf_results=knn_results,
            faiss_results=faiss_results,
            k_final=5
        )

    except Exception as e:
        hybrid_results = []


    # FORMAT OUTPUT JSON
    response = {
        "user_id": user_id,
        "knn": [{"article_id": art, "score": float(score)} for (art, score) in knn_results],
        "faiss": [{"article_id": art, "score": float(score)} for (art, score) in faiss_results],
        "hybrid": [{"article_id": art, "score": float(score)} for (art, score) in hybrid_results]
    }

    return func.HttpResponse(
        json.dumps(response),
        status_code=200,
        mimetype="application/json"
    )
