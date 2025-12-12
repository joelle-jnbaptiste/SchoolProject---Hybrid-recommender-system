import numpy as np

# USER PROFILE
def get_user_profile(user_id, clicks_df, emb, meta):
    """
    Compute the user's content-based profile embedding.

    This function retrieves all articles clicked by the user, maps them to their
    corresponding FAISS indices using the metadata table, and averages their
    embedding vectors. The resulting vector represents the user's content
    preferences and is used as the query for FAISS similarity search.

    Args:
        user_id (int):
            The ID of the user for whom the profile must be computed.
        clicks_df (pd.DataFrame):
            DataFrame containing user interactions with at least:
                - "user_id": user identifier
                - "click_article_id": clicked article identifier
        emb (np.ndarray):
            Embedding matrix where each row corresponds to a FAISS index.
        meta (pd.DataFrame):
            Metadata containing a column "article_id" and its associated "faiss_idx".

    Returns:
        np.ndarray or None:
            A float32 vector representing the user's averaged embedding, or
            None if no interactions or valid FAISS indices are found.
    """
    seen = clicks_df[clicks_df["user_id"] == user_id]["click_article_id"].tolist()

    if len(seen) == 0:
        return None

    faiss_ids = meta.loc[meta["article_id"].isin(seen), "faiss_idx"].values

    if len(faiss_ids) == 0:
        return None

    user_emb = emb[faiss_ids].mean(axis=0, keepdims=True).astype("float32")

    return user_emb


# FAISS 
def recommend_for_user_faiss(user_id, clicks_df, emb, index, meta, n=5):
    """
    Generate top-N content-based recommendations using FAISS similarity search.

    This function builds the user's embedding profile, queries the FAISS index
    for nearest-neighbor items, maps FAISS indices back to article IDs using
    the metadata table, and returns the N most relevant articles along with
    their similarity scores.

    Steps performed:
        1. Compute the user content profile (mean embedding of clicked items).
        2. Query FAISS for the closest items based on inner-product similarity.
        3. Validate FAISS indices and retrieve article metadata.
        4. Collect and return the top-N recommended article IDs.

    Args:
        user_id (int):
            The user for whom recommendations must be generated.
        clicks_df (pd.DataFrame):
            Clickstream data used to compute the user profile.
        emb (np.ndarray):
            Matrix of article embeddings aligned with FAISS indexing.
        index (faiss.Index):
            Pre-loaded FAISS index containing all embeddings.
        meta (pd.DataFrame):
            Metadata mapping FAISS indices ("faiss_idx") to article IDs.
        n (int, optional):
            Number of recommendations to return (default: 5).

    Returns:
        list[tuple[int, float]]:
            A list of (article_id, similarity_score) tuples sorted by relevance.
            Returns an empty list if the user has no clicks or FAISS fails.
    """
    user_emb = get_user_profile(user_id, clicks_df, emb, meta)

    if user_emb is None:
        return []

    try:
        scores, ids = index.search(user_emb, n + 20)
    except Exception as e:
        return []

    scores = scores[0]
    ids = ids[0]

    results = []

    for faiss_idx, score in zip(ids, scores):

        if faiss_idx < 0 or faiss_idx >= len(meta):
            continue

        row = meta.loc[meta["faiss_idx"] == int(faiss_idx)]

        if row.empty:
            continue

        article_id = int(row.iloc[0]["article_id"])
        results.append((article_id, float(score)))

    final_results = results[:n]

    return final_results
