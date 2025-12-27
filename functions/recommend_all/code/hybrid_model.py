def normalize_scores(results):
    """
    Normalize a list of (item, score) pairs using min-max scaling.

    This function rescales all scores to the [0, 1] interval so they can be
    meaningfully combined with scores coming from other models (e.g., FAISS or CF).
    If all scores are identical, each item receives a normalized score of 1.0.

    Args:
        results (list of tuples): List of (item_id, score) pairs.

    Returns:
        list of tuples: List of (item_id, normalized_score) pairs.
    """
    if not results:
        return []

    scores = [s for _, s in results]
    mn, mx = min(scores), max(scores)

    if mn == mx:
        return [(item, 1.0) for item, _ in results]

    return [(item, (s - mn) / (mx - mn)) for item, s in results]


def hybrid_recommend(cf_results, faiss_results, k_final=5):
    """
    Produce a hybrid recommendation list by combining CF and FAISS scores.

    Both score lists are first normalized using min-max scaling. Items appearing
    in both models receive the sum of their normalized scores. Items appearing
    in only one model keep only that model's contribution. The final hybrid
    score is averaged based on the number of contributing models.

    Args:
        cf_results (list of tuples): List of (item_id, score) pairs from
            the collaborative filtering model.
        faiss_results (list of tuples): List of (item_id, score) pairs from
            the content-based FAISS model.
        k_final (int): Number of recommendations to return.

    Returns:
        list of tuples: The top-k_final (item_id, hybrid_score) pairs sorted
        by descending hybrid score.
    """
    if not cf_results and not faiss_results:
        return []

    cf_norm = normalize_scores(cf_results)
    faiss_norm = normalize_scores(faiss_results)
    
    combined = {}

    for item, score in cf_norm:
        combined[item] = combined.get(item, 0) + score

    for item, score in faiss_norm:
        combined[item] = combined.get(item, 0) + score

    for item in combined:
        divisor = 0
        divisor += 1 if cf_norm else 0
        divisor += 1 if faiss_norm else 0
        combined[item] /= divisor

    final = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return final[:k_final]
