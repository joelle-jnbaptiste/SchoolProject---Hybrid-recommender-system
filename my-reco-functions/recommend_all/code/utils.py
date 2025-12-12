def top_k(results, k):
    return sorted(results, key=lambda x: x[1], reverse=True)[:k]
