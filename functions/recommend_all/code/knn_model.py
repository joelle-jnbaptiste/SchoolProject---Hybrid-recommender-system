import pandas as pd

def recommend_cf_knn_means(user_id, trainset, anti_testset, model, clicks_df, meta_df, n=5):
    """
    Generate top-N category-based article recommendations using a 
    KNNWithMeans collaborative filtering model.

    The function works in three steps:
    1. For the given user, predict ratings for all items in the anti-testset
       using the trained Surprise KNNWithMeans model.
    2. Select the top-N highest predicted categories.
    3. For each selected category, identify the most popular article 
       (based on historical click frequency) and return one representative 
       article per category.

    This method ensures that recommendations respect the user's preferred 
    categories while promoting the most relevant content within each category.

    Args:
        user_id (int): ID of the target user.
        trainset (surprise.Trainset): The Surprise training dataset.
        anti_testset (list): List of (user, item, default_rating) tuples
            used for prediction on unseen items.
        model (surprise.KNNWithMeans): Trained collaborative filtering model.
        clicks_df (pd.DataFrame): User click interactions containing 
            'user_id' and 'click_article_id'.
        meta_df (pd.DataFrame): Article metadata containing at least
            'article_id' and 'category_id'.
        n (int): Number of recommendations to produce.

    Returns:
        list of tuples: A list of (article_id, score) pairs, one per top category.
    """
    preds = []
    for (uid, iid, _) in anti_testset:
        if int(uid) == int(user_id):
            est = model.predict(uid, iid).est
            preds.append((iid, est)) 

    preds.sort(key=lambda x: x[1], reverse=True)


    top_categories = [cat for (cat, score) in preds[:n]]


    reco_articles = []

    for cat in top_categories:
        subset = meta_df[meta_df["category_id"] == int(cat)]

        if subset.empty:
            continue

        pop = (
            clicks_df[clicks_df["click_article_id"].isin(subset["article_id"])]
            .groupby("click_article_id").size()
            .sort_values(ascending=False)
        )

        if len(pop) == 0:
            continue

        best_article = int(pop.index[0])
        best_score = float(pop.iloc[0])

        reco_articles.append((best_article, best_score))

    return reco_articles[:n]
