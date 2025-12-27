import requests
import streamlit as st
import pandas as pd
import pandas as pd
import random


# LOAD USER IDS 
@st.cache_data
def load_user_ids_for_all(path_csv):
    df = pd.read_csv(path_csv)
    return sorted(df["user_id"].unique())


@st.cache_data
def load_user_ids_faiss(path_csv):
    df = pd.read_csv(path_csv)
    return sorted(df["user_id"].unique())

user_ids_all = load_user_ids_for_all("data/knn_users.csv")
user_ids_faiss = load_user_ids_faiss("data/faiss_users.csv")

user_ids_all_sampled = random.sample(user_ids_all, min(20, len(user_ids_all)))
user_ids_faiss_sampled = random.sample(user_ids_faiss, min(20, len(user_ids_faiss)))


if "user_ids_all_sampled" not in st.session_state:
    st.session_state.user_ids_all_sampled = random.sample(
        user_ids_all, min(20, len(user_ids_all))
    )

if "user_ids_faiss_sampled" not in st.session_state:
    st.session_state.user_ids_faiss_sampled = random.sample(
        user_ids_faiss, min(20, len(user_ids_faiss))
    )

user_ids_all_sampled = st.session_state.user_ids_all_sampled
user_ids_faiss_sampled = st.session_state.user_ids_faiss_sampled


# FUNCTIONS TO CALL MODELS 
def run_all_models(user_id):
    url = "http://localhost:7071/api/recommend_all"

    payload = {"user_id": int(user_id)}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except Exception as e:
        st.error(f"❌ Erreur API recommend_all : {e}")
        return [], [], []

    data = response.json()

    reco_knn = [
        (item["article_id"], float(item["score"]))
        for item in data.get("knn", [])
    ]

    reco_faiss = [
        (item["article_id"], float(item["score"]))
        for item in data.get("faiss", [])
    ]

    reco_hybrid = [
        (item["article_id"], float(item["score"]))
        for item in data.get("hybrid", [])
    ]

    return reco_knn, reco_faiss, reco_hybrid


def run_faiss_only(user_id):
    url = "https://recommendapp-dtbddsh2a9aygkhd.francecentral-01.azurewebsites.net/api/recommend_faiss?code=4gIls9i5t7idtTIhgFgUokPwnHY-orYBOULFlMzawpn3AzFuw9RVjw=="

    payload = {"user_id": int(user_id)}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
    except Exception as e:
        st.error(f"❌ Erreur API recommend_faiss : {e}")
        return []

    data = response.json()

    reco_faiss = [
        (item["article_id"], float(item["score"]))
        for item in data.get("faiss", [])
    ]

    return reco_faiss


# STREAMLIT UI
st.title("Recommendation System Dashboard")


# SECTION ALL MODELS
st.subheader("Local Recommendations — ALL MODELS (KNNMeans / FAISS / Hybrid)")

selected_user_all = st.selectbox(
    "Select a user:",
    user_ids_all_sampled,
    key="user_all"
)



if st.button("Run ALL models"):
    reco_knn, reco_faiss, reco_hybrid = run_all_models(selected_user_all)

    reco_knn = sorted(reco_knn, key=lambda x: x[1], reverse=True)
    reco_faiss = sorted(reco_faiss, key=lambda x: x[1], reverse=True)
    reco_hybrid = sorted(reco_hybrid, key=lambda x: x[1], reverse=True)

    st.markdown(f"### User ID : **{selected_user_all}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### KNNMeans")
        for item, score in reco_knn[:5]:
            st.write(f"- **{item}** — {score:.4f}")

    with col2:
        st.markdown("#### FAISS")
        for item, score in reco_faiss[:5]:
            st.write(f"- **{item}** — {score:.4f}")

    with col3:
        st.markdown("#### Hybrid")
        for item, score in reco_hybrid[:5]:
            st.write(f"- **{item}** — {score:.4f}")


# SECTION FAISS ONLY

st.markdown("---")
st.subheader("FAISS Recommendations from Azure Function")

selected_user_faiss = st.selectbox(
    "Select a user (FAISS model):",
    user_ids_faiss_sampled,
    key="user_faiss_only"
)


if st.button("Run FAISS model"):
    reco_faiss_only = run_faiss_only(selected_user_faiss)

    reco_faiss_only = sorted(reco_faiss_only, key=lambda x: x[1], reverse=True)

    st.markdown(
        f"### User ID : **{selected_user_faiss}** \nTop 5 FAISS results:")

    for item, score in reco_faiss_only[:5]:
        st.write(f"- **{item}** — {score:.4f}")
