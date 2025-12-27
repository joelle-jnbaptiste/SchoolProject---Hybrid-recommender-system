<p align="center">
  <img src="https://img.shields.io/github/stars/joelle-jnbaptiste/SchoolProject---Hybrid-recommender-system?style=for-the-badge" />
  <img src="https://img.shields.io/github/issues/joelle-jnbaptiste/SchoolProject---Hybrid-recommender-system?style=for-the-badge" />
  <img src="https://img.shields.io/github/license/joelle-jnbaptiste/SchoolProject---Hybrid-recommender-system?style=for-the-badge" />
  <img src="https://img.shields.io/badge/School%20Project-ML%20%26%20Data-blueviolet?style=for-the-badge" />
</p>

<h1 align="center">🧙‍♂️ Recommender System — Hybrid Pipeline 🏰</h1>

<p align="center">
  <em>
    A complete hybrid recommendation engine combining collaborative filtering,
    content-based logic, and FAISS-powered vector search — forged as a full
    end-to-end machine learning system.
  </em>
</p>

---

## 📜 About The Project

This project is a **school end-to-end recommender system** designed to showcase a **hybrid recommendation pipeline**.

It combines multiple recommendation paradigms into a single coherent system:

- 🧠 **Collaborative Filtering** (KNN & KNNMeans)
- 🧾 **Content-Based Filtering**
- 🗡️ **FAISS Vector Search** for scalable similarity search
- ⚖️ **Hybrid Scoring Strategy** (fusion of CF + FAISS)
- 🧪 Offline evaluation
- 🧩 Streamlit interface for exploration
- ☁️ Azure Functions deployment for production inference

The project is structured to reflect **real-world ML engineering practices**, from experimentation to deployment.

---

## 🛠️ Built With

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-Statistical%20Viz-4C72B0?style=for-the-badge" />
</p>



---

## 🧙 Dataset — The Source of Knowledge

This project relies on the **Fruits Image Dataset**, used here to simulate a catalog of items for recommendation experiments.

🔗 Dataset link:  
https://www.kaggle.com/datasets/moltean/fruits

The dataset provides:
- Labeled product categories
- Visual diversity
- A realistic foundation for embedding-based similarity

---

## 🎯 Project Objectives

- Analyze user–item interactions
- Implement multiple recommendation strategies
- Compare traditional CF with embedding-based search
- Build a **hybrid recommender**
- Deploy a lightweight model in production
- Expose predictions via API and UI
- Follow ML engineering best practices

---

## 🏗️ System Architecture

### 🧪 Local Mode — Streamlit Grimoire

- Run all models locally
- Explore predictions per user
- Compare algorithms side-by-side

Models available:
- KNN Baseline
- KNNMeans
- Content-Based
- FAISS
- Hybrid fusion

---

### ☁️ Production Mode — Azure Realm

- Deployed via **Azure Functions**
- Only FAISS-based inference (light & scalable)
- Optimized for low latency

Azure Components:
- Blob Storage → embeddings & FAISS index
- Table Storage → users & items
- Function App → inference logic

---

## 🧠 Implemented Models

### 🧩 KNN Baseline & KNNMeans
- User-based collaborative filtering
- Similarity computed from user interactions
- Baseline for comparison

---

### 📜 Content-Based Filtering
- Category similarity
- Popularity-aware recommendations
- Ideal for cold-start users

---

### ⚔️ FAISS Vector Search
- High-dimensional embeddings
- Fast kNN search
- Scalable and production-ready

---

### 🏰 Hybrid Model
Weighted fusion:

    score_final = α × score_CF + β × score_FAISS

Benefits:
- Higher recommendation quality
- Better coverage
- Improved diversity

---

## 📊 Model Evaluation

Metrics used:
- Hit Rate
- Precision@k
- Recall@k
- nDCG

Observations:
- CF struggles on sparse data
- FAISS excels in relevance
- Hybrid model provides the best balance

---

## 📁 Repository Structure

    SchoolProject---Hybrid-recommender-system/
    ├── front/
    │   └── app.py                # Streamlit interface
    │
    ├── functions/
    │   ├── recommend_all/         # Azure Function (full)
    │   ├── recommend_faiss/       # Azure Function (FAISS only)
    │   ├── host.json
    │   └── local.settings.json
    │
    ├── modelisation/
    │   └── notebooks             # Experiments & training
    │
    ├── .gitignore
    ├── .gitattributes
    └── README.md

---

## 🧭 Final Notes

This project was designed as a **pedagogical yet realistic ML system**, bridging:

- Data science experimentation
- Engineering constraints
- Deployment trade-offs

It demonstrates how **hybrid recommenders** can be built, evaluated, and deployed in a modern production environment.

🧙 *May your embeddings be dense and your recommendations precise.*
