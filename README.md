<p align="center">
  <img src="https://img.shields.io/github/license/joelle-jnbaptiste/SchoolProject---Hybrid-recommender-system?style=for-the-badge" />
  <img src="https://img.shields.io/badge/School%20Project-ML%20%26%20Data-blueviolet?style=for-the-badge" />
</p>

<h1 align="center">✨ Recommender System — Hybrid Pipeline ✨</h1>

<div align="center">
  <em>
     Blending signals, embeddings and similarity to guide recommendations
  </em>
</br>

 <b>
   End-to-end hybrid recommender system combining collaborative filtering,
   content-based filtering and FAISS-based vector search
 </b>
</br>
</br>
🗃️ <b>Dataset</b>  

      https://www.kaggle.com/datasets/moltean/fruits
  
</div>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>🧭 Table of Contents</summary>
  <ol>
    <li>About The Project</li>
    <li>Dataset</li>
    <li>System Architecture</li>
    <li>Models presentation</li>
    <li>Model Evaluation</li>
    <li>Repository Structure</li>
    <li>Getting Started</li>
    <li>License</li>
    <li>Contact</li>
  </ol>
</details>

---

### ✨ Built With

[![Python][Python-shield]][Python-url]
[![Jupyter][Jupyter-shield]][Jupyter-url]
[![Pandas][Pandas-shield]][Pandas-url]
[![NumPy][NumPy-shield]][NumPy-url]
[![Matplotlib][Matplotlib-shield]][Matplotlib-url]
[![Seaborn][Seaborn-shield]][Seaborn-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## 🎯 About The Project

This project is a complete hybrid recommender system designed to showcase how
multiple recommendation paradigms can be combined into a single coherent pipeline.

It brings together:
- Collaborative filtering (KNN-based)
- Content-based recommendation
- FAISS vector search for scalable similarity
- A hybrid scoring strategy combining CF and vector similarity

The system is structured to reflect real-world machine learning engineering
practices, from experimentation to local inference and production-oriented design.

---

## 🗃️ Dataset

The project relies on the **Fruits Image Dataset**, used here to simulate a product
catalog for recommendation experiments.

Dataset link:

      https://www.kaggle.com/datasets/moltean/fruits

The dataset provides:
- Labeled product categories
- Visual diversity
- A realistic foundation for embedding-based similarity search

---

## 🏰 System Architecture

The system is organized into two complementary modes:

- **Local Mode**:
  - Streamlit interface for exploration
  - Side-by-side comparison of recommendation strategies
  - Interactive inspection of results

- **Production Mode**:
  - Azure Functions deployment
  - FAISS-based inference for scalability
  - Optimized for low latency and modular deployment

This separation ensures clarity between experimentation and production concerns.

---

## 🪄 Models presentation

The following models and strategies are implemented:

- **Collaborative Filtering**
  - User-based KNN
  - Item-based KNN
  - Baseline for behavioral similarity

- **Content-Based Filtering**
  - Category-based similarity
  - Popularity-aware recommendations
  - Useful for cold-start scenarios

- **FAISS Vector Search**
  - High-dimensional embeddings
  - Fast nearest-neighbor search
  - Production-ready similarity retrieval

- **Hybrid Model**
  - Weighted fusion of CF and FAISS scores
  - Improved coverage and recommendation quality

---

## 👑 Model Evaluation

Models are evaluated using recommender system metrics such as:

- Hit Rate
- Precision@K
- Recall@K
- nDCG

Key observations:
- Collaborative filtering struggles with sparse interactions
- FAISS excels in relevance-based similarity
- The hybrid model provides the best overall balance

---

## 🗺️ Repository Structure

    SchoolProject---Hybrid-recommender-system/
    ├── front/
    │   └── app.py                # Streamlit interface
    │
    ├── functions/
    │   ├── recommend_all/         # Azure Function (full hybrid)
    │   ├── recommend_faiss/       # Azure Function (FAISS only)
    │   ├── host.json
    │   └── local.settings.json
    │
    ├── modelisation/
    │   └── notebooks              # Experiments & training
    │
    ├── .gitignore
    ├── .gitattributes
    └── README.md

---

## ⚔️ Getting Started

### 1. Clone the repository

    git clone https://github.com/joelle-jnbaptiste/SchoolProject---Hybrid-recommender-system.git

### 2. Run the local Streamlit app

    cd front
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    streamlit run app.py

### 3. (Optional) Run Azure Functions locally

    cd functions
    func start

---

## ✒️ License

This project is intended for educational and demonstration purposes.

---

## 🕊️ Contact

Joëlle JEAN BAPTISTE  
LinkedIn:

      https://fr.linkedin.com/in/joëllejnbaptiste  

Project Link:

      https://github.com/joelle-jnbaptiste/SchoolProject---Hybrid-recommender-system

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

[Python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/

[Jupyter-shield]: https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white
[Jupyter-url]: https://jupyter.org/

[Pandas-shield]: https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]: https://pandas.pydata.org/

[NumPy-shield]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/

[Matplotlib-shield]: https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge
[Matplotlib-url]: https://matplotlib.org/

[Seaborn-shield]: https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge
[Seaborn-url]: https://seaborn.pydata.org/
