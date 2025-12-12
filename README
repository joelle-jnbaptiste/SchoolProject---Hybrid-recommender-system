# Recommender System — Hybrid Pipeline (KNNMeans / Content-Based / FAISS)

Ce projet implémente un **système de recommandation complet**, combinant plusieurs approches (Collaborative Filtering, Content-Based, FAISS) avec un **déploiement partiel via Azure Functions** et une interface **Streamlit** pour tester les prédictions localement.

---

## Objectifs du projet

- Analyser les données de clics utilisateurs.
- Construire différents modèles de recommandation :
  - **KNN Baseline**
  - **KNN With Means**
  - **Content-Based (corrélation entre catégories)**
  - **FAISS (embeddings haute dimension)**
  - **Modèle Hybride** (fusion CF + FAISS)
- Mettre en place un **front Streamlit** permettant d’afficher des recommandations par utilisateur.
- Déployer une **Azure Function** pour servir un modèle léger (FAISS) en production.
- Structurer un pipeline complet : preprocessing → modèles → stockage → API.

---

## Architecture du système

### Mode local (Streamlit)
Utilise **tous les modèles** :
- KNN Means (CF)
- Content-Based
- FAISS
- Modèle hybride

Permet d’avoir une vue complète avant le déploiement.

### Mode production (Azure Functions)
Déployé uniquement avec :
- **FAISS** (modèle le plus léger et scalable)


### Stockage Azure
- **Table Storage** : utilisateurs et articles
- **Blob Storage** : embeddings + index FAISS
- **Function App** : logique de recommandation

---

## Modèles implémentés

### ** KNN Baseline & KNNMeans**
Basés sur les similarités entre utilisateurs, selon leurs clics par **catégorie d’articles**  
→ `Surprise` n’est pas utilisé, mais une logique similaire est réécrite.

### ** Content-Based**
Recommande :
- les catégories les plus similaires,
- puis les articles les plus populaires dans ces catégories,
- tout en excluant ceux déjà vus.

Approche idéale pour le **cold-start**.

### ** FAISS**
- Utilisation d’embeddings d’articles
- Vectorisation → Index FAISS → Recherche rapide kNN
- Modèle déployé en production (performance + taille réduite)

### ** Modèle Hybride**
Combinaison pondérée : score_final = α * score_CF + β * score_FAISS

Avantages :
- Qualité supérieure
- Couverture améliorée
- Meilleure diversité de recommandations

---

## Évaluation des modèles

Plusieurs métriques sont calculées :
- **Hit Rate**
- **Precision@k**
- **Recall@k**
- **nDCG**

Les modèles basés CF ont montré peu de performance (dataset très sparse).  
FAISS obtient de meilleurs résultats grâce à la structure des embeddings.

---

## Stack technique

| Composant | Technologie |
|----------|-------------|
| Front local | Streamlit |
| API | Azure Functions (Python) |
| Visualisation | Streamlit |

---

## 📁 Structure du repository
project/
│
├── src/
│ ├── preprocessing/
│ ├── models/
│ ├── faiss/
│ ├── azure_function/
│ └── streamlit_app/
│
├── data/
├── models/
├── environment.yml
├── requirements.txt
└── README.md
