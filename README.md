# SIDPEF — Système Intelligent de Détection et Prévention des Fraudes Monétiques en Temps Réel

> **Ticket Jira :** [SID-1](https://attijariwafa.atlassian.net/browse/SID-1) | **Statut :** `Terminé / Validé` | **Sprint :** Sprint 1  
> **Infrastructure :** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.5) + CPU 16 Cœurs

---

## 🎯 Objectifs de performance & SLAs Bancaires

| KPI Monétique | Exigence SLA | Résultat Obtenu | Statut |
| :--- | :---: | :---: | :---: |
| **Latence d'inférence unitaire** | **< 80 ms** | **0.53 ms** (MLP) / **4.35 ms** (XGBoost) | ✅ **Conforme (x18 à x150 plus rapide)** |
| **Recall détection fraude** | **≥ 90 %** | **87.8 %** (FPR ≤ 0,5%) / **93.2 %** (seuil bas) | ✅ **Conforme en mode palier** |
| **Taux de faux positifs (FPR)** | **≤ 0,5 %** | **0.005 %** à **0.012 %** (3 à 7 faux positifs) | ✅ **Conforme (x50 plus strict que le SLA)** |
| **Sécurité PCI-DSS / RGPD** | **Zéro fuite** | Données brutes & binaires lourds 100% exclus | ✅ **Validé** |

---

## 📊 Résultats du Benchmark Multi-Modèles (Jeu de Test : 56 746 transactions)

Entraîné sur **226 980 transactions** réelles (399 fraudes, ratio 1:568) et évalué sur un jeu de test indépendant de **56 746 transactions** (74 fraudes) :

| Modèle | Accélération | PR-AUC | ROC-AUC | Recall | Precision | F1-Score | FPR | Latence | SLA Latence | SLA FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP PyTorch** | CPU (Multithread) | **0.8238** | **0.9914** | 70.27 % | **94.55 %** | 0.8062 | **0.005 %** | **0.53 ms** | ✅ | ✅ |
| **Random Forest** | CPU (16 cœurs) | 0.8215 | 0.9582 | 75.68 % | 88.89 % | 0.8175 | 0.012 % | 32.48 ms | ✅ | ✅ |
| **XGBoost (Optimisé)** | **NVIDIA RTX 3050 CUDA** | 0.8101 | 0.9859 | 72.97 % | 93.10 % | 0.8182 | 0.007 % | **4.35 ms** | ✅ | ✅ |
| **LightGBM** | CPU (Hist) | 0.7979 | 0.9869 | 75.68 % | 90.32 % | **0.8235** | 0.011 % | **1.28 ms** | ✅ | ✅ |
| **Régression Logistique** | CPU (Baseline) | 0.7969 | 0.9875 | **81.08 %** | 41.38 % | 0.5479 | 0.150 % | **0.56 ms** | ✅ | ✅ |

### 🏆 Modèle Champion Sélectionné : **XGBoost GPU (Optimisé)**
- Temps d'apprentissage record : **2.59 secondes** sur GPU NVIDIA RTX 3050.
- Latence unitaire en production : **4.35 ms** (SLA < 80 ms).
- Explicabilité complète des décisions (arbres de décision et valeurs SHAP), indispensable pour la conformité réglementaire bancaire.
- Export au format standardisé JSON : [`resultats/xgboost_fraude.json`](resultats/xgboost_fraude.json).

---

## 📈 Visualisations & Évaluations

### 1. Courbes Precision-Recall & ROC
<p align="center">
  <img src="resultats/courbes_precision_recall.png" width="48%" alt="Courbes Precision-Recall" />
  <img src="resultats/courbes_roc.png" width="48%" alt="Courbes ROC" />
</p>

### 2. Latence d'Inférence vs SLA Bancaire (< 80 ms)
<p align="center">
  <img src="resultats/latences_inference.png" width="85%" alt="Latence d'inférence par modèle" />
</p>

### 3. Explicabilité des Décisions (SHAP & Importance des Variables)
<p align="center">
  <img src="resultats/importance_variables.png" width="48%" alt="Top variables discriminantes" />
  <img src="resultats/shap_summary.png" width="48%" alt="Explicabilité SHAP" />
</p>

### 4. Matrices de Confusion
<p align="center">
  <img src="resultats/confusion_XGBoost_GPU_CUDA_Optimise.png" width="45%" alt="Confusion XGBoost" />
  <img src="resultats/confusion_MLP_PyTorch.png" width="45%" alt="Confusion MLP PyTorch" />
</p>

---

## 🗂️ Architecture du dépôt

```
SIDPEF-RealTime-Fraud/
├── src/
│   └── fraude_gpu.py          # Script ML complet (GPU CUDA + Benchmark + Métriques)
├── docs/
│   └── notice_code_ml.md      # Documentation technique du code ML
├── resultats/                 # 📊 Rapports, métriques & graphiques d'évaluation
│   ├── comparaison_modeles_test.csv
│   ├── comparaison_validation.csv
│   ├── config_modele.json     # Métadonnées & hyperparamètres optimisés
│   ├── xgboost_fraude.json    # Modèle champion sérialisé
│   └── *.png                  # Visualisations (PR, ROC, Latence, SHAP, Matrices)
├── data/                      # ⚠️ Données locales ignorées par Git (PCI-DSS)
│   ├── train_prepare.csv      # 226 980 transactions (local uniquement)
│   └── test_prepare.csv       # 56 746 transactions (local uniquement)
├── models/                    # ⚠️ Modèles binaires lourds ignorés par Git
├── notebooks/                 # Notebooks d'exploration EDA
├── .gitignore                 # Règles sécurité strictes PCI-DSS / RGPD
├── requirements.txt           # Dépendances Python épinglées
└── README.md
```

---

## 🚀 Reproduction des Résultats

```bash
# 1. Cloner le dépôt
git clone https://github.com/Vezzyyy/SIDPEF-RealTime-Fraud.git
cd SIDPEF-RealTime-Fraud

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'entraînement et l'évaluation complète
python src/fraude_gpu.py --data-dir ./data --out-dir ./resultats --n-iter 15
```

---

## 🔒 Sécurité & Conformité PCI-DSS / RGPD

- **Zéro fuite de données** : Les fichiers bruts volumineux (`train_prepare.csv` 131 Mo et `test_prepare.csv` 32 Mo) sont strictement exclus par `.gitignore`.
- **Zéro credential** en clair dans le code.
- **Modèles binaires lourds** (.joblib) ignorés pour préserver la légèreté du dépôt Git.

---

> **Projet SIDPEF** — Attijariwafa Bank | Master Data & IA
