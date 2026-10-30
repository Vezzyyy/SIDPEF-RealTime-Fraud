# SIDPEF — Système Intelligent de Détection et Prévention des Fraudes Monétiques en Temps Réel

> **Ticket Jira :** [SID-1](https://attijariwafa.atlassian.net/browse/SID-1) | **Statut :** `In Progress` | **Sprint :** Sprint 1

## 🎯 Objectifs de performance

| KPI | Cible | Statut |
|-----|-------|--------|
| Latence d'inférence | **< 80 ms** | 🔄 En cours |
| Recall détection fraude | **≥ 90 %** | 🔄 En cours |
| Taux faux positifs (FPR) | **≤ 0,5 %** | 🔄 En cours |
| Conformité PCI-DSS / RGPD | **Zéro commit données** | ✅ Validé |

---

## 🗂️ Architecture du dépôt

```
SIDPEF-RealTime-Fraud/
├── src/
│   └── fraude_gpu.py          # Script ML principal (XGBoost GPU + MLP PyTorch)
├── docs/
│   └── notice_code_ml.md      # Documentation technique du code ML
├── data/                      # ⚠️ Ignoré par Git (PCI-DSS)
│   ├── train_prepare.csv      # Données d'entraînement (local uniquement)
│   └── test_prepare.csv       # Données de test (local uniquement)
├── models/                    # ⚠️ Ignoré par Git (binaires lourds)
├── notebooks/                 # Notebooks d'exploration EDA
├── .gitignore                 # Règles sécurité PCI-DSS / RGPD
├── requirements.txt           # Dépendances Python
└── README.md
```

---

## 🚀 Installation & Utilisation

### Prérequis
- Python ≥ 3.10
- NVIDIA GPU + CUDA ≥ 11.8 (fallback CPU automatique)
- Données locales dans `data/` (non versionnées)

### Installation

```bash
pip install -r requirements.txt
# PyTorch GPU (adapter selon CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Lancement de l'entraînement

```bash
python src/fraude_gpu.py --data-dir ./data --out-dir ./resultats
```

### Options disponibles

```
--data-dir   Dossier contenant train_prepare.csv et test_prepare.csv (défaut: ./data)
--out-dir    Dossier de sortie des résultats et graphiques (défaut: ./resultats)
--n-iter     Nombre d'itérations RandomizedSearchCV XGBoost (défaut: 25)
--seed       Graine aléatoire pour reproductibilité (défaut: 42)
```

---

## 🤖 Modèles comparés

| Modèle | Accélération | Description |
|--------|-------------|-------------|
| Régression Logistique | CPU | Baseline de référence |
| Random Forest | CPU | Ensemble robuste |
| **XGBoost** | **GPU (CUDA)** | **Modèle principal — hyperparamètres optimisés** |
| MLP PyTorch | GPU (CUDA) | Deep Learning avec early stopping |

---

## 📊 Sorties générées (`./resultats/`)

| Fichier | Description |
|---------|-------------|
| `comparaison_validation.csv` | Métriques sur la validation temporelle |
| `comparaison_modeles_test.csv` | Métriques finales sur le jeu de test |
| `courbes_precision_recall.png` | Courbes PR pour tous les modèles |
| `matrice_confusion.png` | Matrice de confusion du meilleur modèle |
| `importance_variables.png` | Top 15 features XGBoost |
| `shap_summary.png` | Explicabilité SHAP (si installé) |
| `xgboost_fraude.json` | Modèle XGBoost exporté |
| `mlp_fraude.pt` | Poids MLP PyTorch |
| `config_modele.json` | Configuration complète + métriques |

---

## 🔒 Sécurité & Conformité

- **PCI-DSS / RGPD** : Toutes les données CSV, modèles `.pkl`/`.joblib`/`.pt` sont exclus du dépôt via `.gitignore`
- **Aucune clé / credential** en clair dans le code
- **Variables d'environnement** via `.env` (ignoré par Git)

---

## 📋 Contexte Projet

**Périmètre :** Détection de fraude monétique temps-réel pour Attijariwafa Bank
- Canaux couverts : Attijari Mobile, GAB (Distributeurs automatiques), E-commerce
- Stack : Python · XGBoost · PyTorch · scikit-learn · SHAP

> **Master Data & IA** — Projet académique à vocation bancaire
