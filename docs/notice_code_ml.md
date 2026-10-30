# Détection de fraude : version GPU (RTX 3050)

## 1. Prérequis
- Windows 10/11 avec un **driver NVIDIA récent** (https://www.nvidia.com/drivers). Pas besoin d'installer CUDA séparément.
- **Python 3.10, 3.11 ou 3.12** (https://www.python.org/downloads, cocher "Add Python to PATH").

## 2. Installation (PowerShell, dans ce dossier)
    python -m venv venv
    venv\Scripts\activate
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

## 3. Vérifier que le GPU est vu
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Doit afficher : True NVIDIA GeForce RTX 3050 ...

## 4. Données
Créer un dossier `data` à côté du script et y copier depuis le Drive :
- train_prepare.csv
- test_prepare.csv

## 5. Lancer
    python fraude_gpu.py --data-dir ./data --out-dir ./resultats
Option pour un test rapide : `--n-iter 5`

## 6. Résultats (dans ./resultats)
- comparaison_modeles_test.csv : tableau final (PR-AUC, Recall, Precision, F1)
- comparaison_validation.csv
- courbes_precision_recall.png, matrice_confusion.png
- importance_variables.png (et shap_summary.png)
- xgboost_fraude.json, mlp_fraude.pt, mlp_scaler.joblib, config_modele.json (seuils, paramètres)
