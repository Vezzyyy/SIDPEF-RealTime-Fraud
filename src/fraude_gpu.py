#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Étape 5 : Conception et développement de la solution IA
Détection de fraude bancaire (type Attijariwafa bank)

Modèles comparés :
  1. Régression logistique        (CPU, baseline)
  2. Random Forest                (CPU)
  3. XGBoost                      (GPU NVIDIA via CUDA)  <- modèle principal
  4. MLP PyTorch                  (GPU NVIDIA via CUDA)  <- Deep Learning

Utilisation :
    python fraude_gpu.py --data-dir ./data --out-dir ./resultats
"""
import argparse
import glob
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : les graphiques sont enregistrés en PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
TARGET = "Class"


# ----------------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------------
def titre(txt):
    print("\n" + "=" * 70 + f"\n{txt}\n" + "=" * 70)


def trouver_fichier(dossier, nom):
    res = glob.glob(os.path.join(dossier, "**", nom), recursive=True)
    if not res:
        raise FileNotFoundError(f"{nom} introuvable dans '{dossier}'. "
                                f"Copie les CSV du Drive dans ce dossier.")
    return res[0]


def evaluer(nom, y_true, proba, seuil):
    pred = (proba >= seuil).astype(int)
    return {"Modèle": nom,
            "PR-AUC": average_precision_score(y_true, proba),
            "ROC-AUC": roc_auc_score(y_true, proba),
            "Seuil": seuil,
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred)}


def meilleur_seuil_f1(y_true, proba):
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[np.argmax(f1)])


# ----------------------------------------------------------------------------
# MLP PyTorch (GPU) avec une interface type scikit-learn
# ----------------------------------------------------------------------------
class TorchMLP:
    def __init__(self, device, hidden=(128, 64), dropout=0.2, lr=1e-3,
                 batch_size=2048, max_epochs=80, patience=10, seed=42):
        self.device, self.hidden, self.dropout, self.lr = device, hidden, dropout, lr
        self.batch_size, self.max_epochs, self.patience, self.seed = batch_size, max_epochs, patience, seed
        self.best_epoch = None

    def _build(self, d):
        couches, prev = [], d
        for h in self.hidden:
            couches += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        couches.append(nn.Linear(prev, 1))
        return nn.Sequential(*couches).to(self.device)

    def _proba(self, Xt):
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(Xt)).squeeze(1).cpu().numpy()

    def fit(self, X, y, X_val=None, y_val=None, epochs=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32, device=self.device)
        if X_val is not None:
            Xv = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32, device=self.device)

        self.model = self._build(Xt.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        # Poids de la classe fraude : racine du ratio (compromis stable, le seuil corrige le reste)
        ratio = float((yt == 0).sum() / max((yt == 1).sum(), 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio ** 0.5], device=self.device))

        n_epochs = epochs if epochs else self.max_epochs
        meilleur, etat, sans_progres = -1.0, None, 0
        for ep in range(1, n_epochs + 1):
            self.model.train()
            perm = torch.randperm(Xt.shape[0], device=self.device)
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
            if X_val is not None and epochs is None:  # early stopping sur la PR-AUC de validation
                ap = average_precision_score(y_val, self._proba(Xv))
                if ap > meilleur:
                    meilleur, sans_progres, self.best_epoch = ap, 0, ep
                    etat = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    sans_progres += 1
                    if sans_progres >= self.patience:
                        break
        if etat is not None:
            self.model.load_state_dict(etat)
        return self

    def predict_proba(self, X):
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
        p = self._proba(Xt)
        return np.column_stack([1 - p, p])


# ----------------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data", help="dossier contenant train_prepare.csv et test_prepare.csv")
    ap.add_argument("--out-dir", default="./resultats", help="dossier de sortie")
    ap.add_argument("--n-iter", type=int, default=25, help="nb d'essais de la recherche d'hyperparamètres")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- GPU ---------------------------------------------------------------
    titre("0. Vérification du GPU")
    gpu = torch.cuda.is_available()
    device_torch = torch.device("cuda" if gpu else "cpu")
    device_xgb = "cuda" if gpu else "cpu"
    if gpu:
        print(f"GPU détecté : {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go)")
    else:
        print("ATTENTION : aucun GPU CUDA détecté, exécution sur CPU (plus lent). "
              "Vérifie le driver NVIDIA et l'installation de PyTorch CUDA.")

    # ---- 1. Données --------------------------------------------------------
    titre("1. Chargement des données")
    p_train = trouver_fichier(args.data_dir, "train_prepare.csv")
    p_test = trouver_fichier(args.data_dir, "test_prepare.csv")
    train, test = pd.read_csv(p_train), pd.read_csv(p_test)
    print(p_train, train.shape)
    print(p_test, test.shape)
    print(f"Fraudes train : {int(train[TARGET].sum())} ({train[TARGET].mean() * 100:.3f} %)")
    print(f"Fraudes test  : {int(test[TARGET].sum())} ({test[TARGET].mean() * 100:.3f} %)")

    # ---- 2. Features -------------------------------------------------------
    titre("2. Préparation des features")
    exclus = [TARGET, "Time"]
    features = [c for c in train.columns if c not in exclus and pd.api.types.is_numeric_dtype(train[c])]
    X_train, y_train = train[features].astype("float32"), train[TARGET].astype(int)
    X_test, y_test = test[features].astype("float32"), test[TARGET].astype(int)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"{len(features)} features : {features}")
    print(f"scale_pos_weight = {spw:.1f}")

    # Validation temporelle : 80 % premiers du train / 20 % derniers
    cut = int(len(X_train) * 0.8)
    Xa, ya = X_train.iloc[:cut], y_train.iloc[:cut]
    Xv, yv = X_train.iloc[cut:], y_train.iloc[cut:]
    print(f"Validation : {len(Xv)} lignes, {int(yv.sum())} fraudes")

    xgb_base = dict(eval_metric="aucpr", tree_method="hist", device=device_xgb,
                    random_state=args.seed)

    # ---- 3. Comparaison par défaut ----------------------------------------
    titre("3. Entraînement et comparaison (réglages par défaut, validation)")
    val_models = {
        "Régression logistique": make_pipeline(StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, n_jobs=-1, random_state=args.seed),
        "XGBoost (GPU, défaut)": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                  scale_pos_weight=spw, **xgb_base),
        "MLP PyTorch (GPU)": TorchMLP(device_torch, seed=args.seed),
    }
    lignes = []
    for nom, m in val_models.items():
        t = time.time()
        if isinstance(m, TorchMLP):
            m.fit(Xa, ya, Xv, yv)
            print(f"  MLP : meilleure époque = {m.best_epoch}")
        else:
            m.fit(Xa, ya)
        lignes.append(evaluer(nom, yv, m.predict_proba(Xv)[:, 1], 0.5))
        print(f"  {nom} entraîné en {time.time() - t:.1f}s")
    tab_val = pd.DataFrame(lignes).round(4).sort_values("PR-AUC", ascending=False)
    print(tab_val.drop(columns="Seuil").to_string(index=False))
    tab_val.to_csv(os.path.join(args.out_dir, "comparaison_validation.csv"), index=False)
    mlp_epochs = val_models["MLP PyTorch (GPU)"].best_epoch

    # ---- 4. Paramétrage XGBoost -------------------------------------------
    titre("4. Paramétrage d'XGBoost (RandomizedSearchCV + TimeSeriesSplit, sur GPU)")
    param_dist = {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5],
        "scale_pos_weight": [spw * 0.1, spw * 0.5, spw],
    }
    search = RandomizedSearchCV(XGBClassifier(**xgb_base), param_distributions=param_dist,
                                n_iter=args.n_iter, scoring="average_precision",
                                cv=TimeSeriesSplit(n_splits=4), n_jobs=1,
                                random_state=args.seed, verbose=1)
    search.fit(X_train, y_train)
    best_params = search.best_params_
    print("Meilleurs paramètres :", best_params)
    print(f"PR-AUC (validation croisée temporelle) : {search.best_score_:.4f}")

    # ---- 5. Seuils de décision (choisis sur la validation, jamais sur le test)
    titre("5. Réglage des seuils de décision")
    xgb_tuned_val = XGBClassifier(**best_params, **xgb_base).fit(Xa, ya)
    probas_val = {
        "Régression logistique": val_models["Régression logistique"].predict_proba(Xv)[:, 1],
        "Random Forest": val_models["Random Forest"].predict_proba(Xv)[:, 1],
        "XGBoost (GPU, réglé)": xgb_tuned_val.predict_proba(Xv)[:, 1],
        "MLP PyTorch (GPU)": val_models["MLP PyTorch (GPU)"].predict_proba(Xv)[:, 1],
    }
    seuils = {nom: meilleur_seuil_f1(yv, p) for nom, p in probas_val.items()}
    for nom, s in seuils.items():
        print(f"  Seuil F1 max - {nom} : {s:.4f}")

    # ---- 6. Modèles finaux sur tout le train, évaluation sur le TEST --------
    titre("6. Entraînement final et évaluation sur le TEST")
    finaux = {
        "Régression logistique": make_pipeline(StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                  min_samples_leaf=2, n_jobs=-1, random_state=args.seed),
        "XGBoost (GPU, réglé)": XGBClassifier(**best_params, **xgb_base),
        "MLP PyTorch (GPU)": TorchMLP(device_torch, seed=args.seed),
    }
    probas_test, lignes = {}, []
    for nom, m in finaux.items():
        t = time.time()
        if isinstance(m, TorchMLP):
            m.fit(X_train, y_train, epochs=mlp_epochs or 30)
        else:
            m.fit(X_train, y_train)
        probas_test[nom] = m.predict_proba(X_test)[:, 1]
        lignes.append(evaluer(nom, y_test, probas_test[nom], seuils[nom]))
        print(f"  {nom} : {time.time() - t:.1f}s")
    tab_test = pd.DataFrame(lignes).round(4).sort_values("PR-AUC", ascending=False)
    print("\nRÉSULTATS SUR LE TEST")
    print(tab_test.to_string(index=False))
    tab_test.to_csv(os.path.join(args.out_dir, "comparaison_modeles_test.csv"), index=False)

    # Courbes Precision-Recall
    plt.figure(figsize=(7, 5))
    for nom, p in probas_test.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        plt.plot(rc, pr, label=f"{nom} (PR-AUC={average_precision_score(y_test, p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Courbes Precision-Recall (test)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "courbes_precision_recall.png"), dpi=150); plt.close()

    # Matrice de confusion du meilleur modèle (PR-AUC)
    meilleur = tab_test.iloc[0]["Modèle"]
    pred = (probas_test[meilleur] >= seuils[meilleur]).astype(int)
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Fraude"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Fraude"])
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel"); ax.set_title(f"Matrice de confusion : {meilleur}")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "matrice_confusion.png"), dpi=150); plt.close()
    tn, fp, fn, tp = cm.ravel()
    print(f"\nMeilleur modèle : {meilleur}")
    print(f"Fraudes détectées : {tp}/{tp + fn} | Fausses alertes : {fp} | Fraudes manquées : {fn}")

    # ---- 7. Explicabilité --------------------------------------------------
    titre("7. Explicabilité (importance des variables XGBoost)")
    xgb_final = finaux["XGBoost (GPU, réglé)"]
    imp = pd.Series(xgb_final.feature_importances_, index=features).sort_values().tail(15)
    plt.figure(figsize=(7, 5)); imp.plot(kind="barh")
    plt.title("Top 15 des variables (XGBoost)"); plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "importance_variables.png"), dpi=150); plt.close()
    try:
        import shap
        ech = X_test.sample(min(2000, len(X_test)), random_state=args.seed)
        sv = shap.TreeExplainer(xgb_final).shap_values(ech)
        shap.summary_plot(sv, ech, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "shap_summary.png"), dpi=150); plt.close()
        print("Graphique SHAP enregistré.")
    except Exception as e:
        print(f"SHAP ignoré ({type(e).__name__}) : l'importance des variables est enregistrée.")

    # ---- 8. Sauvegarde -----------------------------------------------------
    titre("8. Sauvegarde")
    xgb_final.save_model(os.path.join(args.out_dir, "xgboost_fraude.json"))
    mlp = finaux["MLP PyTorch (GPU)"]
    torch.save(mlp.model.state_dict(), os.path.join(args.out_dir, "mlp_fraude.pt"))
    joblib.dump(mlp.scaler, os.path.join(args.out_dir, "mlp_scaler.joblib"))
    config = {"features": features, "seuils": seuils, "meilleur_modele": meilleur,
              "xgboost_params": best_params, "mlp_epochs": mlp_epochs,
              "gpu": torch.cuda.get_device_name(0) if gpu else None,
              "duree_totale_s": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "config_modele.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"Tout est enregistré dans : {os.path.abspath(args.out_dir)}")
    print(f"Durée totale : {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
