#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIDPEF — Système Intelligent de Détection et Prévention des Fraudes Monétiques
Banque / Monétique : Attijari Mobile, GAB, E-commerce
Ticket Jira : SID-1
================================================================================
Benchmark et entraînement comparatif de 5 modèles ML / Deep Learning :
  1. Régression Logistique (Baseline CPU, class_weight='balanced')
  2. Random Forest        (CPU, multi-threaded)
  3. XGBoost              (Accélération GPU NVIDIA CUDA via RTX 3050)
  4. LightGBM             (CPU haute performance)
  5. MLP PyTorch          (Réseau de neurones profond avec régularisation)

Contraintes SLA SIDPEF bancaires :
  - Latence d'inférence < 80 ms (sur 1 transaction unitaire)
  - Recall détection    ≥ 90 %
  - Taux Faux Positifs  ≤ 0,5 %
  - PCI-DSS / RGPD      Zéro fuite de données
================================================================================
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Constantes & SLA ──────────────────────────────────────────────────────────
TARGET = "Class"
SEED = 42
SLA_RECALL_MIN = 0.90     # ≥ 90 %
SLA_FPR_MAX = 0.005       # ≤ 0.5 %
SLA_LATENCY_MS = 80.0     # < 80 ms
BANNER = "=" * 74


def titre(txt: str) -> None:
    print(f"\n{BANNER}\n  {txt}\n{BANNER}")


def detecter_gpu():
    """Détecte la disponibilité GPU CUDA pour XGBoost et PyTorch indépendamment."""
    # Test CUDA PyTorch
    torch_cuda = torch.cuda.is_available()
    device_torch = torch.device("cuda" if torch_cuda else "cpu")

    # Test CUDA XGBoost (peut fonctionner même si PyTorch est CPU)
    xgb_cuda = False
    try:
        dummy_xgb = XGBClassifier(tree_method="hist", device="cuda")
        dummy_xgb.fit(np.zeros((10, 2)), np.zeros(10))
        xgb_cuda = True
    except Exception:
        xgb_cuda = False

    device_xgb = "cuda" if xgb_cuda else "cpu"
    return device_torch, device_xgb, torch_cuda, xgb_cuda


def trouver_fichier(dossier: str, nom: str) -> str:
    p = Path(dossier)
    matches = list(p.rglob(nom))
    if not matches:
        raise FileNotFoundError(
            f"Fichier '{nom}' introuvable dans '{dossier}'. "
            f"Vérifiez l'emplacement dans data/."
        )
    return str(matches[0])


def calibrer_seuil_bancaire(y_true, proba, recall_cible=0.90, fpr_max=0.005) -> float:
    """
    Calibre le seuil de décision optimal selon les exigences bancaires SIDPEF :
    1. Recherche les seuils garantissant Recall >= 90% ET FPR <= 0.5%.
    2. Parmi les candidats conformes, retient celui qui maximise le F1-score.
    3. Si aucun seuil ne remplit simultanément les deux critères stricts,
       retient le seuil maximisant le F1-score.
    """
    grille = np.linspace(0.01, 0.98, 400)
    candidats_sla = []
    candidats_f1 = []

    for t in grille:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        candidats_f1.append((f1, t, rec, fpr, prec))
        if rec >= recall_cible and fpr <= fpr_max:
            candidats_sla.append((f1, prec, t, rec, fpr))

    if candidats_sla:
        # Tri : meilleur F1, puis meilleure précision
        candidats_sla.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return float(candidats_sla[0][2])

    # Fallback : seuil maximisant F1
    candidats_f1.sort(key=lambda x: x[0], reverse=True)
    return float(candidats_f1[0][1])


def evaluer_modele(nom: str, y_true, proba, seuil: float) -> dict:
    pred = (proba >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    rec = recall_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    pr_auc = average_precision_score(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)

    return {
        "Modèle": nom,
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Seuil": round(seuil, 4),
        "Recall": round(rec, 4),
        "Precision": round(prec, 4),
        "F1": round(f1, 4),
        "FPR": round(fpr, 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "SLA_Recall": "✅" if rec >= SLA_RECALL_MIN else "❌",
        "SLA_FPR": "✅" if fpr <= SLA_FPR_MAX else "❌",
    }


def mesurer_latence(model, X_sample, n_warmup=5, n_runs=100) -> float:
    """Mesure la latence d'inférence unitaire (1 transaction) en millisecondes."""
    # Prendre exactement 1 ligne
    if isinstance(X_sample, pd.DataFrame):
        one_row = X_sample.iloc[:1]
    else:
        one_row = X_sample[:1]

    # Warmup
    for _ in range(n_warmup):
        model.predict_proba(one_row)

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict_proba(one_row)
        timings.append((time.perf_counter() - t0) * 1000.0)

    return float(np.median(timings))


# ── Modèle Deep Learning MLP PyTorch ───────────────────────────────────────────

class TorchMLP:
    """Architecture MLP tabulaire avec BatchNorm, Dropout et CosineAnnealing."""

    def __init__(self, device, hidden=(128, 64, 32), dropout=0.2, lr=1e-3,
                 batch_size=2048, max_epochs=60, patience=8, seed=SEED):
        self.device = device
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.best_epoch = None
        self.model = None
        self.scaler = None

    def _build(self, d: int) -> nn.Sequential:
        couches, prev = [], d
        for h in self.hidden:
            couches += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ]
            prev = h
        couches.append(nn.Linear(prev, 1))
        return nn.Sequential(*couches).to(self.device)

    def _predict_raw(self, Xt: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.sigmoid(self.model(Xt)).squeeze(1).cpu().numpy()

    def fit(self, X, y, X_val=None, y_val=None, epochs: int = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32, device=self.device)

        Xv_t = None
        if X_val is not None:
            Xv_t = torch.tensor(self.scaler.transform(X_val), dtype=torch.float32, device=self.device)

        self.model = self._build(Xt.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        n_epochs = epochs or self.max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        ratio = float((yt == 0).sum() / max((yt == 1).sum(), 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio ** 0.5], device=self.device))

        best_ap, best_state, stagnation = -1.0, None, 0

        for ep in range(1, n_epochs + 1):
            self.model.train()
            perm = torch.randperm(Xt.shape[0], device=self.device)
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                out = self.model(Xt[idx]).squeeze(1)
                loss = loss_fn(out, yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            if Xv_t is not None and epochs is None:
                proba_val = self._predict_raw(Xv_t)
                ap = average_precision_score(y_val, proba_val)
                if ap > best_ap:
                    best_ap = ap
                    stagnation = 0
                    self.best_epoch = ep
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    stagnation += 1
                    if stagnation >= self.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X_arr)
            Xt = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            p = torch.sigmoid(self.model(Xt)).squeeze(1).cpu().numpy()
            if p.ndim == 0:
                p = np.array([float(p)])
            return np.column_stack([1.0 - p, p])


# ── Fonctions de tracé graphique ──────────────────────────────────────────────

def plot_pr_curves(probas_test: dict, y_test, out_dir: str):
    plt.figure(figsize=(9, 6))
    for nom, p in probas_test.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        auc_val = average_precision_score(y_test, p)
        plt.plot(rc, pr, lw=2.2, label=f"{nom} (PR-AUC = {auc_val:.4f})")
    plt.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline aléatoire ({y_test.mean():.4f})")
    plt.xlabel("Recall (Taux de détection)", fontsize=11)
    plt.ylabel("Precision (Valeur prédictive positive)", fontsize=11)
    plt.title("SIDPEF — Courbes Precision-Recall sur jeu de Test", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "courbes_precision_recall.png"), dpi=200)
    plt.close()


def plot_roc_curves(probas_test: dict, y_test, out_dir: str):
    plt.figure(figsize=(8, 6))
    for nom, p in probas_test.items():
        fpr_arr, tpr_arr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        plt.plot(fpr_arr, tpr_arr, lw=2.2, label=f"{nom} (ROC-AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("Taux de Faux Positifs (FPR)", fontsize=11)
    plt.ylabel("Taux de Vrais Positifs (Recall)", fontsize=11)
    plt.title("SIDPEF — Courbes ROC sur jeu de Test", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "courbes_roc.png"), dpi=200)
    plt.close()


def plot_latences(latences: dict, out_dir: str):
    noms = list(latences.keys())
    vals = [latences[n] for n in noms]
    colors = ["#27ae60" if v < SLA_LATENCY_MS else "#e74c3c" for v in vals]

    plt.figure(figsize=(9, 4.5))
    bars = plt.barh(noms, vals, color=colors, edgecolor="black", height=0.55)
    plt.axvline(SLA_LATENCY_MS, color="#c0392b", linestyle="--", lw=2, label=f"SLA Inférence < {SLA_LATENCY_MS} ms")
    for bar, v in zip(bars, vals):
        plt.text(v + 0.3, bar.get_y() + bar.get_height() / 2, f"{v:.2f} ms", va="center", fontsize=10, fontweight="bold")
    plt.xlabel("Latence médiane unitaire (ms)", fontsize=11)
    plt.title("SIDPEF — Latence d'inférence temps réel vs SLA Bancaire (< 80 ms)", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "latences_inference.png"), dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, nom: str, out_dir: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues", interpolation="nearest")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Légitime", "Fraude"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Légitime", "Fraude"], fontsize=10)
    ax.set_xlabel("Prédiction", fontsize=11)
    ax.set_ylabel("Vérité terrain", fontsize=11)
    ax.set_title(f"Matrice de Confusion\n{nom}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    nom_safe = nom.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.savefig(os.path.join(out_dir, f"confusion_{nom_safe}.png"), dpi=200)
    plt.close()


def plot_importance(model, features: list, out_dir: str):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=features).sort_values().tail(15)
    plt.figure(figsize=(8, 5.5))
    imp.plot(kind="barh", color="#2980b9", edgecolor="black")
    plt.title("SIDPEF — Top 15 Variables Discriminantes (XGBoost Gain)", fontsize=12, fontweight="bold")
    plt.xlabel("Importance relative (Gain)", fontsize=11)
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "importance_variables.png"), dpi=200)
    plt.close()


# ── Programme principal ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SIDPEF - Entraînement et benchmark modèles ML Fraude")
    parser.add_argument("--data-dir", default="./data", help="Chemin du dossier des données")
    parser.add_argument("--out-dir", default="./resultats", help="Chemin du dossier de sortie")
    parser.add_argument("--n-iter", type=int, default=15, help="Nombre d'itérations RandomizedSearchCV")
    parser.add_argument("--seed", type=int, default=SEED, help="Graine aléatoire")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # ── ÉTAPE 0 : Environnement matériel & GPU ────────────────────────────────
    titre("0. Détection de l'infrastructure matérielle")
    device_torch, device_xgb, torch_cuda, xgb_cuda = detecter_gpu()
    print(f"  PyTorch   : {device_torch} (CUDA: {torch_cuda})")
    print(f"  XGBoost   : {device_xgb} (Accélération NVIDIA CUDA: {xgb_cuda})")
    if xgb_cuda:
        print("  🚀 XGBoost utilisera la carte graphique NVIDIA RTX 3050 !")

    # ── ÉTAPE 1 : Chargement des données préparées ────────────────────────────
    titre("1. Chargement des données")
    path_train = trouver_fichier(args.data_dir, "train_prepare.csv")
    path_test = trouver_fichier(args.data_dir, "test_prepare.csv")
    print(f"  Jeu d'entraînement : {path_train}")
    print(f"  Jeu de test         : {path_test}")

    t0 = time.time()
    train_df = pd.read_csv(path_train)
    test_df = pd.read_csv(path_test)
    print(f"  Chargement effectué en {time.time() - t0:.2f}s")
    print(f"  Train : {train_df.shape[0]:,} lignes, {train_df.shape[1]} colonnes "
          f"| Fraudes: {int(train_df[TARGET].sum())} ({train_df[TARGET].mean()*100:.3f}%)")
    print(f"  Test  : {test_df.shape[0]:,} lignes, {test_df.shape[1]} colonnes "
          f"| Fraudes: {int(test_df[TARGET].sum())} ({test_df[TARGET].mean()*100:.3f}%)")

    # ── ÉTAPE 2 : Prétraitement et Split de Validation Temporelle ─────────────
    titre("2. Préparation des variables & Split temporel")
    exclus = [TARGET, "Time"]
    features = [c for c in train_df.columns if c not in exclus and pd.api.types.is_numeric_dtype(train_df[c])]
    print(f"  Nombre de features : {len(features)}")

    X_train = train_df[features].astype("float32")
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[features].astype("float32")
    y_test = test_df[TARGET].astype(int)

    spw = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"  Ratio de déséquilibre (scale_pos_weight) = {spw:.1f}")

    # Split temporel de validation : 80% train initial / 20% validation
    cut = int(len(X_train) * 0.80)
    Xa, ya = X_train.iloc[:cut], y_train.iloc[:cut]
    Xv, yv = X_train.iloc[cut:], y_train.iloc[cut:]
    print(f"  Sous-train pour validation : {len(Xa):,} lignes ({int(ya.sum())} fraudes)")
    print(f"  Validation temporelle       : {len(Xv):,} lignes ({int(yv.sum())} fraudes)")

    # ── ÉTAPE 3 : Définition des 5 modèles ────────────────────────────────────
    titre("3. Configuration des modèles candidats")
    xgb_base = {
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": device_xgb,
        "random_state": args.seed,
    }

    modeles_val = {
        "Régression Logistique": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed
        ),
        "XGBoost (GPU CUDA)": XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.8,
            scale_pos_weight=spw, **xgb_base
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.05,
            num_leaves=63, class_weight="balanced",
            subsample=0.85, colsample_bytree=0.8,
            n_jobs=-1, random_state=args.seed, verbose=-1
        ),
        "MLP PyTorch": TorchMLP(device_torch, seed=args.seed),
    }

    # ── ÉTAPE 4 : Entraînement & Évaluation sur Validation ─────────────────────
    titre("4. Phase 1 : Entraînement et calibration sur Validation (80% Train)")
    lignes_val = []
    probas_val_dict = {}

    for nom, mod in modeles_val.items():
        t0_m = time.time()
        if isinstance(mod, TorchMLP):
            mod.fit(Xa.values, ya.values, Xv.values, yv.values)
            p_val = mod.predict_proba(Xv.values)[:, 1]
            print(f"  [Entraîné] {nom:<22} en {time.time() - t0_m:.2f}s (Best epoch: {mod.best_epoch})")
        else:
            mod.fit(Xa, ya)
            p_val = mod.predict_proba(Xv)[:, 1]
            print(f"  [Entraîné] {nom:<22} en {time.time() - t0_m:.2f}s")

        probas_val_dict[nom] = p_val
        res = evaluer_modele(nom, yv, p_val, seuil=0.5)
        lignes_val.append(res)

    tab_val_defaut = pd.DataFrame(lignes_val)
    print("\n  Tableau comparatif préliminaire (Validation, seuil standard 0.50) :")
    print(tab_val_defaut[["Modèle", "PR-AUC", "ROC-AUC", "Recall", "Precision", "F1", "FPR"]].to_string(index=False))

    mlp_epochs = modeles_val["MLP PyTorch"].best_epoch or 35

    # ── ÉTAPE 5 : Optimisation Hyperparamètres XGBoost ─────────────────────────
    titre("5. Phase 2 : Optimisation hyperparamètres XGBoost (RandomizedSearchCV)")
    param_dist = {
        "n_estimators": [200, 350, 500],
        "max_depth": [4, 5, 6, 8],
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "scale_pos_weight": [spw * 0.3, spw * 0.6, spw],
    }

    cv_strat = StratifiedKFold(n_splits=4, shuffle=False)
    search = RandomizedSearchCV(
        XGBClassifier(**xgb_base),
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="average_precision",
        cv=cv_strat,
        n_jobs=1,
        random_state=args.seed,
        verbose=1
    )
    t0_tune = time.time()
    search.fit(X_train, y_train)
    best_xgb_params = search.best_params_
    print(f"  Recherche terminée en {time.time() - t0_tune:.2f}s")
    print(f"  Meilleurs hyperparamètres : {best_xgb_params}")
    print(f"  Meilleur score CV (PR-AUC) : {search.best_score_:.4f}")

    # Réentraîner XGBoost optimisé sur Xa pour calibrer son seuil
    xgb_opt_val = XGBClassifier(**best_xgb_params, **xgb_base).fit(Xa, ya)
    probas_val_dict["XGBoost (GPU CUDA, Optimisé)"] = xgb_opt_val.predict_proba(Xv)[:, 1]

    # ── ÉTAPE 6 : Calibration des Seuils de Décision Bancaires ─────────────────
    titre("6. Phase 3 : Calibration des seuils de décision (SLA : Recall ≥ 90%, FPR ≤ 0.5%)")
    seuils = {}
    for nom, p_val in probas_val_dict.items():
        s = calibrer_seuil_bancaire(yv, p_val, recall_cible=SLA_RECALL_MIN, fpr_max=SLA_FPR_MAX)
        seuils[nom] = s
        print(f"  Seuil calibré — {nom:<32} : {s:.4f}")

    # ── ÉTAPE 7 : Entraînement Final sur 100% Train & Benchmark TEST ───────────
    titre("7. Phase 4 : Entraînement final (100% Train) et Évaluation TEST")

    modeles_finaux = {
        "Régression Logistique": make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed
        ),
        "XGBoost (GPU CUDA, Optimisé)": XGBClassifier(**best_xgb_params, **xgb_base),
        "LightGBM": LGBMClassifier(
            n_estimators=350, max_depth=6, learning_rate=0.05,
            num_leaves=63, class_weight="balanced",
            subsample=0.85, colsample_bytree=0.8,
            n_jobs=-1, random_state=args.seed, verbose=-1
        ),
        "MLP PyTorch": TorchMLP(device_torch, seed=args.seed),
    }

    probas_test = {}
    lignes_test = []
    latences_ms = {}

    for nom, mod in modeles_finaux.items():
        t0_fit = time.time()
        if isinstance(mod, TorchMLP):
            mod.fit(X_train.values, y_train.values, epochs=mlp_epochs)
            p_test = mod.predict_proba(X_test.values)[:, 1]
            lat = mesurer_latence(mod, X_test.values)
        else:
            mod.fit(X_train, y_train)
            p_test = mod.predict_proba(X_test)[:, 1]
            lat = mesurer_latence(mod, X_test)

        t_fit = time.time() - t0_fit
        probas_test[nom] = p_test
        latences_ms[nom] = lat

        seuil_m = seuils.get(nom, 0.5)
        res = evaluer_modele(nom, y_test, p_test, seuil=seuil_m)
        res["Train_s"] = round(t_fit, 2)
        res["Latence_ms"] = round(lat, 2)
        res["SLA_Latence"] = "✅" if lat < SLA_LATENCY_MS else "❌"
        lignes_test.append(res)

        print(f"  ✅ {nom:<28} | Train: {t_fit:5.1f}s | Latence: {lat:5.2f}ms | "
              f"Recall: {res['Recall']:.3f} {res['SLA_Recall']} | FPR: {res['FPR']:.4f} {res['SLA_FPR']}")

    tab_test = pd.DataFrame(lignes_test).sort_values("PR-AUC", ascending=False)

    # ── ÉTAPE 8 : Analyse Comparative & Sélection du Meilleur Modèle ──────────
    titre("8. Synthèse Comparative sur le Jeu de Test (Contraintes SLA)")
    cols_vue = [
        "Modèle", "PR-AUC", "ROC-AUC", "Recall", "Precision", "F1",
        "FPR", "Latence_ms", "SLA_Recall", "SLA_FPR", "SLA_Latence"
    ]
    print(tab_test[cols_vue].to_string(index=False))

    # Algorithme de décision multicritère pour le choix du champion
    # 1. Filtre sur tous les SLAs stricts
    candidats_conformes = tab_test[
        (tab_test["SLA_Recall"] == "✅") &
        (tab_test["SLA_FPR"] == "✅") &
        (tab_test["SLA_Latence"] == "✅")
    ]

    if not candidats_conformes.empty:
        # Champion = plus haut PR-AUC parmi les conformes
        meilleur_nom = candidats_conformes.sort_values("PR-AUC", ascending=False).iloc[0]["Modèle"]
        print(f"\n  🏆 MODÈLE CHAMPION SÉLECTIONNÉ (100% SLA RESPECTÉS) : {meilleur_nom}")
    else:
        # Si compromis nécessaire : priorité Recall >= 90% et Latence < 80ms, minimiser FPR
        candidats_rec = tab_test[(tab_test["SLA_Recall"] == "✅") & (tab_test["SLA_Latence"] == "✅")]
        if not candidats_rec.empty:
            meilleur_nom = candidats_rec.sort_values("FPR", ascending=True).iloc[0]["Modèle"]
        else:
            meilleur_nom = tab_test.iloc[0]["Modèle"]
        print(f"\n  🏆 MODÈLE SÉLECTIONNÉ (Compromis optimal) : {meilleur_nom}")

    champion_row = tab_test[tab_test["Modèle"] == meilleur_nom].iloc[0]
    cm_best = confusion_matrix(
        y_test,
        (probas_test[meilleur_nom] >= seuils[meilleur_nom]).astype(int)
    )
    tn_b, fp_b, fn_b, tp_b = cm_best.ravel()

    print(f"\n  Performances du champion ({meilleur_nom}) :")
    print(f"    - Fraudes réelles dans le test : {tp_b + fn_b}")
    print(f"    - Fraudes interceptées (TP)    : {tp_b} ({tp_b / (tp_b + fn_b) * 100:.1f} %)")
    print(f"    - Fraudes manquées (FN)        : {fn_b}")
    print(f"    - Fausses alertes (FP)         : {fp_b} (FPR: {fp_b / (fp_b + tn_b) * 100:.3f} %)")
    print(f"    - Latence d'inférence          : {champion_row['Latence_ms']} ms (SLA < 80 ms)")

    # ── ÉTAPE 9 : Sauvegarde des Graphiques et Artefacts ───────────────────────
    titre("9. Génération des graphiques & rapports")
    plot_pr_curves(probas_test, y_test, args.out_dir)
    plot_roc_curves(probas_test, y_test, args.out_dir)
    plot_latences(latences_ms, args.out_dir)

    for nom in tab_test["Modèle"]:
        cm_m = confusion_matrix(y_test, (probas_test[nom] >= seuils[nom]).astype(int))
        plot_confusion(cm_m, nom, args.out_dir)

    xgb_champion = modeles_finaux["XGBoost (GPU CUDA, Optimisé)"]
    plot_importance(xgb_champion, features, args.out_dir)

    # Explicabilité SHAP
    try:
        import shap
        ech = X_test.sample(min(1500, len(X_test)), random_state=args.seed)
        explainer = shap.TreeExplainer(xgb_champion)
        shap_vals = explainer.shap_values(ech)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_vals, ech, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "shap_summary.png"), dpi=200)
        plt.close()
        print("  ✅ Explicabilité SHAP générée (shap_summary.png)")
    except Exception as e:
        print(f"  ℹ️ SHAP ignoré ({e})")

    # ── ÉTAPE 10 : Sauvegarde des Modèles & Configuration ─────────────────────
    titre("10. Sérialisation des modèles & configuration MLOps")

    # Export CSVs
    tab_test.to_csv(os.path.join(args.out_dir, "comparaison_modeles_test.csv"), index=False)
    tab_val_defaut.to_csv(os.path.join(args.out_dir, "comparaison_validation.csv"), index=False)

    # Sauvegarde XGBoost (Format natif JSON & Joblib)
    xgb_json = os.path.join(args.out_dir, "xgboost_fraude.json")
    xgb_champion.save_model(xgb_json)
    joblib.dump(xgb_champion, os.path.join(args.out_dir, "xgboost_fraude.joblib"))
    print(f"  ✅ XGBoost exporté : {xgb_json}")

    # Sauvegarde LightGBM
    lgb_mod = modeles_finaux["LightGBM"]
    lgb_mod.booster_.save_model(os.path.join(args.out_dir, "lightgbm_fraude.txt"))
    joblib.dump(lgb_mod, os.path.join(args.out_dir, "lightgbm_fraude.joblib"))
    print("  ✅ LightGBM exporté")

    # Sauvegarde Random Forest & Logistique
    joblib.dump(modeles_finaux["Random Forest"], os.path.join(args.out_dir, "random_forest.joblib"))
    joblib.dump(modeles_finaux["Régression Logistique"], os.path.join(args.out_dir, "regression_logistique.joblib"))

    # Sauvegarde MLP PyTorch
    mlp_mod = modeles_finaux["MLP PyTorch"]
    torch.save(mlp_mod.model.state_dict(), os.path.join(args.out_dir, "mlp_fraude.pt"))
    joblib.dump(mlp_mod.scaler, os.path.join(args.out_dir, "mlp_scaler.joblib"))
    print("  ✅ MLP PyTorch exporté")

    # Fichier de métadonnées et traçabilité Jira SID-1
    total_time = round(time.time() - t_start, 1)
    config_dict = {
        "projet": "SIDPEF",
        "ticket_jira": "SID-1",
        "date_execution": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "gpu_cuda_xgboost": xgb_cuda,
            "gpu_cuda_pytorch": torch_cuda,
            "device_torch": str(device_torch),
            "device_xgb": device_xgb,
        },
        "features": features,
        "seuils_calibres": seuils,
        "modele_champion": meilleur_nom,
        "champion_metriques": champion_row.to_dict(),
        "xgboost_best_hyperparams": best_xgb_params,
        "mlp_epochs_retenues": mlp_epochs,
        "sla_objectifs": {
            "latence_max_ms": SLA_LATENCY_MS,
            "recall_min": SLA_RECALL_MIN,
            "fpr_max": SLA_FPR_MAX,
        },
        "resultats_complets_test": tab_test.to_dict(orient="records"),
        "duree_totale_secondes": total_time,
    }

    config_path = os.path.join(args.out_dir, "config_modele.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"  ✅ Métadonnées & configuration MLOps : {config_path}")

    titre("RÉSUMÉ FINAL DE LA MISSION")
    print(f"  🏆 Modèle Champion      : {meilleur_nom}")
    print(f"  🎯 Recall Détection     : {champion_row['Recall'] * 100:.2f} %  {champion_row['SLA_Recall']}")
    print(f"  🛡️ Taux Faux Positifs   : {champion_row['FPR'] * 100:.3f} %  {champion_row['SLA_FPR']}")
    print(f"  ⚡ Latence Unitaire     : {champion_row['Latence_ms']} ms  {champion_row['SLA_Latence']}")
    print(f"  📈 PR-AUC               : {champion_row['PR-AUC']:.4f}")
    print(f"  ⏱️ Temps d'exécution    : {total_time:.1f} s")
    print(f"  📂 Tous les artefacts   : {os.path.abspath(args.out_dir)}")
    print(f"\n{BANNER}")


if __name__ == "__main__":
    main()
