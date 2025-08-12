# modules/vol_core.py  # Nom du fichier et emplacement dans le projet

import pandas as pd  # Importer la bibliothèque pandas pour la manipulation de données
import numpy as np  # Importer NumPy pour les opérations numériques
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator  # Importer les méthodes d'interpolation
from pathlib import Path  # Importer Path pour gérer les chemins de fichiers
from typing import Callable, Literal, Optional, Union  # Importer les types pour l'annotation
import json  # Importer json pour la sérialisation des données
import datetime as dt  # Importer datetime pour manipuler les dates
import yaml  # Importer PyYAML pour lire les fichiers YAML
from pathlib import Path  # Pour gérer les chemins de fichiers de manière portable

# === Chemin vers le fichier de configuration YAML ===
CONFIG_PATH = Path(__file__).parent.parent / "config" / "vol_settings.yaml"  # Remonte d'un dossier (de /modules vers racine) et va dans /config

# === Fonction pour charger la configuration YAML ===
def load_config():
    """Charge le fichier vol_settings.yaml et retourne un dictionnaire Python."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:  # Ouvrir le fichier YAML en lecture
        config = yaml.safe_load(f)  # Charger le contenu YAML dans un dictionnaire Python
    return config  # Retourner le dictionnaire

# Charger la configuration une seule fois au démarrage
settings = load_config()  # Dictionnaire contenant toutes les valeurs définies dans vol_settings.yaml


# -----------------------------
# LECTURE ET NORMALISATION
# -----------------------------

def read_vol_file(file_like) -> pd.DataFrame:  # Définir la fonction de lecture de fichier de volatilité
    name = getattr(file_like, "name", "").lower()  # Récupérer le nom du fichier en minuscules
    if name.endswith(".xlsx") or name.endswith(".xls"):  # Vérifier si le fichier est Excel
        return pd.read_excel(file_like)  # Lire le fichier Excel et retourner un DataFrame
    else:  # Sinon, traiter comme fichier texte
        content = file_like.read().decode("utf-8") if hasattr(file_like, "read") else str(file_like)  # Lire et décoder le contenu
        for sep in [",", ";", "\t", "|"]:  # Boucle sur les séparateurs possibles
            try:
                df = pd.read_csv(pd.compat.StringIO(content), sep=sep)  # Essayer de lire avec ce séparateur
                if df.shape[1] > 1:  # Si le fichier contient plus d'une colonne
                    return df  # Retourner le DataFrame
            except Exception:
                continue  # Ignorer les erreurs et tester le séparateur suivant
        return pd.read_csv(pd.compat.StringIO(content), engine="python")  # Fallback lecture CSV générique

def normalize_tenor_col(s: pd.Series) -> pd.Series:  # Normaliser une colonne Tenor/Maturity en années
    today = dt.date.today()  # Date actuelle
    out = []  # Liste pour stocker les valeurs converties
    for v in s.astype(str):  # Boucler sur chaque valeur convertie en string
        v_strip = v.strip().upper()  # Nettoyer et mettre en majuscules
        if "-" in v_strip and len(v_strip.split("-")) == 3:  # Si format date YYYY-MM-DD
            try:
                y, m, d = map(int, v_strip.split("-"))  # Extraire année, mois, jour
                dd = dt.date(y, m, d)  # Créer un objet date
                out.append((dd - today).days / 365.25)  # Convertir en années et ajouter à la liste
                continue  # Passer à la prochaine valeur
            except Exception:
                pass  # Ignorer si erreur
        if v_strip.endswith("Y"):  # Si format X années
            out.append(float(v_strip[:-1]))  # Convertir en float et ajouter
        elif v_strip.endswith("M"):  # Si format X mois
            out.append(float(v_strip[:-1]) / 12.0)  # Convertir en années et ajouter
        else:  # Sinon essayer conversion directe
            try:
                out.append(float(v_strip))  # Conversion directe
            except Exception:
                out.append(np.nan)  # Valeur manquante si échec
    return pd.Series(out)  # Retourner la série normalisée

def normalize_surface(df: pd.DataFrame) -> pd.DataFrame:  # Normaliser une surface wide en format long
    dfw = df.copy()  # Copier le DataFrame
    dfw.columns = [str(c).strip() for c in dfw.columns]  # Nettoyer les noms de colonnes
    mat_col = dfw.columns[0]  # Nom de la colonne des maturités
    mats_years = normalize_tenor_col(dfw[mat_col])  # Conversion en années
    dfw.insert(0, "TenorY", mats_years)  # Insérer la colonne TenorY en première position
    long = dfw.melt(id_vars=[mat_col, "TenorY"], var_name="Strike", value_name="Vol")  # Transformer en format long
    long["StrikeParsed"] = pd.to_numeric(long["Strike"].str.replace("%", "").str.replace(",", "."), errors="ignore")  # Parser le strike
    long["VolPct"] = pd.to_numeric(long["Vol"].astype(str).str.replace("%", "").str.replace(",", "."), errors="coerce")  # Parser la vol
    return long.dropna(subset=["TenorY", "VolPct"])  # Supprimer les lignes incomplètes

# -----------------------------
# INTERPOLATION
# -----------------------------

def interp_curve(tenors, vols, method: Literal["linear","spline","pchip"]="linear", clip: bool=True) -> Callable:  # Interpolation d'une courbe
    tenors = np.asarray(tenors, dtype=float)  # Conversion en array float
    vols = np.asarray(vols, dtype=float)  # Conversion en array float
    order = np.argsort(tenors)  # Tri des maturités
    tenors, vols = tenors[order], vols[order]  # Réordonner
    mask = ~np.isnan(tenors) & ~np.isnan(vols)  # Masque des valeurs valides
    tenors, vols = tenors[mask], vols[mask]  # Appliquer le masque

    if len(tenors) < 2:  # Si moins de deux points, erreur
        raise ValueError("Pas assez de points pour interpoler")  # Lever une erreur

    if method == "spline" and len(tenors) >= 3:  # Si spline et au moins 3 points
        base = CubicSpline(tenors, vols, bc_type="natural", extrapolate=True)  # Interpolation spline cubique
    elif method == "pchip":  # Si PCHIP
        base = PchipInterpolator(tenors, vols, extrapolate=True)  # Interpolation PCHIP
    else:  # Sinon linéaire
        base = interp1d(tenors, vols, kind="linear", fill_value="extrapolate", assume_sorted=True)  # Interpolation linéaire

    t_min, t_max = tenors.min(), tenors.max()  # Min et max des maturités
    v_min, v_max = vols.min(), vols.max()  # Min et max des vols

    def f(x):  # Fonction interpolée
        y = base(x)  # Calcul des valeurs
        if clip:  # Si clip activé
            y = np.where(x < t_min, v_min, y)  # Clip en-dessous
            y = np.where(x > t_max, v_max, y)  # Clip au-dessus
        return y  # Retourner les valeurs
    return f  # Retourner la fonction interpolante

# -----------------------------
# CALIBRATION SABR (simplifiée ici)
# -----------------------------

def calibrate_sabr(smile_df: pd.DataFrame, beta: float=0.5, bounds: Optional[dict]=None) -> dict:  # Calibration SABR simplifiée
    return {  # Retourner un dict simulé
        "alpha": 0.05,  # Paramètre alpha
        "beta": beta,  # Paramètre beta choisi
        "rho": -0.1,  # Paramètre rho
        "nu": 0.4,  # Paramètre nu
        "rmse": 0.002  # Erreur quadratique moyenne simulée
    }

# -----------------------------
# CONVERSIONS DE VOLATILITÉ
# -----------------------------

def convert_vol_black_to_bachelier(vol_black: float, fwd: float, strike: float, tenor: float) -> float:  # Conversion Black → Bachelier
    return vol_black * fwd / 100.0  # Formule simplifiée

def convert_vol_bachelier_to_black(vol_bach: float, fwd: float, strike: float, tenor: float) -> float:  # Conversion Bachelier → Black
    return vol_bach * 100.0 / fwd  # Formule simplifiée

# -----------------------------
# STRESS TESTS
# -----------------------------

def stress_surface(df_long: pd.DataFrame, mode: Literal["shift","percent"], value: float) -> pd.DataFrame:  # Appliquer stress sur surface
    stressed = df_long.copy()  # Copier le DataFrame
    if mode == "shift":  # Si mode shift
        stressed["VolPct"] = stressed["VolPct"] + value  # Ajouter en points
    elif mode == "percent":  # Si mode pourcentage
        stressed["VolPct"] = stressed["VolPct"] * (1 + value/100)  # Multiplier
    return stressed  # Retourner la surface modifiée

# -----------------------------
# COMPARAISONS
# -----------------------------

def compare_surfaces(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:  # Comparer deux surfaces
    merged = pd.merge(df_a, df_b, on=["TenorY","StrikeParsed"], suffixes=("_A","_B"))  # Fusionner sur TenorY et StrikeParsed
    merged["Diff"] = merged["VolPct_B"] - merged["VolPct_A"]  # Calculer la différence
    return merged  # Retourner le DataFrame

# -----------------------------
# EXPORTS
# -----------------------------

def export_excel(payload: dict, path: Union[str, Path]) -> None:  # Exporter vers Excel multi-onglets
    with pd.ExcelWriter(path) as writer:  # Créer un writer Excel
        for sheet_name, df in payload.items():  # Boucler sur chaque onglet
            df.to_excel(writer, sheet_name=sheet_name, index=False)  # Écrire le DataFrame
