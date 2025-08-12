# pages/07.3_Volatilite.py  # Nom du fichier et emplacement dans le projet

import io  # Importer io pour gérer les flux en mémoire
import os  # Importer os pour variables d'environnement si nécessaire
import datetime as dt  # Importer datetime pour manipuler les dates
import numpy as np  # Importer NumPy pour opérations numériques
import pandas as pd  # Importer pandas pour manipulations tabulaires
import streamlit as st  # Importer Streamlit pour l'interface utilisateur
import plotly.graph_objects as go  # Importer Plotly Graph Objects pour les graphiques interactifs
import plotly.express as px  # Importer Plotly Express pour graphes rapides
from pathlib import Path  # Importer Path pour chemins
from typing import Optional  # Importer Optional pour annotations
from modules.vol_core import settings  # Importer la config YAML déjà chargée

# Exemple d'utilisation dans Streamlit
import streamlit as st

st.title("Analyse de volatilité")

# Afficher les maturités par défaut depuis le YAML
st.write("Maturités par défaut :", settings["default_maturities"])

# Utiliser la config pour afficher un message conditionnel
if settings["data_providers"]["yahoo_finance"]:
    st.success("Yahoo Finance activé dans la configuration.")
else:
    st.warning("Yahoo Finance désactivé dans la configuration.")


# Imports internes depuis les modules métier  # Indiquer clairement les dépendances internes
from modules.vol_core import (  # Importer les fonctions cœur
    read_vol_file,  # Lecture de fichiers de volatilité
    normalize_surface,  # Conversion surface wide→long
    normalize_tenor_col,  # Conversion tenor→années
    interp_curve,  # Interpolateur de courbe 1D
    calibrate_sabr,  # Calibration SABR (stub simplifié)
    convert_vol_black_to_bachelier,  # Conversion Black→Bachelier
    convert_vol_bachelier_to_black,  # Conversion Bachelier→Black
    stress_surface,  # Application de stress sur surface
    compare_surfaces,  # Comparaison de deux surfaces
    export_excel  # Export Excel multi-onglets
)  # Fin des imports vol_core

from modules.vol_providers import (  # Importer les providers et gestion identifiants
    fetch_surface,  # Interface unifiée pour récupérer une surface
    save_credential,  # Sauvegarder un identifiant
    load_credential,  # Charger un identifiant
    delete_credential  # Supprimer un identifiant
)  # Fin des imports vol_providers

# -----------------------------
# CONFIGURATION GÉNÉRALE UI
# -----------------------------

st.set_page_config(page_title="Volatilité", page_icon=None, layout="wide")  # Configurer la page Streamlit
st.title("Volatilité — Courbes, Surfaces, Calibration et Stress")  # Titre principal de l’onglet

# Initialiser le state si nécessaire  # Garantir l'existence des clés session_state
for key, default in [
    ("data_raw", None),  # Données brutes importées ou fetchées
    ("data_long", None),  # Surface en format long normalisé
    ("data_long_ref", None),  # Surface de référence pour comparaisons
    ("scenarios", {}),  # Dictionnaire de scénarios de stress nommés
    ("active_provider", "file"),  # Provider actif par défaut
    ("sabr_params", {}),  # Paramètres SABR calibrés par maturité
    ("message_log", []),  # Journal de messages utilisateur
]:  # Boucler sur les paires clé/valeur par défaut
    if key not in st.session_state:  # Si la clé n'existe pas encore
        st.session_state[key] = default  # L'initialiser avec la valeur par défaut

# -----------------------------
# SIDEBAR : PROVIDERS & SOURCES
# -----------------------------

with st.sidebar:  # Ouvrir la barre latérale
    st.header("Sources de données")  # En-tête de la section
    provider = st.selectbox(  # Sélecteur de provider
        "Provider actif",  # Label du champ
        ["file", "yahoo", "bloomberg_desktop", "bloomberg_hapi"],  # Options possibles
        index=["file", "yahoo", "bloomberg_desktop", "bloomberg_hapi"].index(st.session_state["active_provider"])  # Positionner sur le state
    )  # Fin du selectbox provider
    st.session_state["active_provider"] = provider  # Mettre à jour le provider actif dans le state

    st.divider()  # Séparateur visuel

    if provider == "file":  # Si source fichiers
        st.caption("Import manuel de fichiers CSV/Excel.")  # Indiquer le mode d’import
        file = st.file_uploader("Charger un fichier de volatilité", type=["csv", "xlsx", "xls"])  # Uploader de fichier
        if file is not None:  # Si un fichier est choisi
            try:
                df = read_vol_file(file)  # Lire le fichier via helper
                st.session_state["data_raw"] = df  # Stocker les données brutes
                st.success(f"Fichier chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")  # Message de succès
            except Exception as e:  # Capturer toute erreur de lecture
                st.error(f"Erreur de lecture du fichier : {e}")  # Afficher l’erreur

    elif provider == "yahoo":  # Si source Yahoo Finance
        ticker = st.text_input("Ticker Yahoo (ex: ^VIX, AAPL)", value="AAPL")  # Champ pour saisir un ticker
        if st.button("Récupérer options / vol"):  # Bouton pour lancer la récupération
            try:
                df = fetch_surface("yahoo", ticker=ticker)  # Appeler l’interface provider
                st.session_state["data_raw"] = df  # Stocker les données brutes
                st.success(f"Données Yahoo chargées : {df.shape[0]} lignes")  # Message de succès
            except Exception as e:  # En cas d’erreur
                st.error(f"Erreur Yahoo : {e}")  # Afficher l’erreur

    elif provider == "bloomberg_desktop":  # Si Bloomberg Desktop API
        st.caption("Nécessite un Terminal Bloomberg ouvert (BLPAPI).")  # Rappel du prérequis
        bb_ticker = st.text_input("Ticker Bloomberg (ex: SPX Index)", value="SPX Index")  # Champ ticker
        bb_fields = st.text_input("Champs Bloomberg (liste séparée par des virgules)", value="OPT_IMPLIED_VOLATILITY")  # Champs demandés
        if st.button("Tester connexion & récupérer"):  # Bouton d’action
            try:
                fields = [f.strip() for f in bb_fields.split(",") if f.strip()]  # Parser la liste des champs
                df = fetch_surface("bloomberg_desktop", ticker=bb_ticker, fields=fields)  # Appeler provider
                if df is None or df.empty:  # Vérifier si résultat exploitable
                    st.warning("Réponse vide ou non parsée. Implémentation extraction à compléter côté Desktop API.")  # Avertir si vide
                st.session_state["data_raw"] = df  # Stocker éventuellement
                st.info("Prototype Desktop API : extraction détaillée à compléter selon vos champs.")  # Note informative
            except Exception as e:  # En cas d’erreur
                st.error(f"Erreur Bloomberg Desktop : {e}")  # Afficher l’erreur

    elif provider == "bloomberg_hapi":  # Si Bloomberg HAPI OAuth
        st.caption("Authentification OAuth2 client credentials requise.")  # Rappel de l’authentification
        client_id = st.text_input("Client ID", value=load_credential("bloomberg_hapi", "client_id") or "", type="password")  # Champ client_id masqué
        client_secret = st.text_input("Client Secret", value=load_credential("bloomberg_hapi", "client_secret") or "", type="password")  # Champ secret masqué
        remember = st.checkbox("Se souvenir sur cet appareil", value=True)  # Choix de persistance
        if st.button("Enregistrer identifiants"):  # Bouton pour sauvegarder
            try:
                save_credential("bloomberg_hapi", "client_id", client_id, remember=remember)  # Sauvegarder client_id
                save_credential("bloomberg_hapi", "client_secret", client_secret, remember=remember)  # Sauvegarder secret
                st.success("Identifiants enregistrés.")  # Message de succès
            except Exception as e:  # Erreur éventuelle
                st.error(f"Erreur enregistrement identifiants : {e}")  # Afficher l’erreur
        if st.button("Oublier identifiants"):  # Bouton pour supprimer
            delete_credential("bloomberg_hapi", "client_id")  # Supprimer client_id
            delete_credential("bloomberg_hapi", "client_secret")  # Supprimer secret
            st.info("Identifiants supprimés.")  # Confirmer suppression

        endpoint = st.text_input("Endpoint HAPI (path relatif)", value="marketdata/volatility")  # Endpoint API
        params_raw = st.text_area("Paramètres JSON", value='{"ticker": "AAPL"}')  # Paramètres JSON bruts
        if st.button("Appeler HAPI"):  # Bouton d’appel
            try:
                import json as _json  # Import local json
                params = _json.loads(params_raw)  # Parser le JSON
                resp = fetch_surface("bloomberg_hapi", endpoint=endpoint, params=params)  # Appeler provider HAPI
                st.session_state["data_raw"] = pd.DataFrame(resp) if isinstance(resp, dict) else resp  # Convertir en DataFrame si possible
                st.success("Réponse HAPI récupérée.")  # Message de succès
            except Exception as e:  # En cas d’échec
                st.error(f"Erreur HAPI : {e}")  # Afficher l’erreur

# -----------------------------
# PANNEAU DONNÉES & NORMALISATION
# -----------------------------

st.subheader("Données")  # Sous-titre pour la section données
colA, colB = st.columns([2, 1])  # Créer deux colonnes pour mise en page

with colA:  # Colonne A pour l’aperçu
    if st.session_state["data_raw"] is not None:  # Si des données sont chargées
        st.write("Aperçu des données brutes")  # Titre de l’aperçu
        st.dataframe(st.session_state["data_raw"].head(50), use_container_width=True)  # Afficher un échantillon
    else:  # Sinon
        st.info("Chargez une source de données pour commencer.")  # Message d’information

with colB:  # Colonne B pour la normalisation
    st.write("Normalisation & format")  # Titre de la section
    mode = st.radio("Type de données", options=["Courbe (Tenor, Vol)", "Surface (maturités × strikes)"], index=1)  # Choisir structure
    if st.session_state["data_raw"] is not None:  # Si données présentes
        if mode.startswith("Courbe"):  # Si données de courbe
            df = st.session_state["data_raw"].copy()  # Copier le DataFrame
            if df.shape[1] < 2:  # Vérifier qu’il y a au moins 2 colonnes
                st.error("Courbe attend au moins 2 colonnes : Tenor, Vol")  # Message d’erreur
            else:  # Sinon on poursuit
                tenors = normalize_tenor_col(df.iloc[:, 0])  # Normaliser la première colonne en années
                vols = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # Forcer la seconde en numérique
                long_df = pd.DataFrame({"TenorY": tenors, "StrikeParsed": 0.0, "VolPct": vols})  # Construire un long minimal (Strike=0 pour courbe ATM)
                long_df = long_df.dropna(subset=["TenorY", "VolPct"])  # Nettoyer les NaN
                st.session_state["data_long"] = long_df  # Stocker la version normalisée
                st.success("Courbe normalisée.")  # Confirmer
        else:  # Sinon surface
            try:
                long_df = normalize_surface(st.session_state["data_raw"])  # Convertir en long format
                st.session_state["data_long"] = long_df  # Stocker
                st.success("Surface normalisée (format long).")  # Confirmer
            except Exception as e:  # En cas d’erreur
                st.error(f"Erreur de normalisation : {e}")  # Afficher l’erreur

# -----------------------------
# PARAMÈTRES MODÈLES & OPTIONS
# -----------------------------

st.subheader("Paramètres de modèle")  # Sous-titre de section
c1, c2, c3, c4 = st.columns(4)  # Quatre colonnes pour les paramètres

with c1:  # Colonne 1
    interp_kind = st.selectbox("Interpolation courbe", ["linear", "spline", "pchip"], index=2)  # Choisir méthode interp 1D
with c2:  # Colonne 2
    extrap_clip = st.checkbox("Extrapolation bornée (clip)", value=True)  # Option pour clipper l’extrapolation
with c3:  # Colonne 3
    conv_choice = st.selectbox("Convention volatilité", ["Black (lognormale)", "Bachelier (normale)"], index=0)  # Choisir convention
with c4:  # Colonne 4
    sabr_beta = st.number_input("SABR β", min_value=0.0, max_value=1.0, value=0.5, step=0.1)  # Choisir β pour SABR

# -----------------------------
# VISUALISATIONS PRINCIPALES
# -----------------------------

st.subheader("Visualisations")  # Sous-titre pour la section visuelle
tabs = st.tabs(["Courbe", "Smiles par maturité", "Surface 3D", "Heatmap Δvol", "Comparaison"])  # Créer des onglets

# ---- Onglet Courbe ----
with tabs[0]:  # Onglet Courbe
    if st.session_state["data_long"] is None:  # Vérifier que la surface normalisée existe
        st.info("Normalisez des données pour afficher la courbe.")  # Message d’information
    else:  # Sinon on peut tracer
        df_long = st.session_state["data_long"]  # Récupérer le long format
        curve_df = (df_long.groupby("TenorY", as_index=False)["VolPct"].median().sort_values("TenorY"))  # Construire une courbe ATM par médiane
        try:
            f = interp_curve(curve_df["TenorY"].values, curve_df["VolPct"].values, method=interp_kind, clip=extrap_clip)  # Obtenir l’interpolateur
            x_grid = np.linspace(max(0.01, curve_df["TenorY"].min()), curve_df["TenorY"].max(), 200)  # Grille de maturités
            y_fit = f(x_grid)  # Vol interpolée
        except Exception as e:  # Si l’interpolation échoue
            st.error(f"Interpolation impossible : {e}")  # Afficher l’erreur
            y_fit = None  # Nullifier la courbe

        colC1, colC2 = st.columns([3, 1])  # Colonnes pour graphe et picker
        with colC1:  # Graphe à gauche
            fig = go.Figure()  # Créer la figure
            fig.add_trace(go.Scatter(x=curve_df["TenorY"], y=curve_df["VolPct"], mode="markers", name="Points"))  # Ajouter points
            if y_fit is not None:  # Si fit disponible
                fig.add_trace(go.Scatter(x=x_grid, y=y_fit, mode="lines", name=f"Interpolée ({interp_kind})"))  # Tracer la courbe
            fig.update_layout(xaxis_title="Maturité (années)", yaxis_title="Vol (%)", height=420, legend=dict(orientation="h"))  # Configurer axes et layout
            st.plotly_chart(fig, use_container_width=True)  # Afficher le graphique

        with colC2:  # Panneau de sélection à droite
            tenor_pick = st.number_input("Maturité cible (années)", value=float(np.round(curve_df["TenorY"].median(), 2)))  # Saisie d’une maturité
            if y_fit is not None:  # Si on peut évaluer la courbe
                vol_pick = float(f(tenor_pick))  # Calculer la vol interpolée à la maturité choisie
                st.metric("Vol interpolée (%)", f"{vol_pick:.2f}")  # Afficher la vol
            else:  # Sinon
                st.metric("Vol interpolée (%)", "NA")  # Indiquer indisponible

# ---- Onglet Smiles ----
with tabs[1]:  # Onglet Smiles
    if st.session_state["data_long"] is None:  # Vérifier données
        st.info("Normalisez des données pour afficher les smiles.")  # Message
    else:  # Sinon
        df_long = st.session_state["data_long"].copy()  # Copier le long format
        mats = np.sort(df_long["TenorY"].dropna().unique())  # Extraire les maturités uniques
        if len(mats) == 0:  # Si aucune maturité
            st.warning("Aucune maturité détectée.")  # Avertissement
        else:  # Sinon
            show_mats = st.multiselect("Maturités à afficher", options=list(mats), default=list(mats[:min(5, len(mats))]))  # Choisir sous-ensemble de maturités
            if len(show_mats) == 0:  # Si rien sélectionné
                st.info("Sélectionnez au moins une maturité.")  # Indiquer
            else:  # Sinon tracer
                fig_s = go.Figure()  # Créer figure smiles
                for m in show_mats:  # Boucler sur maturités sélectionnées
                    sl = df_long[df_long["TenorY"] == m].copy()  # Slice par maturité
                    sl = sl[pd.to_numeric(sl["StrikeParsed"], errors="coerce").notna()]  # Garder strikes numériques
                    sl = sl.sort_values("StrikeParsed")  # Trier par strike
                    if sl.empty:  # Si vide
                        continue  # Passer
                    fig_s.add_trace(go.Scatter(x=sl["StrikeParsed"], y=sl["VolPct"], mode="lines+markers", name=f"{m:.2f}Y"))  # Tracer le smile
                fig_s.update_layout(xaxis_title="Strike / Moneyness", yaxis_title="Vol (%)", height=460, legend=dict(orientation="h"))  # Config layout
                st.plotly_chart(fig_s, use_container_width=True)  # Afficher

# ---- Onglet Surface 3D ----
with tabs[2]:  # Onglet Surface 3D
    if st.session_state["data_long"] is None:  # Vérifier présence données
        st.info("Normalisez des données pour afficher la surface 3D.")  # Message
    else:  # Sinon
        df_long = st.session_state["data_long"].copy()  # Copier
        df_num = df_long[pd.to_numeric(df_long["StrikeParsed"], errors="coerce").notna()].copy()  # Filtrer strikes numériques
        if df_num.empty:  # Si pas de strikes numériques
            st.warning("Pas assez de strikes numériques pour une surface 3D.")  # Avertir
        else:  # Sinon
            mats = np.sort(df_num["TenorY"].unique())  # Maturités uniques
            strikes = np.sort(df_num["StrikeParsed"].unique())  # Strikes uniques
            M, K = np.meshgrid(mats, strikes, indexing="ij")  # Maillage maturité×strike
            Z = np.full_like(M, np.nan, dtype=float)  # Initialiser la grille Z

            for i, m in enumerate(mats):  # Boucler sur les maturités
                sl = df_num[df_num["TenorY"] == m].sort_values("StrikeParsed")  # Slice maturité
                if len(sl) >= 2:  # Besoin de ≥ 2 points pour interp 1D
                    from scipy.interpolate import interp1d as _interp1d  # Import local pour optimiser temps de chargement
                    f1 = _interp1d(sl["StrikeParsed"].values, sl["VolPct"].values, kind="linear", fill_value="extrapolate", assume_sorted=True)  # Interpolateur 1D sur strike
                    Z[i, :] = f1(strikes)  # Remplir la ligne de Z

            fig3d = go.Figure(data=[go.Surface(x=strikes, y=mats, z=Z, showscale=True, name="Vol")])  # Créer la surface 3D
            fig3d.update_layout(scene=dict(xaxis_title="Strike / Moneyness", yaxis_title="Maturité (années)", zaxis_title="Vol (%)"), height=520)  # Configurer la scène
            st.plotly_chart(fig3d, use_container_width=True)  # Afficher

# ---- Onglet Heatmap Δvol ----
with tabs[3]:  # Onglet Heatmap Δvol
    if st.session_state["data_long"] is None:  # Vérifier les données
        st.info("Normalisez des données pour produire une heatmap de différences.")  # Message
    else:  # Sinon
        df_base = st.session_state["data_long"]  # Surface de base
        df_ref = st.session_state["data_long_ref"]  # Surface de référence
        if df_ref is None:  # Si aucune référence
            st.info("Définissez une surface de référence dans la section Stress & Scénarios pour voir Δvol.")  # Indiquer la marche à suivre
        else:  # Sinon calculer les différences
            diff = compare_surfaces(df_ref, df_base)  # Calculer A→B
            diff_num = diff[pd.to_numeric(diff["StrikeParsed"], errors="coerce").notna()].copy()  # Garder strikes numériques
            if diff_num.empty:  # Si vide
                st.warning("Pas de données numériques pour la heatmap.")  # Avertir
            else:  # Sinon construire la matrice pivot
                pivot = diff_num.pivot_table(index="TenorY", columns="StrikeParsed", values="Diff", aggfunc="median")  # Tableau pivot
                fig_hm = px.imshow(pivot.values, aspect="auto", origin="lower",  # Construire la heatmap
                                   x=pivot.columns, y=pivot.index, color_continuous_scale="RdBu", zmin=-np.nanmax(np.abs(pivot.values)), zmax=np.nanmax(np.abs(pivot.values)))  # Échelle symétrique
                fig_hm.update_layout(xaxis_title="Strike / Moneyness", yaxis_title="Maturité (années)", height=520)  # Configurer layout
                st.plotly_chart(fig_hm, use_container_width=True)  # Afficher

# ---- Onglet Comparaison ----
with tabs[4]:  # Onglet Comparaison
    st.write("Comparaison de surfaces ou dates différentes")  # Titre
    colCmp1, colCmp2 = st.columns(2)  # Deux colonnes pour sélection et résultats
    with colCmp1:  # Colonne gauche
        if st.button("Définir la surface actuelle comme référence"):  # Bouton pour fixer la référence
            st.session_state["data_long_ref"] = st.session_state["data_long"]  # Enregistrer la référence
            st.success("Surface de référence définie.")  # Confirm
        if st.session_state["data_long_ref"] is not None:  # Si référence existe
            st.write("Aperçu référence")  # Titre
            st.dataframe(st.session_state["data_long_ref"].head(20), use_container_width=True)  # Aperçu
    with colCmp2:  # Colonne droite
        if st.session_state["data_long"] is not None:  # Si surface courante existe
            st.write("Aperçu surface courante")  # Titre
            st.dataframe(st.session_state["data_long"].head(20), use_container_width=True)  # Aperçu
        if st.session_state["data_long"] is not None and st.session_state["data_long_ref"] is not None:  # Si les deux existent
            res = compare_surfaces(st.session_state["data_long_ref"], st.session_state["data_long"])  # Calculer la diff
            st.write("Statistiques d’écarts")  # Titre
            st.dataframe(res.describe(include="all"), use_container_width=True)  # Décrire les différences

# -----------------------------
# STRESS TESTS & SCÉNARIOS
# -----------------------------

st.subheader("Stress & scénarios")  # Sous-titre section stress
sc_col1, sc_col2, sc_col3 = st.columns([2, 2, 1])  # Trois colonnes pour inputs et actions

with sc_col1:  # Première colonne
    stress_mode = st.selectbox("Mode de stress", ["shift", "percent"], index=0)  # Choisir mode de stress
    stress_value = st.number_input("Valeur de stress", value=10.0, step=1.0)  # Valeur de stress
with sc_col2:  # Deuxième colonne
    stress_scope = st.selectbox("Portée", ["Global", "Par maturité", "Par strike"], index=0)  # Sélection de portée
    scope_value = st.text_input("Filtre (ex: TenorY==1.0 ou StrikeParsed>100)", value="")  # Filtre optionnel
with sc_col3:  # Troisième colonne
    scenario_name = st.text_input("Nom scénario", value=f"stress_{int(dt.datetime.now().timestamp())}")  # Nom de scénario

sc_b1, sc_b2 = st.columns([1, 1])  # Colonnes pour boutons d’action
with sc_b1:  # Colonne bouton 1
    if st.button("Appliquer stress"):  # Bouton appliquer
        if st.session_state["data_long"] is None:  # Vérifier données
            st.error("Aucune surface chargée.")  # Erreur
        else:  # Sinon appliquer
            base = st.session_state["data_long"].copy()  # Copier base
            if stress_scope == "Global":  # Si global
                stressed = stress_surface(base, mode=stress_mode, value=stress_value)  # Appliquer stress
            else:  # Sinon on applique un filtre conditionnel
                try:
                    mask = base.eval(scope_value) if scope_value.strip() else np.ones(len(base), dtype=bool)  # Construire le masque
                    stressed = base.copy()  # Copier base
                    stressed.loc[mask, "VolPct"] = stress_surface(base.loc[mask], mode=stress_mode, value=stress_value)["VolPct"]  # Appliquer sur zone masquée
                except Exception as e:  # En cas d’expression invalide
                    st.error(f"Filtre invalide : {e}")  # Afficher l’erreur
                    stressed = None  # Annuler
            if stressed is not None:  # Si résultat disponible
                st.session_state["scenarios"][scenario_name] = stressed  # Enregistrer le scénario
                st.success(f"Scénario '{scenario_name}' enregistré.")  # Confirmer
with sc_b2:  # Colonne bouton 2
    if st.button("Définir la surface stressée comme courante"):  # Bouton pour activer un scénario
        if st.session_state["scenarios"]:  # Vérifier existence de scénarios
            last_name = list(st.session_state["scenarios"].keys())[-1]  # Prendre le dernier ajouté
            st.session_state["data_long"] = st.session_state["scenarios"][last_name]  # Remplacer la surface courante
            st.success(f"Surface courante remplacée par le scénario '{last_name}'.")  # Confirmer
        else:  # Sinon
            st.info("Aucun scénario disponible.")  # Indiquer

if st.session_state["scenarios"]:  # S’il existe des scénarios
    st.write("Scénarios enregistrés")  # Titre
    names = list(st.session_state["scenarios"].keys())  # Récupérer les noms
    pick = st.selectbox("Choisir un scénario à visualiser", options=names, index=len(names)-1)  # Sélecteur
    st.dataframe(st.session_state["scenarios"][pick].head(20), use_container_width=True)  # Aperçu du scénario

# -----------------------------
# CALIBRATION & CONVERSIONS
# -----------------------------

st.subheader("Calibration et conversions")  # Sous-titre
cal_c1, cal_c2, cal_c3 = st.columns([2, 2, 2])  # Trois colonnes

with cal_c1:  # Colonne calibration
    if st.session_state["data_long"] is None:  # Vérifier données
        st.info("Normalisez des données pour calibrer SABR.")  # Message
    else:  # Sinon calibrer
        df_long = st.session_state["data_long"].copy()  # Copier surface
        mats = np.sort(df_long["TenorY"].dropna().unique())  # Maturités
        mat_for_cal = st.selectbox("Maturité à calibrer (SABR)", options=list(mats), index=0)  # Choisir une maturité
        slice_df = df_long[df_long["TenorY"] == mat_for_cal].copy()  # Slice de smile
        slice_df = slice_df[pd.to_numeric(slice_df["StrikeParsed"], errors="coerce").notna()].sort_values("StrikeParsed")  # Nettoyer strikes
        if slice_df.shape[0] < 3:  # Vérifier nombre de points
            st.warning("Au moins 3 points nécessaires pour une calibration robuste.")  # Avertir
        else:  # Sinon lancer la calibration
            if st.button("Calibrer SABR sur cette maturité"):  # Bouton calibrer
                params = calibrate_sabr(slice_df, beta=sabr_beta, bounds=None)  # Appeler la calibration (stub dans core)
                st.session_state["sabr_params"][mat_for_cal] = params  # Stocker les paramètres
                st.success(f"SABR calibré sur {mat_for_cal:.2f}Y : {params}")  # Afficher résultat
            if mat_for_cal in st.session_state["sabr_params"]:  # Si déjà calibré
                st.write("Paramètres actuels")  # Titre
                st.json(st.session_state["sabr_params"][mat_for_cal])  # Afficher paramètres au format JSON

with cal_c2:  # Colonne conversions
    st.write("Conversion de volatilité")  # Titre
    fwd = st.number_input("Forward (pour conversion)", value=100.0, step=1.0)  # Saisir le forward
    strike = st.number_input("Strike (pour conversion)", value=100.0, step=1.0)  # Saisir le strike
    tenor_years = st.number_input("Maturité (années) pour conversion", value=1.0, step=0.25)  # Saisir la maturité
    vol_input = st.number_input("Vol entrée (%)", value=20.0, step=0.1)  # Saisir la vol d’entrée
    if st.button("Black → Bachelier"):  # Bouton de conversion Black→Bachelier
        vb = convert_vol_black_to_bachelier(vol_input, fwd, strike, tenor_years)  # Conversion
        st.metric("Vol Bachelier (%)", f"{vb:.4f}")  # Afficher résultat
    if st.button("Bachelier → Black"):  # Bouton de conversion Bachelier→Black
        vl = convert_vol_bachelier_to_black(vol_input, fwd, strike, tenor_years)  # Conversion
        st.metric("Vol Black (%)", f"{vl:.4f}")  # Afficher résultat

with cal_c3:  # Colonne utilitaires
    st.write("Utilitaires")  # Titre
    if st.session_state["data_long"] is not None:  # Si surface disponible
        if st.button("Isoler un graphe (mode focus)"):  # Bouton pour montrer un seul graphe
            st.session_state["message_log"].append("Focus demandé sur la dernière visualisation.")  # Journaliser l’action
            st.info("Utilisez les onglets pour n’afficher qu’un seul graphique à la fois.")  # Instruction simple
    st.write("Journal")  # Titre journal
    st.code("\n".join(st.session_state["message_log"][-10:]) or "Aucun événement.", language="text")  # Afficher les logs récents

# -----------------------------
# EXPORTS
# -----------------------------

st.subheader("Exports")  # Sous-titre
ex_c1, ex_c2 = st.columns([1, 1])  # Deux colonnes export
with ex_c1:  # Colonne export CSV
    if st.session_state["data_long"] is not None:  # Vérifier surface
        csv_name = st.text_input("Nom fichier CSV", value="surface_normalisee.csv")  # Nom de fichier CSV
        if st.button("Exporter CSV"):  # Bouton exporter CSV
            csv_bytes = st.session_state["data_long"].to_csv(index=False).encode("utf-8")  # Générer CSV en bytes
            st.download_button("Télécharger", data=csv_bytes, file_name=csv_name, mime="text/csv")  # Bouton de téléchargement
    else:  # Sinon
        st.info("Aucune surface normalisée à exporter en CSV.")  # Message

with ex_c2:  # Colonne export Excel
    xlsx_name = st.text_input("Nom fichier Excel", value="vol_exports.xlsx")  # Nom du fichier Excel
    payload = {}  # Dictionnaire d’onglets à exporter
    if st.session_state["data_raw"] is not None:  # Si brutes présentes
        payload["raw"] = st.session_state["data_raw"]  # Ajouter l’onglet raw
    if st.session_state["data_long"] is not None:  # Si normalisées présentes
        payload["normalized"] = st.session_state["data_long"]  # Ajouter l’onglet normalized
    if st.session_state["data_long_ref"] is not None:  # Si référence présente
        payload["reference"] = st.session_state["data_long_ref"]  # Ajouter l’onglet reference
    for name, df in st.session_state["scenarios"].items():  # Boucler sur scénarios
        payload[f"scenario_{name}"] = df  # Ajouter chaque scénario comme onglet
    if st.button("Exporter Excel multi-onglets"):  # Bouton exporter Excel
        if not payload:  # Si rien à exporter
            st.info("Aucune donnée à exporter.")  # Message
        else:  # Sinon
            tmp_path = Path(st.experimental_get_query_params().get("tmp_dir", ["."])[0]) / xlsx_name  # Déterminer chemin temporaire
            try:
                export_excel(payload, tmp_path)  # Appeler l’export Excel du core
                with open(tmp_path, "rb") as f:  # Ouvrir le fichier binaire
                    st.download_button("Télécharger", data=f, file_name=xlsx_name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")  # Proposer le download
                st.success(f"Export écrit : {tmp_path}")  # Confirmer
            except Exception as e:  # En cas d’échec
                st.error(f"Erreur export Excel : {e}")  # Afficher l’erreur
            finally:
                try:
                    if tmp_path.exists():  # Si le fichier existe
                        tmp_path.unlink()  # Le supprimer après téléchargement
                except Exception:
                    pass  # Ignorer si suppression impossible
