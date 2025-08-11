# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from Modules.Pricing import (
    payoff_barrier_curve_pedagogique,
    barrier_price_batch,
)

st.set_page_config(page_title="Options Barrières (Pro)", page_icon="🧱", layout="wide")
st.title("🧱 Options Barrières — Version pro (MC rapide + graphe pédagogique)")

REQUIRED_COLS = [
    "Option ID",
    "Option Type",
    "Barrier Effect",
    "Barrier Direction",
    "S0",
    "K",
    "B",
    "T",
    "r",
    "sigma",
]
OPTIONAL_COLS = ["Rebate"]

# ---- Helpers lecture fichier ----
def read_table_auto(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".csv"):
        for sep in [",", ";", "\t", "|"]:
            file.seek(0)
            try:
                df = pd.read_csv(file, sep=sep)
                if df.shape[1] >= 3:
                    return df
            except Exception:
                pass
        file.seek(0)
        return pd.read_csv(file, engine="python")
    else:
        st.error("Format non supporté. Dépose un .csv, .xlsx ou .xls.")
        return pd.DataFrame()

# ---- Paramètres Monte Carlo (sidebar) ----
with st.sidebar:
    st.header("Paramètres de calcul (Monte Carlo rapide)")
    monitoring = st.selectbox("Monitoring barrière",
    ["discrete", "continuous"],  # discrete = plus rapide
        index=0,
        help="continuous applique une approx Brownian-Bridge (plus précis, plus lent).",)
    n_steps = st.number_input("Pas par an (n_steps)", min_value=10, value=125, step=5, help="125 ≈ ~2 jours ouvrés.")
    target_stderr = st.number_input("Précision visée (StdErr)", min_value=0.001, value=0.02, step=0.001, format="%.3f",
        help="Arrêt anticipé quand l'écart-type MC est inférieur à ce seuil.")
    chunk_paths = st.number_input("Chemins par bloc", min_value=1000, value=10000, step=1000, help="Blocs simulés par itération.")
    max_paths = st.number_input("Chemins max", min_value=10000, value=120000, step=10000, help="Plafond total de chemins.")

# ---- Saisie manuelle d'une option ----
with st.expander("➕ Saisie rapide d'une option"):
    c1, c2, c3 = st.columns(3)
    with c1:
        oid = st.text_input("Option ID", "OPT-001")
        option_type = st.selectbox("Option Type", ["Call", "Put"])
        barrier_effect = st.selectbox("Barrier Effect", ["Knock-In", "Knock-Out"])
        barrier_direction = st.selectbox("Barrier Direction", ["Up", "Down"])
    with c2:
        S0 = st.number_input("S0", min_value=0.0, value=100.0, step=1.0)
        K = st.number_input("K", min_value=0.0, value=100.0, step=1.0)
        B = st.number_input("B", min_value=0.0, value=120.0, step=1.0)
        rebate = st.number_input("Rebate (optionnel)", min_value=0.0, value=0.0, step=1.0)
    with c3:
        T = st.number_input("T (années)", min_value=0.0, value=1.0, step=0.25)
        r = st.number_input("r", value=0.02, step=0.01, format="%.4f")
        sigma = st.number_input("sigma", value=0.25, step=0.01, format="%.4f")

    if st.button("Ajouter la ligne", type="primary"):
        new = pd.DataFrame([{
                    "Option ID": oid,
                    "Option Type": option_type,
                    "Barrier Effect": barrier_effect,
                    "Barrier Direction": barrier_direction,
                    "S0": S0,
                    "K": K,
                    "B": B,
                    "T": T,
                    "r": r,
                    "sigma": sigma,
                    "Rebate": rebate,}])
        if "bar_df" not in st.session_state or st.session_state.bar_df.empty:
            st.session_state.bar_df = new
        else:
            df = st.session_state.bar_df.copy()
            df = df[df["Option ID"] != oid]  # remplace si même ID
            st.session_state.bar_df = pd.concat([df, new], ignore_index=True)
        st.success(f"Ligne '{oid}' ajoutée/mise à jour.")

# ---- Import CSV/Excel ----
st.subheader("📥 Import CSV/Excel (mix KI/KO)")
st.caption(
    "Colonnes minimales : "
    + ", ".join(REQUIRED_COLS)
    + " — Optionnel : Rebate")

file = st.file_uploader("Dépose un fichier (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])
if file is not None:
    df_in = read_table_auto(file)
    if not df_in.empty:
        missing = [c for c in REQUIRED_COLS if c not in df_in.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
        else:
            keep = REQUIRED_COLS + [c for c in OPTIONAL_COLS if c in df_in.columns]
            df_in = df_in[keep].copy()
            if "Rebate" not in df_in.columns:
                df_in["Rebate"] = 0.0
            # types numériques
            for c in ["S0", "K", "B", "T", "r", "sigma", "Rebate"]:
                df_in[c] = pd.to_numeric(df_in[c], errors="coerce")
            if "bar_df" not in st.session_state or st.session_state.bar_df.empty:
                st.session_state.bar_df = df_in
            else:
                df_old = st.session_state.bar_df.copy()
                df_old = df_old[~df_old["Option ID"].isin(df_in["Option ID"])]
                st.session_state.bar_df = pd.concat([df_old, df_in], ignore_index=True)
            st.success(f"{df_in.shape[0]} lignes importées.")

# ---- Tableau courant ----
st.subheader("🧾 Options chargées")
if "bar_df" not in st.session_state or st.session_state.bar_df.empty:
    st.info("Aucune option pour l’instant. Ajoute une ligne ou importe un fichier.")
    st.stop()

df = st.session_state.bar_df.copy()
st.dataframe(df, use_container_width=True, hide_index=True)

# ---- Sélection pour le graphe ----
ids = df["Option ID"].astype(str).tolist()
if "sel_ids" not in st.session_state:
    st.session_state.sel_ids = ids.copy()

colA, colB, colC = st.columns([1, 1, 4])
with colA:
    if st.button("Tout activer"):
        st.session_state.sel_ids = ids.copy()
with colB:
    if st.button("Tout désactiver"):
        st.session_state.sel_ids = []

selected = st.multiselect(
    "Afficher sur le graphe (payoff pédagogique — non path-dependent) :",
    options=ids,
    default=st.session_state.sel_ids,)

st.session_state.sel_ids = selected

# ---- Graphe des payoff (pédagogiques) ----
st.subheader("📈 Payoff à maturité (pédagogique)")
K_all = df["K"].astype(float).tolist()
B_all = df["B"].astype(float).tolist()
if K_all and B_all:
    xmax = max(max(K_all), max(B_all)) * 1.2
else:
    xmax = 200.0
xmax = float(np.clip(xmax, 50.0, 10000.0))

S_T = np.linspace(0.0, xmax, 1201)
by_id = df.set_index("Option ID")

fig = go.Figure()
for oid in st.session_state.sel_ids:
    row = by_id.loc[oid]

    y = payoff_barrier_curve_pedagogique(
        S_T=S_T,
        option_type=str(row["Option Type"]),
        barrier_effect=str(row["Barrier Effect"]),
        barrier_direction=str(row["Barrier Direction"]),
        K=float(row["K"]),
        B=float(row["B"]),)
    label = (
        f"{oid} | {row['Option Type']} "
        f"{row['Barrier Effect'].replace('Knock-','K')}-{row['Barrier Direction']} "
        f"| K={row['K']}, B={row['B']}")
    fig.add_trace(go.Scatter(x=S_T, y=y, mode="lines", name=label))

fig.update_layout(
    xaxis_title="Prix du sous-jacent à maturité  S_T",
    yaxis_title="Payoff",
    hovermode="x unified",
    legend_title="Options",
    height=520,
    margin=dict(l=10, r=10, t=30, b=10),)
st.plotly_chart(fig, use_container_width=True)

# ---- Pricing path-dependent (Monte Carlo rapide) ----
st.subheader("💰 Pricing path-dependent (Monte Carlo)")
if st.button("Calculer les prix", type="primary"):
    with st.spinner("Calcul Monte Carlo en cours..."):
        priced = barrier_price_batch(
                df,
                monitoring="discrete" if monitoring == "discrete" else "continuous",
                target_stderr=float(target_stderr),
                chunk_paths=int(chunk_paths),
                max_paths=int(max_paths),
                n_steps=int(n_steps),
                seed=None,)


    st.success("Calcul terminé.")
    st.dataframe(priced, use_container_width=True, hide_index=True)

    csv = priced.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger résultats (CSV)",
        data=csv,
        file_name="barrier_pricing_results.csv",
        mime="text/csv")
