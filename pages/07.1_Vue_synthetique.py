import streamlit as st
import pandas as pd
import io
from Modules.MarketDataLive import get_market_snapshot, style_market_table

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

# Titre principal
st.title(" Vue synthétique des marchés")

# Explication en haut de page
st.markdown(
    """
    Cette page affiche un tableau **live** des conditions de marché : niveaux actuels, variations sur 1j / 5j / 1m, volatilité réalisée et signaux de marché.
    Les données sont récupérées automatiquement via [Yahoo Finance](https://finance.yahoo.com) grâce à `yfinance`.
    """
)

# Bouton d’actualisation des données
if st.button(" Actualiser les données"):
    st.session_state['market_data'] = get_market_snapshot()

# Si première ouverture, charger les données
if "market_data" not in st.session_state:
    st.session_state['market_data'] = get_market_snapshot()

# Données originales
df_all = st.session_state['market_data'].copy()

# Liste des classes disponibles pour le filtre
all_classes = [
    "ÉQUITY",
    "OBLIGATIONS SOUVERAINES",
    "OBLIGATIONS — PERFORMANCE",
    "CRÉDIT IG / HY",
    "TAUX DE RÉFÉRENCE",
    "COMMODITIES",
    "FOREX"
]

# Filtre dynamique par classe d’actif
col_select, col_button = st.columns([3, 1])
with col_select:
    selected_class = st.selectbox(" Filtrer par classe d’actif :", ["Toutes"] + all_classes)
with col_button:
    if st.button("↩ Réinitialiser"):
        selected_class = "Toutes"

# Application du filtre
df = df_all.copy()
if selected_class != "Toutes":
    sep_label = f"--- {selected_class.upper()} ---"
    start_idx = df[df["Actif"] == sep_label].index
    if not start_idx.empty:
        i = start_idx[0]
        df = df.iloc[i:]
        # Cherche la prochaine ligne de séparation
        next_sep = df[df["Actif"].str.startswith("---") & (df["Actif"] != sep_label)].index
        if not next_sep.empty:
            df = df.iloc[: next_sep[0] - i]

# Sous-titre
st.subheader(" Données marché live")
st.markdown("---")

# Détection des erreurs
df_erreurs = df[df["Niveau"].isnull()]
if not df_erreurs.empty:
    st.warning(f" Problème de récupération pour : {', '.join(df_erreurs['Actif'].tolist())}")

# Application du style
styled_df = style_market_table(df)

# Affichage du tableau stylé
st.write(styled_df, use_container_width=True, hide_index=True)

# Export Excel
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name="Marché", index=False)
excel_buffer.seek(0)

# Bouton de téléchargement
st.download_button(
    label="📥 Télécharger le tableau marché (Excel)",
    data=excel_buffer,
    file_name="snapshot_marche.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ================================
# MOTEUR D’ALERTE COMPACT
# ================================
st.markdown("## Moteur d'alerte – Top mouvements (global)")

# Utilise les données complètes pour l'alerte (pas filtrées)
df_alerts = df_all[df_all['∆ 1j'].notna() & ~df_all['Actif'].str.startswith('---')].copy()

# Calcul des variations %
df_alerts['% 1j'] = (df_alerts['∆ 1j'] / (df_alerts['Niveau'] - df_alerts['∆ 1j'])) * 100
df_alerts['% 5j'] = (df_alerts['∆ 5j'] / (df_alerts['Niveau'] - df_alerts['∆ 5j'])) * 100

# Top actifs par critère
top_hausses_1j = df_alerts.sort_values(by='% 1j', ascending=False).head(5)[['Actif', '% 1j']]
top_baisses_1j = df_alerts.sort_values(by='% 1j', ascending=True).head(5)[['Actif', '% 1j']]
top_hausses_5j = df_alerts.sort_values(by='% 5j', ascending=False).head(5)[['Actif', '% 5j']]
top_baisses_5j = df_alerts.sort_values(by='% 5j', ascending=True).head(5)[['Actif', '% 5j']]
top_vol = df_alerts.sort_values(by='Vol réalisée (%)', ascending=False).head(5)[['Actif', 'Vol réalisée (%)']]

# Affichage en 3 colonnes
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📈 Top 5 Hausses 1j")
    st.dataframe(top_hausses_1j, hide_index=True, use_container_width=True)
    st.markdown("#### 🚀 Top 5 Hausses 5j")
    st.dataframe(top_hausses_5j, hide_index=True, use_container_width=True)

with col2:
    st.markdown("#### 📉 Top 5 Baisses 1j")
    st.dataframe(top_baisses_1j, hide_index=True, use_container_width=True)
    st.markdown("#### 📉 Top 5 Baisses 5j")
    st.dataframe(top_baisses_5j, hide_index=True, use_container_width=True)

with col3:
    st.markdown("#### 🔺 Volatilité Réalisée Annualisée (%)")
    st.dataframe(top_vol, hide_index=True, use_container_width=True)
