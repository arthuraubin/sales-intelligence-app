import streamlit as st
import pandas as pd
from Modules.Pricing import forward_fx_price

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

st.title("💱 Forward FX")

st.markdown("### Pricing via fichier CSV")
csv_file_fx = st.file_uploader("Uploader un fichier CSV (colonnes : s, r_dom, r_for, t)", type=["csv"], key="csv_fx_upload")

if csv_file_fx:
    try:
        df = pd.read_csv(csv_file_fx)
        df.columns = df.columns.str.strip().str.lower()
        expected_cols = {"s", "r_dom", "r_for", "t"}

        if not expected_cols.issubset(set(df.columns)):
            st.error(f"Le fichier doit contenir : {sorted(expected_cols)}")
        else:
            df["taux_forward"] = df.apply(lambda row: forward_fx_price(row["s"], row["r_dom"], row["r_for"], row["t"]), axis=1)
            st.success("Forwards calculés")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Erreur de fichier : {e}")

st.markdown("---")
st.markdown("### Saisie manuelle")

S = st.number_input("Taux spot (S)", value=1.10, key="fx_S")
r_dom = st.number_input("Taux domestique", value=0.03, format="%.4f", key="fx_r_dom")
r_for = st.number_input("Taux étranger", value=0.01, format="%.4f", key="fx_r_for")
T = st.number_input("Maturité (années)", value=0.5, key="fx_T")

if st.button("Calculer le taux forward", key="fx_btn"):
    forward = forward_fx_price(S, r_dom, r_for, T)
    st.success(f"Taux forward à {T:.2f} an(s) : {forward:.4f}")
