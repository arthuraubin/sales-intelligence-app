import streamlit as st
import pandas as pd
from Modules.Pricing import digital_option_price

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")
    
st.title("Option Digitale")

st.markdown("Pricing via fichier CSV")

csv_file = st.file_uploader("Uploader un CSV (colonnes : s, k, t, r, sigma, payoff, option_type)", type=["csv"], key="csv_digital_upload")

if csv_file:
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip().str.lower()
        expected = {"s", "k", "t", "r", "sigma", "payoff", "option_type"}
        if not expected.issubset(df.columns):
            st.error(f"Colonnes manquantes : {sorted(expected)}")
        else:
            df["prix_digital"] = df.apply(
                lambda row: digital_option_price(row["s"], row["k"], row["t"], row["r"], row["sigma"], row["option_type"], row["payoff"]), axis=1
            )
            st.success("Options digitales pricées")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Erreur : {e}")

st.markdown("---")
st.markdown("Saisie manuelle")

option_type = st.selectbox("Type d'option", ["call", "put"], key="digi_type")
S = st.number_input("Prix spot (S)", value=100, key="digi_S")
K = st.number_input("Strike (K)", value=100, key="digi_K")
T = st.number_input("Maturité", value=1.0, key="digi_T")
r = st.number_input("Taux sans risque", value=0.03, format="%.4f", key="digi_r")
sigma = st.number_input("Volatilité", value=0.20, format="%.4f", key="digi_sigma")
payoff = st.number_input("Payoff (€)", value=10.0, key="digi_payoff")

if st.button("Calculer le prix digital", key="digi_btn"):
    price = digital_option_price(S, K, T, r, sigma, option_type, payoff)
    st.success(f"Prix de l’option {option_type.upper()} : {price:.2f} €")
