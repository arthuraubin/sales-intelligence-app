import streamlit as st
import pandas as pd
from Modules.Pricing import black_scholes_price

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

st.title("Page Option Européenne")


st.markdown("### Pricing de plusieurs options via fichier CSV")

# Upload CSV
csv_file_euro = st.file_uploader("Uploader un fichier CSV contenant les options à pricer", type=["csv"], key="csv_euro_upload")

if csv_file_euro is not None:
    try:
        df = pd.read_csv(csv_file_euro)
        expected_cols = {"s", "k", "t", "r", "sigma", "option_type"}
        df.columns = df.columns.str.strip().str.lower()

        if not expected_cols.issubset(set(df.columns)):
            st.error(f"Le fichier doit contenir les colonnes suivantes : {sorted(expected_cols)}")
        else:
            prices = []
            for idx, row in df.iterrows():
                try:
                    if pd.isna(row["option_type"]) or row["option_type"].lower() not in ["call", "put"]:
                        raise ValueError("option_type invalide (doit être 'call' ou 'put')")
                    price = black_scholes_price(
                        S=row["s"], K=row["k"], T=row["t"], r=row["r"],
                        sigma=row["sigma"], option_type=row["option_type"].lower()
                    )
                    prices.append(price)
                except Exception as e:
                    prices.append(None)
                    st.warning(f"Erreur ligne {idx+1} : {e}")
            df["prix_option_europeenne"] = prices
            st.success("Options pricées avec succès !")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")

# Formulaire manuel
st.markdown("---")
st.markdown("### Saisie manuelle")

option_type = st.selectbox("Type d'option", ["call", "put"])
S = st.number_input("Prix spot (S)", value=100.0, key="euro_S")
K = st.number_input("Strike (K)", value=100.0, key="euro_K")
T = st.number_input("Maturité (en années)", value=1.0, key="euro_T")
r = st.number_input("Taux sans risque (r)", value=0.03, key="euro_r")
sigma = st.number_input("Volatilité implicite (σ)", value=0.20, key="euro_sigma")

if st.button("Calculer le prix", key="euro_btn"):
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    st.success(f"Prix de l'option {option_type.upper()} : {price:.2f} €")
