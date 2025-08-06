import streamlit as st
import pandas as pd
from Modules.Pricing import forward_rate_zero_coupon

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

st.title("Taux forward (à partir de ZC)")

st.markdown("Calcul via fichier CSV")
csv_file = st.file_uploader("CSV avec colonnes r0, r1, t0, t1", type=["csv"], key="csv_fw_rate")

if csv_file:
    try:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip().str.lower()
        if not {"r0", "r1", "t0", "t1"}.issubset(df.columns):
            st.error("Colonnes attendues : r0, r1, t0, t1")
        else:
            df["taux_forward"] = df.apply(lambda row: forward_rate_zero_coupon(row["r0"], row["r1"], row["t0"], row["t1"]), axis=1)
            st.success("Taux calculés avec succès")
            st.dataframe(df)
    except Exception as e:
        st.error(f"Erreur : {e}")

st.markdown("---")
st.markdown("Saisie manuelle")

r0 = st.number_input("Taux ZC T0", value=0.02, format="%.4f", key="fw_r0")
r1 = st.number_input("Taux ZC T1", value=0.025, format="%.4f", key="fw_r1")
T0 = st.number_input("T0 (en années)", value=1.0, key="fw_T0")
T1 = st.number_input("T1 (en années)", value=1.5, key="fw_T1")

if st.button("Calculer le taux forward", key="fw_btn"):
    fwd = forward_rate_zero_coupon(r0, r1, T0, T1)
    st.success(f"Taux forward entre T0={T0} et T1={T1} : {fwd:.4%}")