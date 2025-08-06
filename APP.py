import streamlit as st

# Config de la page
st.set_page_config(
    page_title="Sales Intelligence App",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Header ---
st.markdown("<h1 style='text-align: center; color: #003366;'>Sales Intelligence App</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Plateforme professionnelle d’aide à la vente cross-asset</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Introduction ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("#### 🚀 Fonctionnalités clés :")
    st.markdown("""
    - 📈 Pricer des produits dérivés standards (options, caps/floors, forwards)
    - 🧠 Simuler des scénarios de marché
    - 📊 Visualiser les courbes de volatilité et de taux
    - 📝 Générer des pitchs personnalisés pour vos clients
    """)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Finance_Icon.png/600px-Finance_Icon.png", width=160)

st.markdown("---")

# --- Accès rapide aux modules ---
st.markdown("### 🔎 Accès aux modules")
tabs = st.tabs(["📌 Produits", "📉 Marché", "👤 Clients", "📝 Pitch"])

# Onglet Produits
with tabs[0]:
    st.markdown("#### Produits Dérivés")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.page_link("pages/01_Option_Europeenne.py", label="Option Européenne", icon="📄")
        st.page_link("pages/02_Forward_FX.py", label="Forward FX", icon="📄")
    with col2:
        st.page_link("pages/03_Option_Digitale.py", label="Option Digitale", icon="📄")
        st.page_link("pages/04_Cap_de_taux.py", label="Cap de taux", icon="📄")
    with col3:
        st.page_link("pages/05_Floor_de_taux.py", label="Floor de taux", icon="📄")
        st.page_link("pages/06_Forward_rate_zero_coupon.py", label="Taux Forward (ZC)", icon="📄")

# Onglet Marché
with tabs[1]:
    st.markdown("#### Dashboards Marché")
    st.page_link("pages/07.1_Vue_synthetique.py", label="Vue synthétique", icon="📊")
    st.page_link("pages/07.2_Taux_et_courbes.py", label="Taux & courbes", icon="📉")
    st.page_link("pages/07.3_Volatilite.py", label="Volatilité", icon="🌪️")
    st.page_link("pages/07.4_FX_et_Forwards.py", label="FX & Forwards", icon="💱")
    st.page_link("pages/07.5_Credit.py", label="Crédit", icon="🏦")
    st.page_link("pages/07.6_Insights.py", label="Insights", icon="🧠")

# Onglet Clients
with tabs[2]:
    st.markdown("#### Gestion des clients")
    st.page_link("pages/08_Clients.py", label="Clients", icon="👥")

# Onglet Pitch
with tabs[3]:
    st.markdown("#### Génération de pitch")
    st.page_link("pages/09_Pitch.py", label="Pitch client", icon="📝")
