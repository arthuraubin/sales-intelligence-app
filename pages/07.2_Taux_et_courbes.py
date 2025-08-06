import streamlit as st 
import plotly.express as px
from datetime import datetime, timedelta
from data.taux_loader import get_us_yield_curve_snapshot, load_fred_rates


# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

st.title ("Marché - Taux et Courbes")
st.subheader("Courbe des taux US")

API_KEY = "d50b19d908e8dd0946a660e18ac6f163"

#Graphe  Courbe des taux US 
df_yield = get_us_yield_curve_snapshot(API_KEY)

# Vérifie que des taux ont bien été récupérés
if df_yield['Rate'].isnull().all():
    st.warning("Impossible de récupérer la courbe des taux US. Verifier la connexion de la clef API")
else :
    #Affichage de la courbe 
    fig = px.line(df_yield,x ="Maturity",y = "Rate",markers= True,title = "Courbe des taux US", labels ={"Rate":"Taux(%)","Maturity":"Echéance"})
    fig.update_traces(line_shape="spline") #Courbe lissée
    fig.update_layout(yaxis_tickformat =".2f",xaxis_title="Maturité",yaxis_title="Taux (%)")
    st.plotly_chart(fig,use_container_width=True)


#Sous graphes des taux historiques des taux US par maturité

start_date_10y = (datetime.today() - timedelta(days=365.25 * 10)).strftime("%Y-%m-%d") # Calcul dynamique de la date de début (10 ans glissants)

fred_series = {"DGS1": "1Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y"}

df_hist = load_fred_rates(fred_series,start_date = start_date_10y,api_key = API_KEY)