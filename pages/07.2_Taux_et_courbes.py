import streamlit as st 
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import plotly.subplots as sp


from dotenv import load_dotenv
from data.taux_loader import get_us_yield_curve_snapshot, load_fred_rates_treasuries, load_eur_yield_curve_bce, load_bundesbank_yield_curve_sdmx

load_dotenv()  # charge le fichier .env

API_KEY = os.getenv("FRED_API_KEY")  # lit la clé

import streamlit as st
st.write("Clé FRED détectée :", API_KEY)

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

st.title ("Marché - Taux et Courbes")
st.subheader("Courbe des taux US (Treasury)")



###########################################################
############ Graphe  Courbe des taux US ###################
###########################################################
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

df_hist = load_fred_rates_treasuries(fred_series,start_date = start_date_10y,api_key = API_KEY)



#########################################################
######## 5 sous graphes des taux US historiques #########
#########################################################

#  Calcul dynamique de la date de début (10 ans glissants)
start_date_10y = (datetime.today() - timedelta(days=365.25 * 10)).strftime("%Y-%m-%d")

# Séries FRED à charger
fred_series = {"DGS1": "1Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS10": "10Y",
    "DGS30": "30Y"}

 # Chargement des données
df_hist = load_fred_rates_treasuries(fred_series, start_date=start_date_10y, api_key=API_KEY)

# Affichage sous forme de mini-graphes
st.subheader(" Historique par maturité (10 ans)")

cols = st.columns(len(fred_series))  # crée 5 colonnes côte à côte

for i, (code_fred, label) in enumerate(fred_series.items()):
    col = cols[i]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist.index,y=df_hist[label],mode="lines",name=label,line=dict(color=f"rgba({50 + i*40}, {100 + i*30}, {200 - i*30}, 1)")))

    fig.update_layout(title=label,
        margin=dict(l=20, r=20, t=30, b=20),
        height=250,
        xaxis=dict(showgrid=False, tickformat="%Y"),
        yaxis=dict(title="", showgrid=True, tickformat=".2f"),)
    col.plotly_chart(fig, use_container_width=True)



####################################################################
######## Graphes Yield Curve AAA Zone EURO ########
####################################################################


import streamlit as st
import plotly.graph_objects as go


st.title("🇪🇺 Courbe des taux AAA - Zone Euro (BCE)")

# Chargement des taux
df_eur = load_eur_yield_curve_bce()

if df_eur.empty:
    st.warning("Aucune donnée récupérée depuis la BCE.")
else:
    # Dernière date dispo
    last_row = df_eur.dropna().iloc[-1]
    last_date = df_eur.dropna().index[-1].strftime("%Y-%m-%d")

    # Graphe interactif
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_row.index,
        y=last_row.values,
        mode='lines+markers',
        name="Yield curve AAA"
    ))

    fig.update_layout(
        title=f"Courbe spot AAA - BCE ({last_date})",
        xaxis_title="Maturité",
        yaxis_title="Taux (%)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


    ###################################################################################
    ########### Affichage des 5 sous graphes hisotrique 10 ans par maturité ###########
    ###################################################################################


st.subheader("📉Historique par maturité (10 ans) – Zone Euro (AAA BCE)")

# On limite aux 10 dernières années
df_last_10y = df_eur[df_eur.index >= (pd.Timestamp.today() - pd.DateOffset(years=10))]

# Maturités à tracer
maturities = ["1Y", "3Y", "5Y", "10Y", "30Y"]
cols_found = [m for m in maturities if m in df_last_10y.columns]

# Création des 5 sous-graphes
fig = sp.make_subplots(
    rows=1, cols=5,
    shared_yaxes=False,
    subplot_titles=cols_found,
    horizontal_spacing=0.04
)

for i, mat in enumerate(cols_found):
    fig.add_trace(
        go.Scatter(x=df_last_10y.index, y=df_last_10y[mat], name=mat, mode="lines"),
        row=1, col=i+1
    )

# Mise en forme
fig.update_layout(
    height=350,
    showlegend=False,
    template="plotly_white",
    title_text=None
)

st.plotly_chart(fig, use_container_width=True)

###########################################################################

