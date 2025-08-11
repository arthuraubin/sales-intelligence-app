import pandas as pd
from io import StringIO
import numpy as np
import xml.etree.ElementTree as ET  # Pour parser le XML
from datetime import datetime  # Pour parser les dates
from scipy.interpolate import PchipInterpolator
from fredapi import Fred
from dotenv import load_dotenv
import os
import yfinance as yf
import requests
from io import BytesIO





try:
    from fredapi import Fred  # Pour accéder à l’API FRED
except ImportError:
    Fred = None  # Sécurité si le module n'est pas installé

def get_us_yield_curve_snapshot(api_key: str): # return un  pd.DataFrame
    """
    Récupère les taux US à maturités standard (1Y, 3Y, 5Y, 10Y, 30Y) via l'API FRED
    pour construire une courbe des taux instantanée.

    Parameters:
        api_key (str): Clé API FRED

    Returns:
        pd.DataFrame: DataFrame avec deux colonnes ['Maturity', 'Rate']
                      prêt à être affiché dans un graphique
    """
    if Fred is None:
        raise ImportError("fredapi n'est pas installé. Installez-le avec `pip install fredapi`.")

    fred = Fred(api_key= api_key)

    # Dictionnaire des séries FRED pour la courbe US
    maturity_codes = {"1Y": "DGS1",
        "3Y": "DGS3",
        "5Y": "DGS5",
        "10Y": "DGS10",
        "30Y": "DGS30"}

    data = {}
    for label, code in maturity_codes.items(): #items pour accéder à la fois aux clefs et aux valeurs 
        try:
            # On récupère uniquement la dernière valeur disponible
            series = fred.get_series_latest_release(code)
            latest_value = series.dropna().iloc[-1] if not series.empty else None
            data[label] = latest_value
        except Exception as e:
            print(f"Erreur lors du chargement de {label} ({code}): {e}")
            data[label] = None

    # Création du DataFrame final
    df_curve = pd.DataFrame({"Maturity": list(data.keys()),
        "Rate": list(data.values())})

    return df_curve


# Récupère les données historiques des taux US

def load_fred_rates_treasuries(series_dict: dict, start_date: str, api_key: str) -> pd.DataFrame:
    """
    Récupère l'historique de séries FRED à partir d'une date donnée.
    """
    fred = Fred(api_key=api_key)
    data = {}

    for code, label in series_dict.items():
        try:
            series = fred.get_series(code, observation_start=start_date)
            data[label] = series
        except Exception as e:
            print(f"Erreur lors du chargement de {label} ({code}) : {e}")
            # Série vide mais avec un index datetime vide pour éviter l'erreur
            data[label] = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)

    # Création du DataFrame à partir des séries alignées sur les dates
    df = pd.DataFrame(data)

    # Retrait des lignes où toutes les colonnes sont vides
    df = df.dropna(how="all")

    return df


##########################################################################
################ Récupération yield curve AAA Zone Euro (BCE) #######################
##########################################################################

def load_eur_yield_curve_bce(maturities=["1Y", "3Y", "5Y", "10Y", "30Y"]):
    """
    Récupère les spot rates (zero-coupon) pour la courbe AAA EUR depuis l'API BCE (endpoint officiel).
    Retourne un DataFrame avec les maturités en colonnes.
    """

    # Mapping maturités -> codes BCE
    maturity_map = {
        "1Y": "SR_1Y",
        "2Y": "SR_2Y",
        "3Y": "SR_3Y",
        "4Y": "SR_4Y",
        "5Y": "SR_5Y",
        "7Y": "SR_7Y",
        "10Y": "SR_10Y",
        "15Y": "SR_15Y",
        "20Y": "SR_20Y",
        "30Y": "SR_30Y"
    }

    # Nouveau endpoint API officiel (recommandé)
    base_url = "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{maturity}?format=csvdata"

    data = {}

    for mat in maturities:
        code = maturity_map.get(mat)
        if not code:
            continue

        url = base_url.format(maturity=code)

        try:
            r = requests.get(url)
            if r.status_code == 200:
                df = pd.read_csv(BytesIO(r.content))
                df["TIME_PERIOD"] = pd.to_datetime(df["TIME_PERIOD"])
                df.set_index("TIME_PERIOD", inplace=True)
                df = df.rename(columns={"OBS_VALUE": mat})
                data[mat] = df[[mat]]
            else:
                print(f"Requête échouée ({r.status_code}) pour {mat}")
        except Exception as e:
            print(f"Erreur lors du chargement de {mat} : {e}")

    if not data:
        print(" Aucun taux n’a pu être récupéré depuis l’API BCE.")
        return pd.DataFrame()  #  assure que le retour est un DataFrame vide

    df_curve = pd.concat(data.values(), axis=1)
    df_curve = df_curve.sort_index()
    df_curve.dropna(how='all', inplace=True)

    return df_curve  


    ####################################


