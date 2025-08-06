import pandas as pd
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
def load_fred_rates(series_dict: dict, start_date: str, api_key: str) -> pd.DataFrame:
    """
    Récupère l'historique de séries FRED à partir d'une date donnée.

    Parameters:
        series_dict (dict): Dictionnaire {code_FRED: "label"} ex: {"DGS1": "1Y"}
        start_date (str): Date de début au format "YYYY-MM-DD"
        api_key (str): Clé API FRED (obtenue via https://fred.stlouisfed.org/)

    Returns:
        pd.DataFrame: DataFrame avec les dates en index et une colonne par maturité
    """
    if Fred is None:
        raise ImportError("fredapi n'est pas installé. Installez-le avec `pip install fredapi`.")

    fred = Fred(api_key=api_key)
    data = {}

    for code, label in series_dict.items():
        try :
            series = fred.get_series(code,observation_start = start_date)
            data[label] = series 
        except Exception as e:
            print (f"Erreur lors du chargement de {label}({code}):{e}")
            data[label] = None #On remplit avec None si erreur
    
    df = pd.DataFrame(data).dropna() #On retire les lignes incomplètes
    return df