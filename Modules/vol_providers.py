# modules/vol_providers.py  # Nom du fichier et emplacement dans le projet

import pandas as pd  # Importer pandas pour manipuler les données tabulaires
import datetime as dt  # Importer datetime pour gérer les dates
import os  # Importer os pour accéder aux variables d'environnement
import keyring  # Importer keyring pour stocker les credentials de manière sécurisée
import requests  # Importer requests pour les appels API HTTP
from pathlib import Path  # Importer Path pour manipuler les chemins de fichiers
from typing import Optional, Literal, Dict, Any  # Importer les types pour annotations
import modules.vol_core as vc  # Importer tes fonctions de base
import json  # Import pour manipuler les données JSON
try:
    import blpapi  # type: ignore  # Import de la librairie Bloomberg
except ImportError:
    blpapi = None  # On garde None si pas installé
    print("⚠️ blpapi non installé — fonctionnalités Bloomberg désactivées.")  # Avertissement


# Import interne depuis vol_core
from modules.vol_core import read_vol_file  # Importer la fonction de lecture de fichiers
from modules.vol_core import settings  # Importer la config YAML déjà chargée

# Exemple d'utilisation dans vol_providers.py
def connect_to_bloomberg():
    if not settings["data_providers"]["bloomberg_api"]:  # Vérifie si Bloomberg est activé dans YAML
        raise ValueError("Bloomberg API désactivée dans la configuration.")
    # Ici on utiliserait la clé Bloomberg ou un prompt utilisateur


# ---------------------------------
# GESTION DES IDENTIFIANTS
# ---------------------------------

def save_credential(provider: str, key: str, value: str, remember: bool=True):  # Sauvegarder un identifiant
    if remember:  # Si on souhaite conserver au-delà de la session
        keyring.set_password(f"vol_app_{provider}", key, value)  # Sauvegarder dans le trousseau OS
    else:  # Sinon, stockage temporaire
        os.environ[f"VOL_APP_{provider.upper()}_{key.upper()}"] = value  # Stocker dans variable d'environnement

def load_credential(provider: str, key: str) -> Optional[str]:  # Charger un identifiant
    val = keyring.get_password(f"vol_app_{provider}", key)  # Tenter de lire dans le trousseau OS
    if val is None:  # Si non trouvé
        val = os.environ.get(f"VOL_APP_{provider.upper()}_{key.upper()}")  # Lire dans variables d'environnement
    return val  # Retourner la valeur ou None

def delete_credential(provider: str, key: str):  # Supprimer un identifiant stocké
    try:
        keyring.delete_password(f"vol_app_{provider}", key)  # Supprimer du trousseau OS
    except keyring.errors.PasswordDeleteError:
        pass  # Ignorer si la clé n'existe pas
    os.environ.pop(f"VOL_APP_{provider.upper()}_{key.upper()}", None)  # Supprimer des variables d'environnement

# ---------------------------------
# PROVIDERS - FICHIERS LOCAUX
# ---------------------------------

def fetch_from_file(file_like) -> pd.DataFrame:  # Charger un fichier local
    return read_vol_file(file_like)  # Utiliser la fonction interne de vol_core

# ---------------------------------
# PROVIDERS - YAHOO FINANCE
# ---------------------------------

def fetch_from_yahoo(ticker: str, date: Optional[dt.date]=None) -> pd.DataFrame:  # Récupérer données depuis Yahoo
    try:
        import yfinance as yf  # Importer yfinance (lib externe)
    except ImportError:
        raise ImportError("Le module yfinance est requis pour fetch_from_yahoo")  # Lever erreur si non installé
    
    tk = yf.Ticker(ticker)  # Créer l'objet ticker Yahoo
    if not tk.options:  # Vérifier si le ticker a des options
        raise ValueError(f"Aucune option trouvée pour {ticker}")  # Lever erreur si aucune

    exp = tk.options[0]  # Prendre la première date d'expiration disponible
    opt_chain = tk.option_chain(exp)  # Charger la chaîne d'options pour cette date
    calls = opt_chain.calls  # Extraire les calls
    df = calls[["strike","impliedVolatility"]].copy()  # Garder uniquement strike et volatilité implicite
    df["TenorY"] = 1.0  # Assigner un tenor fictif de 1 an (placeholder)
    df["VolPct"] = df["impliedVolatility"] * 100  # Convertir en pourcentage
    df["StrikeParsed"] = df["strike"]  # Copier les strikes parsés
    return df[["TenorY","StrikeParsed","VolPct"]]  # Retourner les colonnes pertinentes

# ---------------------------------
# PROVIDERS - BLOOMBERG DESKTOP API
# ---------------------------------

def fetch_from_bloomberg_desktop(ticker: str, field_list: list) -> pd.DataFrame:  # Récupérer via Bloomberg Desktop API
    try:
        import blpapi  # type: ignore  # Importer la bibliothèque blpapi
    except ImportError:
        raise ImportError("Le module blpapi est requis pour Bloomberg Desktop API")  # Lever erreur si absent
    
    session = blpapi.Session()  # Créer une session Bloomberg
    if not session.start():  # Démarrer la session
        raise ConnectionError("Impossible de démarrer la session Bloomberg")  # Erreur si échec
    if not session.openService("//blp/refdata"):  # Ouvrir le service refdata
        raise ConnectionError("Impossible d'ouvrir le service refdata")  # Erreur si échec

    service = session.getService("//blp/refdata")  # Récupérer le service refdata
    request = service.createRequest("ReferenceDataRequest")  # Créer une requête de données
    request.getElement("securities").appendValue(ticker)  # Ajouter le ticker à la requête
    for f in field_list:  # Boucler sur les champs demandés
        request.getElement("fields").appendValue(f)  # Ajouter le champ
    session.sendRequest(request)  # Envoyer la requête

    df = pd.DataFrame()  # Créer un DataFrame vide (placeholder)
    while True:  # Boucler sur les événements de réponse
        ev = session.nextEvent()  # Lire l'événement suivant
        for msg in ev:  # Boucler sur les messages
            if msg.hasElement("securityData"):  # Vérifier présence des données
                pass  # Ici on implémenterait l'extraction réelle
        if ev.eventType() == blpapi.Event.RESPONSE:  # Si réponse complète
            break  # Sortir de la boucle
    return df  # Retourner les données (placeholder vide)

# ---------------------------------
# PROVIDERS - BLOOMBERG HAPI
# ---------------------------------

def fetch_from_bloomberg_hapi(endpoint: str, params: dict) -> Dict[str, Any]:  # Récupérer via Bloomberg HAPI
    base_url = "https://api.bloomberg.com"  # URL de base HAPI
    client_id = load_credential("bloomberg_hapi", "client_id")  # Charger client_id
    client_secret = load_credential("bloomberg_hapi", "client_secret")  # Charger client_secret

    if not client_id or not client_secret:  # Vérifier si credentials présents
        raise ValueError("Identifiants HAPI manquants")  # Lever erreur si absent

    token_url = f"{base_url}/auth/oauth2/token"  # URL pour récupérer le token OAuth
    r = requests.post(token_url, data={  # Envoyer la requête de token
        "grant_type": "client_credentials",  # Type de grant
        "client_id": client_id,  # ID client
        "client_secret": client_secret  # Secret client
    })
    r.raise_for_status()  # Lever erreur si HTTP != 200
    token = r.json().get("access_token")  # Extraire le token

    headers = {"Authorization": f"Bearer {token}"}  # Préparer les headers
    resp = requests.get(f"{base_url}/{endpoint}", headers=headers, params=params)  # Appeler l'endpoint
    resp.raise_for_status()  # Vérifier le statut HTTP
    return resp.json()  # Retourner les données JSON

# ---------------------------------
# PROVIDERS - OPENFIGI
# ---------------------------------

def fetch_from_openfigi(mapping_request: list) -> pd.DataFrame:  # Mapper instrument → FIGI
    api_key = load_credential("openfigi", "api_key")  # Charger la clé API
    if not api_key:  # Vérifier si présente
        raise ValueError("API key OpenFIGI manquante")  # Lever erreur si absente

    headers = {"Content-Type": "text/json", "X-OPENFIGI-APIKEY": api_key}  # Headers API
    url = "https://api.openfigi.com/v3/mapping"  # URL mapping OpenFIGI
    r = requests.post(url, headers=headers, data=json.dumps(mapping_request))  # Envoyer la requête POST
    r.raise_for_status()  # Vérifier statut HTTP
    return pd.DataFrame(r.json())  # Retourner la réponse sous forme DataFrame

# ---------------------------------
# INTERFACE UNIFIÉE
# ---------------------------------

def fetch_surface(provider: Literal["file","yahoo","bloomberg_desktop","bloomberg_hapi","openfigi"],
                  **kwargs) -> pd.DataFrame:  # Récupérer une surface de volatilité via provider
    if provider == "file":  # Si fichier local
        return fetch_from_file(kwargs["file_like"])  # Appeler provider fichier
    elif provider == "yahoo":  # Si Yahoo Finance
        return fetch_from_yahoo(kwargs["ticker"])  # Appeler provider Yahoo
    elif provider == "bloomberg_desktop":  # Si Bloomberg Desktop
        return fetch_from_bloomberg_desktop(kwargs["ticker"], kwargs.get("fields", []))  # Appeler Desktop API
    elif provider == "bloomberg_hapi":  # Si Bloomberg HAPI
        return fetch_from_bloomberg_hapi(kwargs["endpoint"], kwargs.get("params", {}))  # Appeler HAPI
    elif provider == "openfigi":  # Si OpenFIGI
        return fetch_from_openfigi(kwargs["mapping_request"])  # Appeler OpenFIGI
    else:  # Sinon provider inconnu
        raise ValueError(f"Provider inconnu : {provider}")  # Lever erreur
