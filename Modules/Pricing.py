import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
import pandas as pd
import csv


def read_csv_auto_sep(file):
    try:
        df = pd.read_csv(file, sep=None, engine="python", encoding="utf-8-sig")
        if df.shape[1] == 1 and ',' in df.columns[0]:
            file.seek(0)
            df = pd.read_csv(file, sep=",", encoding="utf-8-sig")
        return df
    except Exception:
        file.seek(0)
        return pd.read_csv(file, sep=",", encoding="utf-8-sig")



def black_scholes_price (S,K,T,r,sigma,option_type="call"):

#calcul de d1 et d2
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1- sigma *np.sqrt(T)

    if option_type =="call":
        price = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2) #norm.cdf = fonction de répartition de la loi normale strandard N()
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else :
        raise ValueError("option_type doit être 'call' ou 'put'") #raise value error permet de déclencher volontairement l'erreur si le type n'est pas reconnu

    return price



def forward_fx_price(S,r_dom,r_for,T):
    forward = S*np.exp((r_dom-r_for)*T)
    return forward



def digital_option_price (S,K,T,r,sigma,option_type ="call",payoff=10):
    d2 = (np.log(S/K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        price = payoff*np.exp(-r*T)*norm.cdf(d2) #norm.cdf --> proba de finir dans la monnaie (représente la loi normale standard)
    elif option_type == "put":
        price = payoff*np.exp(-r*T)*norm.cdf(-d2)
    else :
        raise ValueError ("option_type doit être 'call' ou 'put'")
    return price



def caplet_price (F,K,T,sigma,r,delta,notional): # Calculer le prix d'un caplet (modèle Black 76)
   if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0  # Évite les erreurs de log ou racine
   d1 = d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
   d2 = d1 - sigma * np.sqrt(T)
   caplet_price = notional*delta*np.exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))
   return caplet_price


def cap_price (forwards,K,maturities,sigmas,r,delta,notional):
    cap_price = 0.0 
    for i in range(len(forwards)):#boucle sur chaque période
        F=forwards[i] #extrait les inputs de la période i
        T=maturities[i]
        sigma=sigmas[i]
        caplet=caplet_price(F,K,T,sigma,r,delta,notional) #calcule le caplet de la fonction existante
        cap_price += caplet #ajoute chaque caplet au total
    return cap_price


def forward_rate_zero_coupon (r0,T0,r1,T1):
    if T1 == T0 :
        return r1
    return (r1*T1-r0*T0)/(T1-T0)


def load_volatility_curve(csv_file):
    df = pd.read_csv(csv_file) # Lecture du fichier CSV contenant la courbe de volatilité dans un DataFrame pandas
    maturities = df["maturity"].values  # Extraction des colonnes 'maturity' (maturités) et 'vol' (volatilités) sous forme de tableaux numpy
    volatilities = df["vol"].values
    return maturities,volatilities # La fonction retourne deux tableaux : les maturités et les volatilités correspondantes


def build_volatility_curve(maturities,volatilities):
    spline = make_interp_spline(maturities,volatilities,k=2) #spline quadratique --> approximer une courbe de vol continue pr un nb limité de points discrets (maturité + vol)
    return lambda T : float(spline(T)) # renvoie une fonction anonyme qui prend T en entrée --> applique la spline pour obtenir la volatilité interpolée --> convertit le résultat en float (nécessaire car spline renvoie un type numpy)


def floorlet_price(F, K, T, sigma, r, delta, notional):
    if F <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    floorlet = notional * delta * np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return floorlet


def floor_price(forwards, K, maturities, sigmas, r, delta, notional):
    price = 0.0
    for i in range(len(forwards)):
        F = forwards[i]
        T = maturities[i]
        sigma = sigmas[i]
        floorlet = floorlet_price(F, K, T, sigma, r, delta, notional)
        price += floorlet
    return price


def stress_test(vol_df, zc_df, cap_or_floor="cap"):
    results = []
    stress_vols = [0.9, 1.0, 1.1]      # Volatilité multipliée par -10%, 0%, +10%
    stress_rates = [-0.005, 0.0, 0.005]  # Taux ZC shifés de -50 bps, 0, +50 bps

    for idx in range(len(vol_df)):
        try:
            # === Paramètres de base ===
            r_base = float(vol_df.iloc[idx]['r'])
            K = float(vol_df.iloc[idx]['k'])
            delta = float(vol_df.iloc[idx]['delta'])
            notional = float(vol_df.iloc[idx]['notional'])

            zc_maturities = [float(x) for x in str(zc_df.iloc[idx]['zc_maturities']).split(';')]
            zc_rates_base = [float(x) for x in str(zc_df.iloc[idx]['zc_rates']).split(';')]
            maturities = [delta * (i + 1) for i in range(len(zc_maturities) - 1)]

            forwards_base = [forward_rate_zero_coupon(zc_rates_base[i], zc_maturities[i],
                                                      zc_rates_base[i + 1], zc_maturities[i + 1])
                             for i in range(len(zc_maturities) - 1)]

            vol_maturities = [float(x) for x in str(vol_df.iloc[idx]['maturities']).split(';')]
            vols_base = [float(x) for x in str(vol_df.iloc[idx]['vols']).split(';')]
            vol_func_base = build_volatility_curve(vol_maturities, vols_base)
            sigmas_base = [vol_func_base(t) for t in maturities]
            vol_moy_base = np.mean(sigmas_base)

            # === Prix de référence ===
            base_price = cap_price(forwards_base, K, maturities, sigmas_base, r_base, delta, notional) \
                         if cap_or_floor == "cap" else \
                         floor_price(forwards_base, K, maturities, sigmas_base, r_base, delta, notional)

            # === Stress test ===
            for vol_multiplier in stress_vols:
                for rate_shift in stress_rates:
                    zc_rates_shifted = [r + rate_shift for r in zc_rates_base]
                    forwards = [forward_rate_zero_coupon(zc_rates_shifted[i], zc_maturities[i],
                                                         zc_rates_shifted[i + 1], zc_maturities[i + 1])
                                for i in range(len(zc_maturities) - 1)]

                    vols_shifted = [v * vol_multiplier for v in vols_base]
                    vol_func = build_volatility_curve(vol_maturities, vols_shifted)
                    sigmas_shifted = [vol_func(t) for t in maturities]
                    vol_moy_shifted = np.mean(sigmas_shifted)
                    r_shifted = r_base + rate_shift

                    stress_price = cap_price(forwards, K, maturities, sigmas_shifted, r_shifted, delta, notional) \
                                   if cap_or_floor == "cap" else \
                                   floor_price(forwards, K, maturities, sigmas_shifted, r_shifted, delta, notional)

                    scenario_label = f"Vol x{vol_multiplier}, ZC {rate_shift * 10000:+.0f}bps"

                    results.append({f"{'Cap' if cap_or_floor == 'cap' else 'Floor'} ID": idx + 1,
                        "Scénario": scenario_label,
                        "Prix non stressé (€)": round(base_price, 2),
                        "Prix stressé (€)": round(stress_price, 2),
                        "Différence (€)": round(stress_price - base_price, 2),
                        "Sensibilité (%)": round(100 * (stress_price - base_price) / base_price, 2) if base_price != 0 else None,
                        "Strike": round(K, 4),
                        "Notional": round(notional),
                        "Taux sans risque stressé": round(r_shifted, 4),
                        "Volatilité moyenne stressée": round(vol_moy_shifted, 4)})

        except Exception as e:
            results.append({"Floor ID": idx + 1,"Scénario": "Erreur","Erreur": str(e)})

    return results
