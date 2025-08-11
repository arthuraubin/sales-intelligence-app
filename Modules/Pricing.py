from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
import pandas as pd
import csv



import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple, Dict



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
                    forwards = [forward_rate_zero_coupon(zc_rates_shifted[i], zc_maturities[i],zc_rates_shifted[i + 1], zc_maturities[i + 1])
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


# ===========================
#  BARRIER OPTIONS — FAST MC
#  Vectorized + Chunking + Early-Stop on target stderr
# ===========================


"""Module Pricing : fonctions de calcul pour Options Barrières.
Version optimisée : Monte Carlo vectorisé + chunking + arrêt sur précision.
Aucune dépendance graphique ici.
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple, Dict

# ---- Types ----
OptionType = Literal["Call", "Put"]
BarrierEffect = Literal["Knock-In", "Knock-Out"]
BarrierDirection = Literal["Up", "Down"]
Monitoring = Literal["discrete", "continuous"]  # continuous = approx Brownian-Bridge

# ---- Utils ----
def _check_inputs_barrier(
    S0: float,
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType,
    barrier_effect: BarrierEffect,
    barrier_direction: BarrierDirection,
) -> None:
    if S0 <= 0 or B <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("Paramètres invalides: S0>0, B>0, sigma>0, T>0 requis.")
    if option_type not in ("Call", "Put"):
        raise ValueError("option_type doit être 'Call' ou 'Put'.")
    if barrier_effect not in ("Knock-In", "Knock-Out"):
        raise ValueError("barrier_effect doit être 'Knock-In' ou 'Knock-Out'.")
    if barrier_direction not in ("Up", "Down"):
        raise ValueError("barrier_direction doit être 'Up' ou 'Down'.")

def _vanilla_payoff(S_T: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    if option_type == "Call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)

# ---- Brownian-Bridge crossing probabilities (vectorisées) ----
def _bb_cross_prob_up(
    logS_t: np.ndarray,
    logS_tp1: np.ndarray,
    logB: float,
    dt: float,
    sigma: float,
) -> np.ndarray:
    """Proba de franchissement up entre t et t+dt si les 2 extrémités < B."""
    below = (logS_t < logB) & (logS_tp1 < logB)
    num = -2.0 * (logB - logS_t) * (logB - logS_tp1)
    den = (sigma * sigma) * dt
    p = np.zeros_like(logS_t)
    with np.errstate(over="ignore"):
        p[below] = np.exp(num[below] / den)
    np.clip(p, 0.0, 1.0, out=p)
    return p

def _bb_cross_prob_down(
    logS_t: np.ndarray,
    logS_tp1: np.ndarray,
    logB: float,
    dt: float,
    sigma: float,
) -> np.ndarray:
    """Proba de franchissement down entre t et t+dt si les 2 extrémités > B."""
    above = (logS_t > logB) & (logS_tp1 > logB)
    num = -2.0 * (logS_t - logB) * (logS_tp1 - logB)
    den = (sigma * sigma) * dt
    p = np.zeros_like(logS_t)
    with np.errstate(over="ignore"):
        p[above] = np.exp(num[above] / den)
    np.clip(p, 0.0, 1.0, out=p)
    return p

# ---- Simulation d'un "chunk" (vectorisée, mémoire safe) ----
def _simulate_chunk_pv(
    S0: float,
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType,
    barrier_effect: BarrierEffect,
    barrier_direction: BarrierDirection,
    rebate: float,
    monitoring: Monitoring,
    n_steps: int,
    n_paths: int,              # <-- bien présent ici
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:


    """Retourne la PV (payoff actualisé) d'un bloc de chemins."""
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)

    # Simule increments log en (n_paths, n_steps)
    Z = rng.standard_normal((n_paths, n_steps))
    inc = drift + vol * Z
    np.cumsum(inc, axis=1, out=inc)

    logS = np.empty((n_paths, n_steps + 1), dtype=float)
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = logS[:, [0]] + inc

    S = np.exp(logS, dtype=float)

    # Hit discret
    if barrier_direction == "Up":
        discrete_hit = (S[:, 1:] >= B).any(axis=1)
    else:
        discrete_hit = (S[:, 1:] <= B).any(axis=1)

    touched = discrete_hit.copy()

    # Correction Brownian-Bridge (approx) pour monitoring "continuous"
    if monitoring == "continuous":
        idx = ~touched
        if np.any(idx):
            x = logS[idx]          # (m, n_steps+1)
            x_t = x[:, :-1]
            x_tp1 = x[:, 1:]
            logB = np.log(B)
            if barrier_direction == "Up":
                p = _bb_cross_prob_up(x_t, x_tp1, logB, dt, sigma)
            else:
                p = _bb_cross_prob_down(x_t, x_tp1, logB, dt, sigma)
            # Proba no-hit = prod(1 - p) ; calcule en log pour stabilité
            with np.errstate(divide="ignore"):
                log_nohit = np.sum(np.log1p(-p), axis=1)
            p_hit = 1.0 - np.exp(log_nohit)
            u = rng.random(p_hit.shape[0])
            touched[idx] |= (u < p_hit)

    payoff_van = _vanilla_payoff(S[:, -1], K, option_type)

    if barrier_effect == "Knock-Out":
        payoff = np.where(touched, rebate, payoff_van)
    else:  # Knock-In
        payoff = np.where(touched, payoff_van, rebate)

    pv = np.exp(-r * T) * payoff
    return pv

# ---- Monte Carlo rapide avec arrêt sur précision ----
def mc_barrier_price_fast(
    S0: float,
    K: float,
    B: float,
    r: float,
    sigma: float,
    T: float,
    option_type: OptionType,
    barrier_effect: BarrierEffect,
    barrier_direction: BarrierDirection,
    rebate: float = 0.0,
    monitoring: Monitoring = "discrete",
    target_stderr: float = 0.02,
    chunk_paths: int = 10_000,
    max_paths: int = 120_000,
    n_steps: int = 125,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Monte Carlo rapide : vectorisé + chunking + arrêt anticipé sur target_stderr."""
    _check_inputs_barrier(S0, K, B, r, sigma, T, option_type, barrier_effect, barrier_direction)

    rng = np.random.default_rng(seed)
    agg_sum = 0.0
    agg_sum2 = 0.0
    total = 0

    while total < max_paths:
        n = min(chunk_paths, max_paths - total)
        pv = _simulate_chunk_pv(
            S0=S0,
            K=K,
            B=B,
            r=r,
            sigma=sigma,
            T=T,
            option_type=option_type,
            barrier_effect=barrier_effect,
            barrier_direction=barrier_direction,
            rebate=rebate,
            monitoring=monitoring,
            n_steps=n_steps,
            n_paths=n)

            
        s = float(pv.sum())
        s2 = float((pv * pv).sum())
        agg_sum += s
        agg_sum2 += s2
        total += n

        mean = agg_sum / total
        var = max(agg_sum2 / total - mean * mean, 0.0)
        stderr = (var ** 0.5) / (total ** 0.5)
        if stderr <= target_stderr:
            break

    return {"price": float(mean), "stderr": float(stderr), "n_paths": int(total), "monitoring": monitoring}

# ---- Batch API ----
def barrier_price_row_fast(
    row: pd.Series,
    monitoring: Monitoring = "discrete",
    target_stderr: float = 0.02,
    chunk_paths: int = 10_000,
    max_paths: int = 120_000,
    n_steps: int = 125,
    seed: Optional[int] = None,
) -> Tuple[float, float, int]:
    """Calcule prix, stderr, n_paths consommés pour une ligne de DataFrame."""
    res = mc_barrier_price_fast(
        S0=float(row["S0"]),
        K=float(row["K"]),
        B=float(row["B"]),
        r=float(row["r"]),
        sigma=float(row["sigma"]),
        T=float(row["T"]),
        option_type=str(row["Option Type"]),
        barrier_effect=str(row["Barrier Effect"]),
        barrier_direction=str(row["Barrier Direction"]),
        rebate=float(row.get("Rebate", 0.0)),
        monitoring=monitoring,
        target_stderr=target_stderr,
        chunk_paths=chunk_paths,
        max_paths=max_paths,
        n_steps=n_steps,
        seed=seed,
    )
    return res["price"], res["stderr"], res["n_paths"]

def barrier_price_batch(
    df: pd.DataFrame,
    monitoring: Monitoring = "discrete",
    target_stderr: float = 0.02,
    chunk_paths: int = 10_000,
    max_paths: int = 120_000,
    n_steps: int = 125,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prix batch pour un tableau mixant KI/KO & Up/Down & Call/Put + rebate.
    Colonnes attendues min:
      ["Option ID","Option Type","Barrier Effect","Barrier Direction","S0","K","B","T","r","sigma"]
    Optionnel: "Rebate"
    """
    required = [
        "Option ID",
        "Option Type",
        "Barrier Effect",
        "Barrier Direction",
        "S0",
        "K",
        "B",
        "T",
        "r",
        "sigma",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    work = df.copy()
    if "Rebate" not in work.columns:
        work["Rebate"] = 0.0

    prices, stderrs, used_paths = [], [], []
    for _, row in work.iterrows():
        p, se, npth = barrier_price_row_fast(
            row,
            monitoring=monitoring,
            target_stderr=target_stderr,
            chunk_paths=chunk_paths,
            max_paths=max_paths,
            n_steps=n_steps,
            seed=seed,
        )
        prices.append(p)
        stderrs.append(se)
        used_paths.append(npth)

    out = work.copy()
    out["Price"] = prices
    out["StdErr"] = stderrs
    out["SimPathsUsed"] = used_paths
    return out

# ---- Payoff pédagogique (pour graphes) ----
def payoff_barrier_curve_pedagogique(
    S_T: np.ndarray,
    option_type: OptionType,
    barrier_effect: BarrierEffect,
    barrier_direction: BarrierDirection,
    K: float,
    B: float,
) -> np.ndarray:
    """Courbe de payoff à maturité (approx pédagogique, non path-dependent)."""
    vanilla = _vanilla_payoff(S_T, K, option_type)
    if barrier_effect == "Knock-Out":
        if barrier_direction == "Up":
            return np.where(S_T >= B, 0.0, vanilla)
        return np.where(S_T <= B, 0.0, vanilla)
    # Knock-In
    if barrier_direction == "Up":
        return np.where(S_T >= B, vanilla, 0.0)
    return np.where(S_T <= B, vanilla, 0.0)
