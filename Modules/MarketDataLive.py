import yfinance as yf
import pandas as pd
import datetime as dt

# Dictionnaire des actifs à surveiller (nom : ticker Yahoo)
TICKERS = {
    # ===== EQUITY =====
    '--- ÉQUITY (Marchés Développés & Émergents) ---': None,
    'S&P 500 (US)': '^GSPC',
    'Nasdaq 100 (US)': '^NDX',
    'EuroStoxx 50 (Europe)': '^STOXX50E',
    'FTSE 100 (UK)': '^FTSE',
    'Nikkei 225 (Japon)': '^N225',
    'S&P/ASX 200 (Australie)': '^AXJO',
    'MSCI Emerging Markets': 'EEM',
    'Small Caps US (Russell 2000)': '^RUT',
    'Small Caps Europe (SCZ ETF)': 'SCZ',

    # ===== BONDS =====
    '--- OBLIGATIONS SOUVERAINES — TAUX NOMINATIFS (%) ---': None,
    'US 10Y': '^TNX',
    'Germany 10Y': 'BUND10Y-DE.BD',  # proxy ou remplacer par ETF
    'Italy 10Y': 'IT10Y-BD.BD',      # idem
    'UK 10Y': 'GILT10Y-GB.BD',       # idem
    'France 10Y': 'FR10Y-BD.BD',
    'Japan 10Y': 'JP10Y-BD.BD',

    # ===== PERFORMANCE OBLIGATAIRE (ETFs) =====
    '--- OBLIGATIONS — PERFORMANCE PRIX (ETFs) ---': None,
    'US Treasuries Long (20Y+)': 'TLT',
    'US Treasuries Court Terme (1–3Y)': 'SHY',

    # ===== CREDIT =====
    '--- CRÉDIT IG / HY ---': None,
    'IG Credit US (LQD)': 'LQD',
    'IG Credit EUR (IEAC)': 'IEAC',
    'HY Credit US (HYG)': 'HYG',
    'HY Credit EUR (EHYG)': 'EHYG',

    # ===== TAUX DE RÉFÉRENCE =====
    '--- TAUX DE RÉFÉRENCE ---': None,
    'Fed Funds Effective Rate': '^IRX',
    'ESTER (proxy ETF)': 'EONIA-ETF.PA',  # proxy si indisponible, sinon à simuler

    # ===== COMMODITIES =====
    '--- COMMODITIES  ---': None,
    'BCOM' :'^BCOM',
    'Gold': 'GC=F',
    'Copper': 'HG=F',
    'Brent Oil': 'BZ=F',

    # ===== FX =====
    '--- FOREX — DEVISES MAJEURES ---': None,
    'EUR/USD': 'EURUSD=X',
    'USD/JPY': 'JPY=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/CHF': 'CHF=X',
    'USD/CNH': 'CNH=X'}


def get_market_snapshot(tickers=TICKERS):
    """
    Récupère les données de marché live via yfinance.
    Retourne un DataFrame avec niveau actuel, variation, volatilité et signaux booléens.
    """

    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=40)

    # Téléchargement des prix de clôture
    tickers_valid = [t for t in tickers.values() if t is not None]
    df_all = yf.download(tickers_valid, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))['Close']

    # Forcer DataFrame si une seule colonne
    if isinstance(df_all, pd.Series):
        df_all = df_all.to_frame()

    snapshot = []
    failed_tickers = []  # Liste des tickers échoués

    for label, ticker in tickers.items():
         # Si c’est une ligne de séparation (catégorie), on ajoute une ligne vide avec juste le nom
        if ticker is None:
            snapshot.append({
                'Actif': label,
                'Niveau': None,
                '∆ 1j': None,
                '∆ 5j': None,
                '∆ 1m': None,
                'Vol réalisée (%)': None,
                'Signal global': None
            })
            continue  # On passe au label suivant

        try:
            prices = df_all[ticker].dropna()
            latest = prices.iloc[-1]
            delta_1d = latest - prices.iloc[-2] if len(prices) >= 2 else None
            delta_5d = latest - prices.iloc[-6] if len(prices) >= 6 else None
            delta_1m = latest - prices.iloc[0] if len(prices) >= 20 else None
            vol_real = prices.pct_change().std() * (252 ** 0.5) * 100

            # Calcul des variations %
            pct_1d = (delta_1d / prices.iloc[-2]) * 100 if delta_1d else None
            pct_5d = (delta_5d / prices.iloc[-6]) * 100 if delta_5d else None

            # Signaux booléens
            vol_elevee = vol_real > 15 if vol_real else False
            vol_faible = vol_real < 5 if vol_real else False
            baisse_1j = pct_1d < -1.5 if pct_1d else False
            hausse_1j = pct_1d > 1.5 if pct_1d else False
            baisse_5j = pct_5d < -3 if pct_5d else False
            hausse_5j = pct_5d > 3 if pct_5d else False

            # Signal global
            signal = []
            if vol_elevee:
                signal.append("🔺 Volatilité élevée")
            elif vol_faible:
                signal.append("🟢 Marché calme")
            if baisse_1j:
                signal.append("🔻 Baisse 1j")
            elif hausse_1j:
                signal.append("📈 Hausse 1j")
            if baisse_5j:
                signal.append("📉 Baisse 5j")
            elif hausse_5j:
                signal.append("🚀 Hausse 5j")
            signal_str = " | ".join(signal)

            snapshot.append({
                'Actif': label,
                'Niveau': round(latest, 4),
                '∆ 1j': round(delta_1d, 4) if delta_1d else None,
                '∆ 5j': round(delta_5d, 4) if delta_5d else None,
                '∆ 1m': round(delta_1m, 4) if delta_1m else None,
                'Vol réalisée (%)': round(vol_real, 2) if vol_real else None,
                ' Vol élevé': vol_elevee,
                ' Vol faible': vol_faible,
                ' Baisse 1j': baisse_1j,
                ' Hausse 1j': hausse_1j,
                ' Baisse 5j': baisse_5j,
                ' Hausse 5j': hausse_5j,
                'Signal global': signal_str
            })

        except Exception as e:
            # En cas d’erreur : None sur les colonnes numériques, texte sur le signal
            snapshot.append({
                'Actif': label,
                'Niveau': None,
                '∆ 1j': None,
                '∆ 5j': None,
                '∆ 1m': None,
                'Vol réalisée (%)': None,
                ' Vol élevé': False,
                ' Vol faible': False,
                ' Baisse 1j': False,
                ' Hausse 1j': False,
                ' Baisse 5j': False,
                ' Hausse 5j': False,
                'Signal global': 'Erreur'
            })
            failed_tickers.append(label)

    # Affiche dans le terminal la liste des tickers échoués
    if failed_tickers:
        print(" Échec de téléchargement pour :", ", ".join(failed_tickers))

    return pd.DataFrame(snapshot)


def style_market_table (df):
    def highlight_row (row):
        #Style par défaut
        style = ['']*len(row)

        if row['Actif'].startswith('___'):
            return ['background-color: #e0e0e0; font-weight: bold'] * len(row)  # ligne catégorie

        # Baisse 1j importante
        if isinstance(row['∆ 1j'], (float, int)) and row['∆ 1j'] < -1.5:
            style[df.columns.get_loc('∆ 1j')] = 'color: red; font-weight: bold'

        # Hausse 1j importante
        if isinstance(row['∆ 1j'], (float, int)) and row['∆ 1j'] > 1.5:
            style[df.columns.get_loc('∆ 1j')] = 'color: green; font-weight: bold'

        # Volatilité élevée
        if isinstance(row['Vol réalisée (%)'], (float, int)) and row['Vol réalisée (%)'] > 15:
            style[df.columns.get_loc('Vol réalisée (%)')] = 'color: orange; font-weight: bold'

        return style

    return df.style.apply(highlight_row, axis=1)