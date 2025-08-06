from doctest import run_docstring_examples
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import io
import xlsxwriter

from Modules.Pricing import cap_price, digital_option_price, forward_fx_price, black_scholes_price,caplet_price,forward_rate_zero_coupon,build_volatility_curve, load_volatility_curve,floorlet_price,floor_price,read_csv_auto_sep



st.set_page_config(page_title="Sales Intelligence App", layout="wide")

st.title(" Sales Intelligence App")
st.write("Bienvenue sur ton app d'aide à la vente en salle des marchés ")

st.success("Tu as bien lancé Streamlit !")

# Configuration de la page 
st.set_page_config(page_title="Sales Intelligence App",layout="wide")

# Menu de navigation de la sidebar
st.sidebar.title("Menu") # Créer un titre dans la barre latérale
menu_selection = st.sidebar.radio("Aller vers...",("Accueil", "Pricing", "Marché", "Clients", "Pitch")) # Affiche un menu à choix unique (boutons radio). La variable menu_selection contriendra le nom de l'onglet choisi

#Affichage du contenu selon l'option sélectionnée
if menu_selection == "Accueil":
    st.title("Sales Intelligence APP") 
    st.write("Bienvenue sur ton app d'aide à la vente en salle des marchés") #

elif menu_selection == "Pricing":
    st.title("Module de Pricing")
    st.write("Ici tu pourras pricer des produits financiers simples.")
    product = st.selectbox("Choisir un produit à pricer", ["Option Européenne", "Forward FX","Option Digitale","Cap de taux","Floor de taux","Forward rate (zero coupon)"])#rajouter Caplet de taux si nécessaire

    if product == "Option Européenne":
        st.markdown(" Pricing de plusieurs options européennes via fichier CSV")
        csv_file_euro = st.file_uploader("Uploader un fichier CSV contenant les options européennes à pricer", type=["csv"], key="csv_euro_upload")

        if csv_file_euro is not None:
            try:
                df = pd.read_csv(csv_file_euro)
                expected_cols = {"s", "k", "t", "r", "sigma", "option_type"}
                df.columns = df.columns.str.strip().str.lower()

                if not expected_cols.issubset(set(df.columns)):
                    st.error(f"Le fichier doit contenir les colonnes suivantes : {sorted(expected_cols)}")
                else:
                    prices = []
                    for idx, row in df.iterrows():
                        try:
                            if pd.isna(row["option_type"]) or row["option_type"].lower() not in ["call", "put"]:
                                raise ValueError("option_type invalide (doit être 'call' ou 'put')")

                            price = black_scholes_price(S=row["s"],K=row["k"],T=row["t"],r=row["r"],sigma=row["sigma"],option_type=row["option_type"].lower())
                            prices.append(price)
                        except Exception as e:
                            prices.append(None)  
                            st.warning(f"Erreur sur une ligne : {e}")

                    df["prix_option_europeenne"] = prices
                    st.success("Options européennes pricées avec succès !")
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Erreur de lecture ou de traitement du fichier : {e}")

        option_type = st.selectbox("type d'option",["call","put"]) #choix du type d'option grâce à selectbox
        S = st.number_input("Prix spot (S)", value = 100.0,key="euro_S") #imputs utilisateurs
        K =st.number_input("Strike (K)", value = 100.0,key="euro_K")
        T = st.number_input("Maturité (en années)", value = 1.0,key="euro_T")
        r = st.number_input("Taux sans risque (r), ex : 0.03 pour 3%", value = 0.03,key="euro_r")
        sigma = st.number_input("volatilité implicite (σ),ex : 0.2 pour 20%)", value = 0.20,key="euro_sigma")

        if st.button("Calculer le prix de l'option"): #bouton "calculer" --> st.button
            price = black_scholes_price(S,K,T,r,sigma,option_type)
            st.success(f"Prix de l'option {option_type.upper()}:{price:.2f}€") #price est un flottant --> .2f permet de garder 2 chiffres après la virgule.
    
    elif product == "Forward FX":
        st.markdown("Pricing de plusieurs Forwards FX via fichier CSV")
        csv_file_fx = st.file_uploader("Uploader un fichier CSV contenant les données Forward FX à pricer", type=["csv"], key="csv_fx_upload")

        if csv_file_fx is not None:
            try:
                df = pd.read_csv(csv_file_fx)
                expected_cols = {"s", "r_dom", "r_for", "t"}
                df.columns = df.columns.str.strip().str.lower()

                if not expected_cols.issubset(set(df.columns)):
                    st.error(f"Le fichier doit contenir les colonnes suivantes : {sorted(expected_cols)}")
                else:
                    forwards = []
                    for idx, row in df.iterrows():
                        try:
                            fwd = forward_fx_price(S=row["s"],r_dom=row["r_dom"],r_for=row["r_for"],T=row["t"])
                            forwards.append(fwd)
                        except Exception as e:
                            forwards.append(None)
                            st.warning(f"Erreur sur la ligne {idx+1} : {e}")

                    df["taux_forward"] = forwards
                    st.success("Taux forwards calculés avec succès !")
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Erreur de lecture ou de traitement du fichier : {e}")

        st.markdown("---")
        st.markdown("Saisie manuelle d'un Forward FX")

        S = st.number_input("Taux spot (S)", value=1.10, key="fx_S")
        r_dom = st.number_input("Taux domestique (r_dom)", value=0.03, format="%.4f", key="fx_r_dom")
        r_for = st.number_input("Taux étranger (r_for)", value=0.01, format="%.4f", key="fx_r_for")
        T = st.number_input("Maturité (en années)", value=0.5, key="fx_T")

        if st.button("Calculer le taux forward", key="fx_btn_calc"):
            forward = forward_fx_price(S, r_dom, r_for, T)
            st.success(f"Taux forward à {T:.2f} an(s) : {forward:.4f}")

            S = st.number_input("Taux spot (S)",value = 1.10,key="fx_S")
            r_dom = st.number_input("Taux domestique (r_dom)",value =0.03,format ="%.4f",key="fx_r_dom")
            r_for = st.number_input("Taux étranger (r_for)",value = 0.01, format="%.4f",key="fx_r_for")
            T = st.number_input("Maturité (en années),",value =0.5,key="fx_T")

            if st.button ("Calculer le taux forward"):
                forward = forward_fx_price(S,r_dom,r_for,T)
                st.success(f"Taux forward à {T:.2f} an(s) : {forward:.4f}")
    
    elif product == "Option Digitale":
        option_type=st.selectbox("type d'option",["call","put"]) #choix du type d'option

        st.markdown("Pricing de plusieurs options digitales via fichier CSV")

        csv_file = st.file_uploader("Uploader un fichier CSV contenant les options digitales à pricer", type=["csv"], key="csv_digital_upload")

        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file) #on charge le fichier csv
                expected_cols = {"s", "k", "t", "r", "sigma", "payoff", "option_type"} #on définit les colonnes attendues
                df.columns = df.columns.str.strip().str.lower() #on normalise les noms de colonnes (nettoyage des espaces et des majuscules)
                if not expected_cols.issubset(df.columns.str.lower()): #on vérifie que toutes les colonnes attendues sont présentes
                    st.error(f"Le fichier doit contenir les colonnes suivantes : {sorted(expected_cols)}")
                else:
                    df.columns = df.columns.str.lower() #on normalise les noms de colonnes (nettoyage des espaces et des majuscules)    
                    prices = [] #on initialise la liste des prix
                    for idx, row in df.iterrows(): #on parcourt chaque ligne du dataframe   
                        st.write(row) #debug
                        try:
                            if pd.isna(row["option_type"]) or row["option_type"].lower() not in ["call", "put"]: #on vérifie que le type d'option est valide
                                raise ValueError("option_type_invalide (doit être 'call' ou 'put')") #on lève une erreur si le type d'option est invalide
                            price = digital_option_price(S=row["s"],K=row["k"],T=row["t"],r=row["r"],sigma=row["sigma"],option_type=row["option_type"].lower(),payoff=row["payoff"]) #on price l'option
                        except Exception as e: #on gère les erreurs
                            price = f"Erreur: {e}" #on affiche l'erreur
                        prices.append(price) #on ajoute le prix à la liste des prix

                st.write(prices) #debug
                df["prix_option_digital"] = prices #on ajoute la colonne des prix à la dataframe
                st.success("Options pricées avec succès !") 
                st.dataframe(df)
            except Exception as e:
                st.error(f"Erreur de lecture ou de traitement du fichier : {e}")

        S = st.number_input("Prix spot (S)",value = 100,key="digi_S" )
        K = st.number_input("Strike (K)", value =  100,key="digi_K")
        T = st.number_input("Maturité (en années)",value = 1.0,key="digi_T")
        r = st.number_input ("Taux sans risque (r), ex : 0.03 pour 3%",value =0.03,format="%.4f",key="digi_r")
        sigma = st.number_input("volatilité implicite (σ),ex : 0.2 pour 20%)", value = 0.20,format="%.4f",key="digi_sigma")
        payoff = st.number_input("Payoff fixe (€)",value = 10.0,key="digi_payoff")

        if st.button("calculer le prix de l'option digitale"):
            price = digital_option_price (S,K,T,r,sigma,option_type,payoff)
            st.success(f"Prix de l'option{option_type.upper()}:{price:.2f}€")
    

    elif product == "Caplet de taux" :
        F = st.number_input("Taux forward (F)",value = 0.02,format = "%.4f")
        K =st.number_input ("Strike (K)",value = 0.025,format ="%.4f")
        T =st.number_input ("Temps jusqu'au paiement (T, en années)", value=1.0)
        sigma = st.number_input ("volatilité implicite (σ),ex : 0.2 pour 20%)", value = 0.20,format="%.4f")
        r = st.number_input ("Taux sans risque (r),ex : 0.03 pour 3%",value =0.03,format="%.4f")
        delta = st.number_input ("Période en années (δ)", value=0.5, format="%.2f")
        notional = st.number_input ("Notional (€)", value=1_000_000) 

        if st.button("Calculer le prix du cap de taux"):
            price = caplet_price(F,K,T,sigma,r,delta,notional)
            st.success(f"Prix du caplet : {price:,.2f} €") #st.success affiche un message en vert dans streamlit. {price}= variable à afficher.le f devant permet de transformer la chaine de caractère en f-sting ce qui permet de replacer le mot price par sa valeur et non de renvoyer seulement le mot "price"
    
    elif product == "Forward rate (zero coupon)":
        st.subheader("Calcul de plusieurs taux forwards via fichier CSV")

        csv_file_forward = st.file_uploader("Uploader un fichier CSV avec les colonnes 'r0', 'r1', 'T0', 'T1'", type=["csv"], key="csv_fw_rate")

        if csv_file_forward is not None:
            try:
                df = pd.read_csv(csv_file_forward)
                expected_cols = {"r0", "r1", "t0", "t1"}
                df.columns = df.columns.str.strip().str.lower()

                if not expected_cols.issubset(set(df.columns)):
                    st.error(f"Le fichier doit contenir les colonnes suivantes : {sorted(expected_cols)}")
                else:
                    forwards = []
                    for idx, row in df.iterrows():
                        try:
                            fwd = forward_rate_zero_coupon(row["r0"], row["r1"], row["t0"], row["t1"])
                            forwards.append(fwd)
                        except Exception as e:
                            forwards.append(None)
                            st.warning(f"Erreur ligne {idx+1} : {e}")

                    df["taux_forward"] = forwards
                    st.success("Taux forwards calculés avec succès !")
                    st.dataframe(df)

                    results =[]
                    if results:
                        df_resultats = pd.DataFrame(results)
                        st.success("Pricing terminé avec succès !")
                        st.dataframe(df_resultats)
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            df_resultats.to_excel(writer, sheet_name='Caps Pricing', index=False)
                            writer.save()
                            processed_data = output.getvalue()

                        st.download_button(label="Télécharger les résultats au format Excel",data=processed_data,file_name="resultats_caps.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            except Exception as e:
                st.error(f"Erreur de lecture ou de traitement du fichier : {e}")

    st.markdown("---")
    st.subheader("Saisie manuelle d’un taux forward")

    r0 = st.number_input("Taux zéro-coupon T0 (r0)", value=0.02, format="%.4f", key="fw_rate_r0_bis")
    r1 = st.number_input("Taux zéro-coupon T1 (r1)", value=0.025, format="%.4f", key="fw_rate_r1")
    T0 = st.number_input("Maturité T0 (en années)", value=1.0, key="fw_rate_T0")
    T1 = st.number_input("Maturité T1 (en années)", value=1.5, key="fw_rate_T1")

    if st.button("Calculer le taux forward", key="fw_manual_btn"):
        try:
            forward = forward_rate_zero_coupon(r0, r1, T0, T1)
            st.success(f"Taux forward entre T0={T0} et T1={T1} : {forward:.4%}")
        except ValueError as e:
            st.error(str(e))

        st.subheader("Calcul du taux forward à partir de taux zéro-coupon")

        r0 = st.number_input("Taux zéro-coupon T0 (r0)", value=0.02, format="%.4f",key="fw_rate_r0")
        r1 = st.number_input("Taux zéro-coupon T1 (r1)", value=0.025, format="%.4f",key="fw_rate_r1")
        T0 = st.number_input("Maturité T0 (en années)", value=1.0,key="fw_rate_T0")
        T1 = st.number_input("Maturité T1 (en années)", value=1.5,key="fw_rate_T1")

        if st.button("Calculer le taux forward"):
            try:
                forward = forward_rate_zero_coupon(r0, r1, T0, T1)
                st.success(f"Taux forward entre T0={T0} et T1={T1} : {forward:.4%}")
            except ValueError as e:
                st.error(str(e))



    elif product == "Cap de taux":
        st.subheader("Pricing d'un Cap ") #faire apparaitre un sous-tire
        col1,col2 = st.columns(2) #on divise l'espace horizontal en 2 colonnes
        
        with col1:
            st.markdown("Courbes de volatilité pour plusieurs Caps (CSV)")
            vol_file_multi = st.file_uploader("Uploader un fichier CSV avec colonnes : zc_maturities, zc_rates", type="csv", key="vol_file_multi")

            vol_df_multi = None
            if vol_file_multi is not None:
                try:
                    vol_df_multi = read_csv_auto_sep(vol_file_multi)
                    vol_df_multi.columns = vol_df_multi.columns.str.strip().str.lower()

                    # Mapping des colonnes attendues
                    col_mapping = {
                        'k': None,
                        'r': None,
                        'notional': None,
                        'delta': None,
                        'maturities': None,
                        'vols': None
                    }

                    for col in vol_df_multi.columns:
                        if 'k' == col or 'strike' in col: col_mapping['k'] = col
                        if col in ['r', 'rate', 'riskfree']: col_mapping['r'] = col
                        if 'notional' in col: col_mapping['notional'] = col
                        if 'delta' in col: col_mapping['delta'] = col
                        if 'maturities' in col and 'vol' in col: col_mapping['maturities'] = col
                        elif col == 'maturities': col_mapping['maturities'] = col
                        if 'vol' in col and 'rate' not in col: col_mapping['vols'] = col

                    missing = [k for k, v in col_mapping.items() if v is None]
                    if missing:
                        st.error(f"Colonnes manquantes dans le fichier : {missing}")
                        vol_df_multi = None
                    else:
                        vol_df_multi = vol_df_multi.rename(columns={v: k for k, v in col_mapping.items()})
                        st.success("Fichier de volatilités chargé avec succès.")
                        st.dataframe(vol_df_multi)

                        if st.checkbox("Afficher les courbes de volatilité", key="show_vol_curves"):
                            fig = go.Figure()
                            for idx, row in vol_df_multi.iterrows():
                                try:
                                    mats = [float(x) for x in str(row["maturities"]).split(';')]
                                    vols = [float(x) for x in str(row["vols"]).split(';')]
                                    fig.add_trace(go.Scatter(x=mats, y=vols, mode='lines+markers', name=f"Cap {idx+1}"))
                                except:
                                    st.warning(f"Erreur de parsing des volatilités pour la ligne {idx+1}")

                            fig.update_layout(title="Courbes de volatilité par Cap",
                                            xaxis_title="Maturité (années)",
                                            yaxis_title="Volatilité (%)",
                                            legend_title="Caps",
                                            hovermode="x unified")
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur de lecture du fichier : {e}")
            else :
                st.info("Aucune courbe de volatilité chargée. Veuillez saisir une volatilité constante manuellement.")

            notional = st.number_input("Notional (€)", value=1_000_000, key="cap_taux_notional")
            K = st.number_input("Strike du cap (%)", value=0.02, format="%.4f", key="cap_taux_K")
            r = st.number_input("Taux sans risque (r)", value=0.03, format="%.4f", key="cap_taux_r")
            sigma = st.number_input("Volatilité (%)", value=0.20, format="%.4f", disabled=(vol_file_multi is not None), key="cap_taux_sigma")
            T_total = st.number_input("Maturité totale (en années)", value=2.0, key="cap_taux_T_total")
            delta = st.number_input("Fréquence des flux (en années)", value=0.5, key="cap_taux_delta")
            n_periods = int(T_total / delta)


        with col2 :
            st.markdown("Courbes des taux zéro-coupon pour plusieurs Caps (CSV)")
            zc_file_multi = st.file_uploader("Uploader un fichier CSV avec colonnes : zc_maturities, zc_rates", type="csv", key="zc_file_multi")

            zc_df_multi = None
            if zc_file_multi is not None:
                try:
                    zc_df_multi = read_csv_auto_sep(zc_file_multi)
                    zc_df_multi.columns = zc_df_multi.columns.str.strip().str.lower()
                    zc_df_multi.columns = [col.replace('\ufeff', '') for col in zc_df_multi.columns]  # retire BOM si présent

                    # Mapping intelligent des noms de colonnes
                    col_mapping_zc = {'zc_maturities': None,'zc_rates': None}

                    for col in zc_df_multi.columns:
                        if 'maturities' in col and 'zc' in col: col_mapping_zc['zc_maturities'] = col
                        elif 'maturities' in col: col_mapping_zc['zc_maturities'] = col
                        if 'rate' in col and 'zc' in col: col_mapping_zc['zc_rates'] = col
                        elif 'zc_rates' in col: col_mapping_zc['zc_rates'] = col

                    missing = [k for k, v in col_mapping_zc.items() if v is None]
                    if missing:
                        st.error(f"Colonnes manquantes dans le fichier : {missing}")
                        zc_df_multi = None
                    else:
                        zc_df_multi = zc_df_multi.rename(columns={v: k for k, v in col_mapping_zc.items()})
                        st.success("Fichier de taux ZC chargé avec succès.")
                        st.dataframe(zc_df_multi)

                        if st.checkbox("Afficher les courbes de taux ZC", key="show_zc_curves"):
                            fig = go.Figure()
                            for idx, row in zc_df_multi.iterrows():
                                try:
                                    mats = [float(x) for x in str(row["zc_maturities"]).split(';')]
                                    rates = [float(x) for x in str(row["zc_rates"]).split(';')]
                                    fig.add_trace(go.Scatter(x=mats, y=rates, mode='lines+markers', name=f"Cap {idx+1}"))
                                except Exception as e:
                                    st.warning(f"Erreur de parsing pour la ligne {idx+1} : {e}")

                            fig.update_layout(title="Courbes des taux ZC par Cap",xaxis_title="Maturité (années)",yaxis_title="Taux ZC (%)",legend_title="Caps",hovermode="x unified")
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur de lecture du fichier : {e}")

            else:
                zc_rates = []
                zc_maturities = []
                for i in range(n_periods + 1):
                    maturity = delta * i
                    rate = st.number_input(f"Taux zéro-coupon pour T={maturity:.2f} ans", value=0.02 + 0.001 * i, format="%.4f", key=f"zc_{i}")
                    zc_maturities.append(maturity)
                    zc_rates.append(rate)

                
        if st.button("Calculer le prix du cap"):
            if vol_df_multi is None or zc_df_multi is None:
                st.error("Veuillez d'abord uploader les deux fichiers : volatilité et taux ZC.")
            elif len(vol_df_multi) != len(zc_df_multi):
                st.error("Le nombre de lignes dans les deux fichiers doit être identique (un cap = une ligne dans chaque fichier).")
            else:
                results = []
                for idx in range(len(vol_df_multi)):
                    try:
                        # Extraction des inputs
                        K = float(vol_df_multi.loc[idx, 'k'])
                        r = float(vol_df_multi.loc[idx, 'r'])
                        notional = float(vol_df_multi.loc[idx, 'notional'])
                        delta = float(vol_df_multi.loc[idx, 'delta'])

                        zc_maturities = [float(x) for x in str(zc_df_multi.loc[idx, 'zc_maturities']).split(';')]
                        zc_rates = [float(x) for x in str(zc_df_multi.loc[idx, 'zc_rates']).split(';')]

                        maturities = [delta * (i + 1) for i in range(len(zc_maturities) - 1)]

                        forwards = []
                        for i in range(len(zc_maturities) - 1):
                            T0 = zc_maturities[i]
                            T1 = zc_maturities[i + 1]
                            r0 = zc_rates[i]
                            r1 = zc_rates[i + 1]
                            fwd = forward_rate_zero_coupon(r0, r1, T0, T1)
                            forwards.append(fwd)

                        vol_maturities = [float(x) for x in str(vol_df_multi.loc[idx, 'maturities']).split(';')]
                        vols = [float(x) for x in str(vol_df_multi.loc[idx, 'vols']).split(';')]
                        vol_func = build_volatility_curve(vol_maturities, vols)
                        sigmas = [vol_func(t) for t in maturities]

                        price = cap_price(forwards, K, maturities, sigmas, r, delta, notional)
                        vol_moyenne = np.mean(sigmas)

                        results.append({"Cap ID": idx + 1,"Prix (€)": round(price, 2),"Strike": K,"Notional": notional,"Volatilité Moy.": round(vol_moyenne, 4),"Taux sans risque": r,"Période": delta,"Nb caplets": len(maturities)})

                    except Exception as e:
                        st.warning(f"Erreur sur le cap {idx + 1} : {e}")

                if results:
                    df_resultats = pd.DataFrame(results)
                    st.success("Pricing terminé avec succès !")
                    st.dataframe(df_resultats)

                    excel_output = io.BytesIO()
                    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                        df_resultats.to_excel(writer, sheet_name='Caps Pricing', index=False)
                    excel_output.seek(0)

                    # Export CSV
                    csv_output = df_resultats.to_csv(index=False).encode('utf-8')

                    # Deux boutons de téléchargement
                    st.download_button(label="Télécharger en Excel",data=excel_output,file_name="resultats_caps.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                    st.download_button(label="Télécharger en CSV",data=csv_output,file_name="resultats_caps.csv",mime="text/csv")
        

        st.markdown("---")
        st.markdown("### 🔧 Zone de test des boutons")

        if st.button("Test bouton affichage", key="test_button_1"):
            st.success("✅ Le bouton a bien été cliqué !")

        if st.button("Test bouton avec calcul", key="test_button_2"):
            résultat = 123 + 456
            st.info(f"Résultat du calcul : {résultat}")

       
    elif product == "Floor de taux":
        st.subheader("Pricing de plusieurs Floors")
        col1, col2 = st.columns(2)

        # === COLONNE DE GAUCHE : VOLATILITÉ ===
        with col1:
            st.markdown("Courbes de volatilité pour plusieurs Floors (CSV)")
            # Upload du fichier CSV de volatilité
            vol_file_floor = st.file_uploader("Uploader un fichier CSV avec colonnes : k, r, notional, delta, maturities, vols", type="csv", key="floor_vol_file_multi")

            vol_df_floor = None
            if vol_file_floor is not None:
                try:
                    # Lecture du fichier avec tentative auto du séparateur
                    vol_df_floor = read_csv_auto_sep(vol_file_floor)
                    vol_df_floor.columns = vol_df_floor.columns.str.strip().str.lower()

                    # Mapping intelligent des noms de colonnes
                    col_mapping = {'k': None,'r': None,'notional': None,'delta': None,'maturities': None,'vols': None}

                    # On détecte dynamiquement chaque colonne clé
                    for col in vol_df_floor.columns:
                        if 'k' == col or 'strike' in col: col_mapping['k'] = col
                        if col in ['r', 'rate', 'riskfree']: col_mapping['r'] = col
                        if 'notional' in col: col_mapping['notional'] = col
                        if 'delta' in col: col_mapping['delta'] = col
                        if 'maturities' in col and 'vol' in col: col_mapping['maturities'] = col
                        elif col == 'maturities': col_mapping['maturities'] = col
                        if 'vol' in col and 'rate' not in col: col_mapping['vols'] = col

                    # Vérifie si toutes les colonnes nécessaires sont là
                    missing = [k for k, v in col_mapping.items() if v is None]
                    if missing:
                        st.error(f"Colonnes manquantes dans le fichier : {missing}")
                        vol_df_floor = None
                    else:
                        # Renommage standardisé des colonnes
                        vol_df_floor = vol_df_floor.rename(columns={v: k for k, v in col_mapping.items()})
                        st.success("Fichier de volatilités chargé avec succès.")
                        st.dataframe(vol_df_floor)

                        # Affichage graphique de chaque courbe de vol
                        if st.checkbox("Afficher les courbes de volatilité", key="floor_show_vol_curves"):
                            fig = go.Figure()
                            for idx, row in vol_df_floor.iterrows():
                                try:
                                    mats = [float(x) for x in str(row["maturities"]).split(';')]
                                    vols = [float(x) for x in str(row["vols"]).split(';')]
                                    fig.add_trace(go.Scatter(x=mats, y=vols, mode='lines+markers', name=f"Floor {idx+1}"))
                                except:
                                    st.warning(f"Erreur de parsing des volatilités pour la ligne {idx+1}")

                            fig.update_layout(title="Courbes de volatilité par Floor",xaxis_title="Maturité (années)",yaxis_title="Volatilité (%)",legend_title="Floors",hovermode="x unified")
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur de lecture du fichier : {e}")

        # === COLONNE DE DROITE : COURBES ZC ===
        with col2:
            st.markdown("Courbes des taux zéro-coupon pour plusieurs Floors (CSV)")
            zc_file_floor = st.file_uploader("Uploader un fichier CSV avec colonnes : zc_maturities, zc_rates", type="csv", key="floor_zc_file_multi")

            zc_df_floor = None
            if zc_file_floor is not None:
                try:
                    # Lecture du fichier et nettoyage des colonnes
                    zc_df_floor = read_csv_auto_sep(zc_file_floor)
                    zc_df_floor.columns = zc_df_floor.columns.str.strip().str.lower()
                    zc_df_floor.columns = [col.replace('\ufeff', '') for col in zc_df_floor.columns]

                    # Mapping intelligent des colonnes
                    col_mapping_zc = {'zc_maturities': None, 'zc_rates': None}
                    for col in zc_df_floor.columns:
                        if 'maturities' in col and 'zc' in col: col_mapping_zc['zc_maturities'] = col
                        elif 'maturities' in col: col_mapping_zc['zc_maturities'] = col
                        if 'rate' in col and 'zc' in col: col_mapping_zc['zc_rates'] = col
                        elif 'zc_rates' in col: col_mapping_zc['zc_rates'] = col

                    # Vérifie les colonnes
                    missing = [k for k, v in col_mapping_zc.items() if v is None]
                    if missing:
                        st.error(f"Colonnes manquantes dans le fichier : {missing}")
                        zc_df_floor = None
                    else:
                        # Renommage
                        zc_df_floor = zc_df_floor.rename(columns={v: k for k, v in col_mapping_zc.items()})
                        st.success("Fichier de taux ZC chargé avec succès.")
                        st.dataframe(zc_df_floor)

                        # Affichage graphique interactif
                        if st.checkbox("Afficher les courbes de taux ZC", key="floor_show_zc_curves"):
                            fig = go.Figure()
                            for idx, row in zc_df_floor.iterrows():
                                try:
                                    mats = [float(x) for x in str(row["zc_maturities"]).split(';')]
                                    rates = [float(x) for x in str(row["zc_rates"]).split(';')]
                                    fig.add_trace(go.Scatter(x=mats, y=rates, mode='lines+markers', name=f"Floor {idx+1}"))
                                except Exception as e:
                                    st.warning(f"Erreur de parsing pour la ligne {idx+1} : {e}")

                            fig.update_layout(title="Courbes des taux ZC par Floor",xaxis_title="Maturité (années)",yaxis_title="Taux ZC (%)",legend_title="Floors",hovermode="x unified")
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur de lecture du fichier : {e}")

        # === PRICING DES FLOORS ===
        if st.button("Calculer les prix des Floors", key="floor_batch_button"):
            if vol_df_floor is None or zc_df_floor is None:
                st.error("Veuillez uploader les deux fichiers : volatilité et taux ZC.")
            elif len(vol_df_floor) != len(zc_df_floor):
                st.error("Le nombre de lignes dans les deux fichiers doit être identique (un floor = une ligne dans chaque fichier).")
            else:
                results = []
                for idx in range(len(vol_df_floor)):
                    try:
                        # Extraction des inputs du fichier de volatilité
                        K = float(vol_df_floor.loc[idx, 'k'])
                        r = float(vol_df_floor.loc[idx, 'r'])
                        notional = float(vol_df_floor.loc[idx, 'notional'])
                        delta = float(vol_df_floor.loc[idx, 'delta'])

                        # Reconstruction des taux ZC
                        zc_maturities = [float(x) for x in str(zc_df_floor.loc[idx, 'zc_maturities']).split(';')]
                        zc_rates = [float(x) for x in str(zc_df_floor.loc[idx, 'zc_rates']).split(';')]
                        maturities = [delta * (i + 1) for i in range(len(zc_maturities) - 1)]

                        # Calcul des taux forwards implicites
                        forwards = []
                        for i in range(len(zc_maturities) - 1):
                            T0 = zc_maturities[i]
                            T1 = zc_maturities[i + 1]
                            r0 = zc_rates[i]
                            r1 = zc_rates[i + 1]
                            fwd = forward_rate_zero_coupon(r0, T0, r1, T1)
                            forwards.append(fwd)

                        # Interpolation de la courbe de vol
                        vol_maturities = [float(x) for x in str(vol_df_floor.loc[idx, 'maturities']).split(';')]
                        vols = [float(x) for x in str(vol_df_floor.loc[idx, 'vols']).split(';')]
                        vol_func = build_volatility_curve(vol_maturities, vols)
                        sigmas = [vol_func(t) for t in maturities]

                        # Pricing du floor
                        price = floor_price(forwards, K, maturities, sigmas, r, delta, notional)
                        vol_moyenne = np.mean(sigmas)

                        # Stockage du résultat
                        results.append({"Floor ID": idx + 1,"Prix (€)": round(price, 2),"Strike": K,"Notional": notional,"Volatilité Moy.": round(vol_moyenne, 4),"Taux sans risque": r,"Période": delta,"Nb floorlets": len(maturities)})

                    except Exception as e:
                        st.warning(f"Erreur sur le floor {idx + 1} : {e}")

                # Affichage + Export
                if results:
                    df_resultats = pd.DataFrame(results)
                    st.success("Pricing terminé avec succès !")
                    st.dataframe(df_resultats)

                    # Export Excel
                    excel_output = io.BytesIO()
                    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                        df_resultats.to_excel(writer, sheet_name='Floors Pricing', index=False)
                    excel_output.seek(0)

                    # Export CSV
                    csv_output = df_resultats.to_csv(index=False).encode('utf-8')

                    st.download_button(label="Télécharger en Excel", data=excel_output, file_name="resultats_floors.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.download_button(label="Télécharger en CSV", data=csv_output, file_name="resultats_floors.csv", mime="text/csv")



elif menu_selection == "Marché":
    st.title("Dashboard de Marché")
    st.write("Données financières, courbes, volatilité...")

elif menu_selection == "Clients":
    st.title("Gestion des Clients")
    st.write("Base client, préférences produits, historique d’idées.")

elif menu_selection == "Pitch":
    st.title("Générateur de Pitch")
    st.write("Crée des pitchs PDF prêts à être envoyés.")
