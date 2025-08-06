import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import plotly.express as px
from Modules.Pricing import floor_price, forward_rate_zero_coupon, build_volatility_curve, read_csv_auto_sep, stress_test

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")

st.title("📉 Floor de taux")

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
            st.session_state["last_inputs_floor"] = {"vol": vol_df_floor,"zc": zc_df_floor,"K": K,"delta": delta,"notional": notional} #session_state permet de stocker des variables entre 2 interactions dans l'app même après un clic sur un bouton. Quand j'appui sur lancer le stress test les inputs sont réinitialisés donc pour les garder il faut cette fonction
            st.session_state["df_resultats_floors"] = df_resultats

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

# Bouton de stress test à part (activable après le pricing)
if st.button("Lancer le stress test floor"):
    if "last_inputs_floor" not in st.session_state:
        st.error("Veuillez d'abord lancer le pricing des floors.")
    else:
        try:
            st.subheader("Résultats du stress test")

            inputs = st.session_state["last_inputs_floor"]
            stress_results = stress_test(vol_df=inputs["vol"],zc_df=inputs["zc"],cap_or_floor="floor")

            df_stress = pd.DataFrame(stress_results)
            st.session_state["df_stress"] = df_stress
            st.dataframe(df_stress)

            # Export CSV du stress test
            csv_stress = df_stress.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger le stress test (CSV)",
                               data=csv_stress,
                               file_name="stress_test_floor.csv",
                               mime="text/csv")
            
            # === Export combiné Excel (Pricing + Stress Test) ===
            if "df_resultats_floors" in st.session_state:
                st.markdown("### Export combiné Excel")

                export_excel_output = io.BytesIO()
                with pd.ExcelWriter(export_excel_output, engine='xlsxwriter') as writer:
                    # Onglet 1 : Pricing
                    st.session_state["df_resultats_floors"].to_excel(writer, sheet_name="Pricing", index=False)
                    # Onglet 2 : Stress Test
                    df_stress.to_excel(writer, sheet_name="Stress Test", index=False)

                    export_excel_output.seek(0)

                    st.download_button(
                    label="Télécharger l'Excel (Pricing + Stress Test)",
                    data=export_excel_output,
                    file_name="floor_pricing_et_stress_test.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        
        except Exception as e:
            st.error(f"Erreur lors du stress test : {e}")

                    # === Représentation graphique ===

# === Représentation graphique ===
if "df_stress" in st.session_state:
    df_stress = st.session_state["df_stress"]

    st.markdown("### Représentation graphique des stress tests")

    # Choix de l'affichage : différence ou sensibilité
    graphe_mode = st.radio("Afficher :", ["Différence (€)", "Sensibilité (%)"], horizontal=True, key="graph_mode")

    # Sélection des floors à afficher
    floor_ids = df_stress["Floor ID"].unique().tolist()

    # Initialise selected_floors si absent
    if "selected_floors" not in st.session_state:
        st.session_state["selected_floors"] = floor_ids.copy()

    # Boutons tout sélectionner / tout désélectionner
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Tout sélectionner"):
            st.session_state["selected_floors"] = floor_ids.copy()
    with col2:
        if st.button("Tout désélectionner"):
            st.session_state["selected_floors"] = []

    # Affichage du multiselect dynamique
    selected_floors = st.multiselect("Choisir les Floor IDs à afficher :",options=floor_ids,default=st.session_state["selected_floors"],key="selected_floors")

    # Filtrage
    df_filtered = df_stress[df_stress["Floor ID"].isin(selected_floors)]

    # Variable à tracer
    y_col = "Différence (€)" if graphe_mode == "Différence (€)" else "Sensibilité (%)"

    fig = px.bar(
        df_filtered,
        x="Scénario",
        y=y_col,
        color="Floor ID",
        barmode="group",
        title=f"Impact du stress test par scénario ({y_col})",
        labels={"Floor ID": "Floor", "Scénario": "Scénario", y_col: y_col},
        height=500)

    fig.update_layout(xaxis_tickangle=-45, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)