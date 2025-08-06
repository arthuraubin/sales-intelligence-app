import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import plotly.express as px
from Modules.Generate_Speech import generate_pitch, generate_summary
from Modules.Pricing import cap_price, forward_rate_zero_coupon, build_volatility_curve, read_csv_auto_sep
from Modules.Export_pitch_excel import export_pitch_excel
import os

# Bouton retour à l'accueil
if st.button("🏠 Retour à l’accueil"):
    st.switch_page("APP.py")


st.title("Cap de taux")

col1, col2 = st.columns(2)
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
            df_resultats["Produit"] = "Cap"  # ✅ Ajoute une colonne "Produit" pour usage dans le pitch
            st.success("Pricing terminé avec succès !")
            st.dataframe(df_resultats)
            st.session_state["df_resultats_cap"] = df_resultats #pn stock la valeur 


            excel_output = io.BytesIO()
            with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
                df_resultats.to_excel(writer, sheet_name='Caps Pricing', index=False)
            excel_output.seek(0)

            # Export CSV
            csv_output = df_resultats.to_csv(index=False).encode('utf-8')

            # Deux boutons de téléchargement
            st.download_button(label="Télécharger en Excel",data=excel_output,file_name="resultats_caps.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.download_button(label="Télécharger en CSV",data=csv_output,file_name="resultats_caps.csv",mime="text/csv")

# === STRESS TEST DU CAP ===

if st.button("Lancer le stress test cap"):
    # Vérification préalable : le pricing doit avoir été effectué
    if vol_df_multi is None or zc_df_multi is None:
        st.error("Veuillez d'abord uploader les deux fichiers nécessaires.")
    else:
        try:
            st.subheader("Résultats du stress test")

            # On utilise les inputs récents (vol, zc)
            from Modules.Pricing import stress_test  # Assure-toi que la fonction stress_test est bien importée

            stress_results = stress_test(vol_df=vol_df_multi, zc_df=zc_df_multi, cap_or_floor="cap")
            

            df_stress = pd.DataFrame(stress_results)
            if "Cap ID" not in df_stress.columns:
                st.error("Erreur : 'Cap ID' manquant dans les résultats du stress test.")
                st.stop()

            st.session_state["df_stress_cap"] = df_stress  # On enregistre pour l'utiliser dans les autres blocs
            st.dataframe(df_stress)

            # === Export CSV du stress test ===
            csv_stress = df_stress.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger le stress test (CSV)",
                               data=csv_stress,
                               file_name="stress_test_cap.csv",
                               mime="text/csv")

            # === Export combiné Excel (Pricing + Stress Test) ===
            if 'df_resultats' in locals() or 'df_resultats' in globals():
                st.markdown("### Export combiné Excel")

                export_excel_output = io.BytesIO()
                with pd.ExcelWriter(export_excel_output, engine='xlsxwriter') as writer:
                    df_resultats.to_excel(writer, sheet_name="Pricing", index=False)
                    df_stress.to_excel(writer, sheet_name="Stress Test", index=False)

                export_excel_output.seek(0)

                st.download_button(label="Télécharger l'Excel (Pricing + Stress Test)",
                    data=export_excel_output,
                    file_name="cap_pricing_et_stress_test.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"Erreur lors du stress test : {e}")

# === AFFICHAGE GRAPHIQUE INTERACTIF ===
if "df_stress_cap" in st.session_state:
    df_stress = st.session_state["df_stress_cap"]

    st.markdown("### Représentation graphique des stress tests")

    # Choix entre écart absolu (€) ou sensibilité (%)
    graphe_mode = st.radio("Afficher :", ["Différence (€)", "Sensibilité (%)"], horizontal=True, key="graph_mode_cap")

    # Sélection dynamique des cap IDs disponibles
    if "Cap ID" not in df_stress.columns:
        st.warning("Aucune donnée de stress test disponible pour affichage.")
        st.stop()

    cap_ids = df_stress["Cap ID"].unique().tolist()

    # Initialisation si nécessaire
    if "selected_caps" not in st.session_state:
        st.session_state["selected_caps"] = cap_ids.copy()

    # Boutons tout sélectionner / désélectionner
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Tout sélectionner", key="select_all_caps"):
            st.session_state["selected_caps"] = cap_ids.copy()
    with col2:
        if st.button("Tout désélectionner", key="deselect_all_caps"):
            st.session_state["selected_caps"] = []

    # Multiselect dynamique pour sélectionner les caps à afficher
    selected_caps = st.multiselect("Choisir les Cap IDs à afficher :",
                                   options=cap_ids,
                                   default=st.session_state["selected_caps"],
                                   key="selected_caps")

    # Filtrage des données en fonction des sélections
    df_filtered = df_stress[df_stress["Cap ID"].isin(selected_caps)]

    y_col = "Différence (€)" if graphe_mode == "Différence (€)" else "Sensibilité (%)"


    # Création du graphe interactif
    fig = px.bar(
        df_filtered,
        x="Scénario",
        y=y_col,
        color="Cap ID",
        barmode="group",
        title=f"Impact du stress test par scénario ({y_col})",
        labels={"Cap ID": "Cap", "Scénario": "Scénario", y_col: y_col},
        height=500)
         
    fig.update_layout(xaxis_tickangle=-45, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


#Générer le Pitch 
if st.button(" Générer le pitch PDF"):
    if "df_resultats_cap" in st.session_state:
        df = st.session_state["df_resultats_cap"]

        # Génération automatique du résumé exécutif
        summary = generate_summary(df)

        # Génération du fichier PDF temporaire
        output_path = "pitch_temp.pdf"
        generate_pitch(
            client_name="Client X",
            df_products=df,
            summary=summary,
            output_path=output_path)

        st.success("📄 Pitch PDF généré avec succès.")

        # Bouton de téléchargement du fichier généré
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger le pitch PDF",
                data=f,
                file_name="Pitch_Client_X.pdf",
                mime="application/pdf"
            )
    else:
        st.warning("Aucun résultat de pricing trouvé. Veuillez d’abord lancer le calcul.")


st.markdown("---")
st.subheader("Export Excel enrichi")

if "df_resultats_cap" in st.session_state:
    df_products = st.session_state["df_resultats_cap"]
    df_stress = st.session_state.get("df_stress_cap", None)  # Stress test facultatif
    client_name = st.text_input("Nom du client pour l’export", "Client X")

    if st.button("⬇Exporter le pitch Excel"):
        output_path = export_pitch_excel(
            df_products=df_products,
            df_stress=df_stress,
            client_name=client_name)
        st.success(f"Export Excel généré avec succès : {os.path.basename(output_path)}")
        st.download_button(
            label="📥 Télécharger le fichier",
            data=open(output_path, "rb"),
            file_name=os.path.basename(output_path),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Veuillez d’abord générer les résultats de pricing pour exporter.")



