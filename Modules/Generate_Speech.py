from reportlab.lib.pagesizes import A4
import pandas as pd
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from datetime import datetime
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Image as RLImage  
import matplotlib.pyplot as plt
import tempfile  # pour créer un fichier temporaire d’image
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment, Font, PatternFill, NamedStyle
from openpyxl.worksheet.table import Table, TableStyleInfo


def generate_pitch(client_name, df_products, summary, output_path="Pitch_Auto.pdf"):
    """
    Génère un pitch PDF professionnel avec page de garde, résumé exécutif et produits proposés.

    Args:
        client_name (str): Nom du client à afficher sur la couverture
        df_products (pd.DataFrame): Tableau des produits proposés
        summary (str): Texte du résumé exécutif (3 à 5 lignes)
        output_path (str): Chemin d'enregistrement du fichier PDF
    """

    # Création du document
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    elements = []  # Liste des blocs à ajouter

    # Styles de texte
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = styles["Heading2"]
    normal_style = styles["Normal"]

    # ---------------------------
    # 1. Page de garde
    # ---------------------------
    title = Paragraph(f"Proposition d’investissement – {client_name}", title_style)
    date = Paragraph(f"Date : {datetime.today().strftime('%d/%m/%Y')}", normal_style)

    elements.append(title)
    elements.append(Spacer(1, 20))
    elements.append(date)
    elements.append(PageBreak())

    # ---------------------------
    # 2. Résumé exécutif
    # ---------------------------
    elements.append(Paragraph("Résumé exécutif", subtitle_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(summary, normal_style))
    elements.append(PageBreak())

    # ---------------------------
    # 3. Produits proposés
    # ---------------------------
    elements.append(Paragraph("Produits proposés", subtitle_style))
    elements.append(Spacer(1, 12))

    if not df_products.empty:
       # Données du tableau
        table_data = [df_products.columns.tolist()] + df_products.values.tolist()

        # Largeurs de colonnes personnalisées (à adapter si besoin)
        col_widths = [30, 50, 40, 70, 70, 70, 40, 40, 50, 60, 50, 50][:len(table_data[0])]  # S’adapte si moins de colonnes

        # Création du tableau avec largeurs définies
        table = Table(table_data, hAlign='LEFT', colWidths=col_widths)

        # Style du tableau (inclut taille de police réduite)
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 6),  # Taille de police réduite
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('TOPPADDING', (0, 1), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),]))


        elements.append(table)
    else:
        elements.append(Paragraph("Aucun produit proposé pour le moment.", normal_style))
    
     # ---------------------------
    # 5. Résultats graphiques (ex : stress test)
    # ---------------------------
    elements.append(PageBreak())
    elements.append(Paragraph("Visualisation des résultats", subtitle_style))
    elements.append(Spacer(1, 12))

    # Exemple de graphique matplotlib
    fig, ax = plt.subplots(figsize=(6, 3))  # Format horizontal adapté au PDF

    # Exemple simple : histogramme des prix
    df_plot = df_products.copy()
    df_plot["Prix (€)"] = pd.to_numeric(df_plot["Prix (€)"], errors="coerce")
    ax.bar(df_plot.index, df_plot["Prix (€)"])
    ax.set_title("Distribution des prix des produits")
    ax.set_xlabel("Produit")
    ax.set_ylabel("Prix (€)")
    plt.tight_layout()

    # Sauvegarde du graphique en fichier image temporaire
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name, format='png')
    plt.close(fig)

    # Insertion de l’image dans le PDF
    elements.append(RLImage(tmpfile.name, width=500, height=250))  # redimensionne selon la page


    # ---------------------------
    # 5. Recommandation finale
    # ---------------------------

    # Ajout d'une nouvelle page
    elements.append(PageBreak())  # On saute une page

    # Titre de la section
    elements.append(Paragraph("Recommandation finale", subtitle_style))  # Titre H2
    elements.append(Spacer(1, 12))  # Espacement vertical

    # Génération automatique du texte de recommandation
    recommendation = generate_recommendation(df_products)  # On appelle la fonction définie plus haut

    # Ajout du texte dans le PDF
    elements.append(Paragraph(recommendation, normal_style))  # On ajoute le paragraphe

   

    # ---------------------------
    # 6. Annexes
    # ---------------------------
    elements.append(PageBreak())  # Nouvelle page
    elements.append(Paragraph("Annexes", subtitle_style))
    elements.append(Spacer(1, 12))

    # Sous-titre 1 : Fonctionnement des produits
    elements.append(Paragraph("Fonctionnement des produits", styles["Heading3"]))
    elements.append(Spacer(1, 6))

    # Liste à puces
    fonctionnement_list = ListFlowable([
        ListItem(Paragraph(
            "Cap : Instrument de couverture contre la hausse des taux. "
            "Il garantit que le taux variable ne dépassera pas un plafond (strike), "
            "tout en profitant des taux bas.", normal_style)),
        ListItem(Paragraph(
            "Floor : Instrument de protection contre la baisse des taux. "
            "Il assure un niveau plancher de rémunération sur un financement ou un placement.", normal_style)),], bulletType='bullet')

    elements.append(fonctionnement_list)
    elements.append(Spacer(1, 12))

    # Sous-titre 2 : Stress test
    elements.append(Paragraph("Méthodologie du stress test", styles["Heading3"]))
    elements.append(Spacer(1, 6))
    texte_stress = (
        "Les stress tests permettent de simuler l’impact de variations de marché sur la valorisation des produits. "
        "Les scénarios incluent :\n"
        "• Une variation de ±50 points de base sur les taux zéro-coupon\n"
        "• Une variation de ±20% sur la volatilité implicite\n"
        "Ces hypothèses permettent d’évaluer la sensibilité des instruments dans divers contextes.")
    elements.append(Paragraph(texte_stress, normal_style))
    elements.append(Spacer(1, 12))

    # Sous-titre 3 : Hypothèses de marché
    elements.append(Paragraph("Hypothèses utilisées", styles["Heading3"]))
    elements.append(Spacer(1, 6))
    texte_hypotheses = (
        "Les valorisations sont réalisées à partir des courbes de taux et de volatilité fournies. "
        "Le modèle utilisé est Black 76 pour les produits de taux (Cap/Floor). "
        "Les résultats sont exprimés en valeur actuelle nette (NAV), sans prise en compte des frais de structuration.")
    elements.append(Paragraph(texte_hypotheses, normal_style))

    # ---------------------------
    # 7. Finalisation
    # ---------------------------
    doc.build(elements)
    print(f" Pitch PDF généré : {os.path.abspath(output_path)}")



def generate_summary(df):
    """
    Génère un résumé exécutif intelligent adapté à chaque produit proposé (Cap ou Floor),
    même si plusieurs lignes sont présentes dans le DataFrame.
    """
    if df.empty:
        return "Aucun produit proposé pour le moment."

    intro_global = "Voici notre analyse des instruments de couverture proposés dans le contexte actuel :"

    lignes = []
    for idx, row in df.iterrows():
        try:
            produit = row.get("Produit", "Cap")
            devise = row.get("Devise", "EUR")

            # Strike formaté en %
            strike_raw = row.get("Strike", "N/A")
            try:
                strike_val = float(strike_raw)
                strike_str = f"{strike_val * 100:.2f}%"
            except:
                strike_str = "N/A"

            # Maturité
            maturite = row.get("Maturité", None)
            if pd.isna(maturite) or maturite in ["", "N/A", None]:
                maturite_str = "la durée du produit"
            else:
                maturite_str = str(maturite)

            # Volatilité
            vol = float(row.get("Volatilité Moy.", 0.20))
            vol_str = f"{vol * 100:.1f}%"
            if vol >= 0.30:
                contexte_vol = f"une volatilité élevée ({vol_str})"
            elif vol <= 0.10:
                contexte_vol = f"une volatilité faible ({vol_str})"
            else:
                contexte_vol = f"une volatilité modérée ({vol_str})"

            # Taux
            taux = float(row.get("Taux sans risque", 0.03))
            taux_str = f"{taux * 100:.2f}%"
            if taux >= 0.035:
                contexte_taux = f"des taux élevés ({taux_str})"
            elif taux <= 0.015:
                contexte_taux = f"des taux historiquement bas ({taux_str})"
            else:
                contexte_taux = f"des taux modérés ({taux_str})"

            # Texte adapté Cap / Floor
            if produit.lower() == "floor":
                phrase = (
                    f"Dans un environnement {contexte_taux} et {contexte_vol}, "
                    f"nous recommandons la mise en place d’un Floor {devise} à {strike_str} sur {maturite_str}, "
                    f"afin de se prémunir contre un risque de baisse des taux tout en assurant un minimum de rendement.")
            else:
                phrase = (
                    f"Dans un environnement {contexte_taux} et {contexte_vol}, "
                    f"nous recommandons la mise en place d’un Cap {devise} à {strike_str} sur {maturite_str}, "
                    f"afin de se protéger contre une hausse des taux tout en conservant la flexibilité du taux variable.")

            lignes.append(phrase)

        except Exception as e:
            lignes.append(f"Ligne {idx+1} ignorée (erreur : {e})")

    return intro_global + "\n\n" + "\n\n".join(lignes)




def generate_recommendation(df):
    """
    Génère un texte de recommandation final à inclure dans le pitch PDF,
    basé sur le type de produit et le contexte de marché.
    """
    if df.empty:
        return "Aucune recommandation ne peut être formulée à ce stade."

    recommandations = []

    # Liste des types de produits présents
    try:
        produits = df["Produit"].str.lower().unique()
    except KeyError:
        return "Colonne 'Produit' absente du tableau."

    for produit in produits:
        sous_df = df[df["Produit"].str.lower() == produit]

        devise = sous_df["Devise"].iloc[0] if "Devise" in sous_df else "EUR"
        strike_moy = sous_df["Strike"].mean()
        # Gestion plus robuste de la maturité
        if "Maturité" in sous_df:
            maturites = sous_df["Maturité"].dropna().astype(str)
            maturite_mode = maturites.mode().iloc[0] if not maturites.empty else "la durée du produit"
        else:
            maturite_mode = "la durée du produit"


        if produit == "cap":
            phrase = (f"Nous suggérons la mise en place de cap(s) {devise} à un niveau moyen de {strike_moy:.2%} "
                f"sur {maturite_mode}, dans le but de protéger contre une éventuelle hausse des taux. "
                f"Cette solution permet de sécuriser un plafond tout en conservant la flexibilité du taux variable.")
        elif produit == "floor":
            phrase = (f"Nous proposons l’utilisation de floor(s) {devise} avec un strike moyen de {strike_moy:.2%} "
                f"sur {maturite_mode}, afin de garantir un niveau plancher de rémunération en cas de baisse des taux. "
                f"Cette stratégie offre une protection contre un environnement de taux bas prolongé.")
        else:
            phrase = f"Produit non identifié : {produit}"
        intro_client = (f"Afin de répondre aux enjeux de couverture de taux identifiés avec {devise} comme devise de référence, ")
        
        recommandations.append(intro_client + phrase)

    # Texte final
    return "\n\n".join(recommandations)




