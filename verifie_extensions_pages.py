import os

# Remplace par le chemin complet si besoin
chemin_pages = "pages"

print(f"\nFichiers dans le dossier '{chemin_pages}':\n")

for fichier in os.listdir(chemin_pages):
    chemin_complet = os.path.join(chemin_pages, fichier)
    if os.path.isfile(chemin_complet):
        nom, ext = os.path.splitext(fichier)
        print(f"- {fichier}   --> Extension détectée : '{ext}'")

print("\n✅ Vérifie que toutes les extensions sont bien '.py'")
