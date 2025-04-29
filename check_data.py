import pandas as pd

# Charger le fichier train.csv
print("Chargement du fichier train.csv...")
df = pd.read_csv('data/train.csv')

# Afficher les informations sur le DataFrame
print("\nInformations sur le DataFrame :")
print(df.info())

# Afficher les colonnes
print("\nColonnes disponibles :")
print(df.columns.tolist())

# Afficher les premières lignes
print("\nPremières lignes du DataFrame :")
print(df.head()) 