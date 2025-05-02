import pandas as pd
from googletrans import Translator
import time
import random

def translate_text(text, translator, max_retries=3):
    if pd.isna(text) or text.strip() == '':
        return ''
    
    for attempt in range(max_retries):
        try:
            # Ajouter un délai aléatoire entre 2 et 5 secondes
            time.sleep(random.uniform(2, 5))
            translation = translator.translate(text, src='fr', dest='en')
            return translation.text
        except Exception as e:
            print(f"Tentative {attempt + 1} échouée pour le texte: {str(e)}")
            if attempt < max_retries - 1:
                # Attendre plus longtemps avant de réessayer
                time.sleep(10)
            else:
                print(f"Échec après {max_retries} tentatives pour le texte: {text[:50]}...")
                return text

def translate_csv(input_file, output_file, batch_size=100):
    translator = Translator()
    
    print(f"Lecture du fichier {input_file}...")
    df = pd.read_csv(input_file)
    
    if 'text' not in df.columns:
        print(f"La colonne 'text' n'existe pas dans {input_file}")
        return
    
    # Créer la colonne pour les traductions
    df['text_en'] = ''
    
    # Traduire par lots
    total_rows = len(df)
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        print(f"Traitement des lignes {start_idx} à {end_idx} sur {total_rows}")
        
        # Traduire le lot actuel
        for i in range(start_idx, end_idx):
            df.at[i, 'text_en'] = translate_text(df.at[i, 'text'], translator)
        
        # Sauvegarder la progression après chaque lot
        print(f"Sauvegarde intermédiaire...")
        df.to_csv(output_file, index=False)
        print(f"Progression sauvegardée: {end_idx}/{total_rows} lignes traitées")
        
        # Pause plus longue entre les lots
        time.sleep(10)

# Liste des fichiers à traduire
files_to_translate = [
    ('data/test.csv', 'data/test_en.csv'),
    ('data/train.csv', 'data/train_en.csv'),
    ('data/sample_submission.csv', 'data/sample_submission_en.csv')
]

# Traduire chaque fichier
for input_file, output_file in files_to_translate:
    try:
        translate_csv(input_file, output_file)
    except Exception as e:
        print(f"Erreur lors du traitement de {input_file}: {str(e)}")

print("Tous les fichiers ont été traités!") 