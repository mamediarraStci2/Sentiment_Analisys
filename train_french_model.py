import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from googletrans import Translator
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

def translate_dataset_to_french(dataset_path, output_path):
    """
    Traduit le dataset d'entraînement en français
    """
    print("Chargement du dataset...")
    df = pd.read_csv(dataset_path)
    
    # Initialisation du traducteur
    translator = Translator()
    
    # Traduction des textes
    print("Traduction des textes en cours...")
    translated_texts = []
    
    for text in tqdm(df['text']):
        try:
            # Traduction en français
            translated = translator.translate(text, dest='fr').text
            translated_texts.append(translated)
            # Pause pour éviter les limites de l'API
            time.sleep(0.5)
        except Exception as e:
            print(f"Erreur lors de la traduction: {e}")
            translated_texts.append(text)  # Garde le texte original en cas d'erreur
    
    # Création du nouveau dataset
    df_french = df.copy()
    df_french['text'] = translated_texts
    
    # Sauvegarde du dataset traduit
    print(f"Sauvegarde du dataset traduit dans {output_path}")
    df_french.to_csv(output_path, index=False)
    
    return df_french

def preprocess_french_text(text):
    """
    Prétraitement du texte en français
    """
    # Nettoyage basique
    text = text.lower()
    text = ' '.join(text.split())
    return text

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
    def __len__(self):
        return len(self.labels)

def train_french_model(dataset_path):
    """
    Entraîne un nouveau modèle sur le dataset français
    """
    print("Chargement du dataset français...")
    df = pd.read_csv(dataset_path)
    
    # Prétraitement des textes
    print("Prétraitement des textes...")
    df['processed_text'] = df['text'].apply(preprocess_french_text)
    
    # Division du dataset
    X = df['processed_text']
    y = pd.get_dummies(df['sentiment'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialisation du tokenizer et du modèle
    print("Initialisation du modèle...")
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=3)
    
    # Préparation des données
    print("Préparation des données...")
    train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True)
    
    train_dataset = SentimentDataset(train_encodings, y_train.values)
    test_dataset = SentimentDataset(test_encodings, y_test.values)
    
    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Entraînement du modèle
    print("Début de l'entraînement...")
    trainer.train()
    
    # Sauvegarde du modèle
    print("Sauvegarde du modèle...")
    model.save_pretrained('./french_sentiment_model')
    tokenizer.save_pretrained('./french_sentiment_model')
    
    return model, tokenizer

if __name__ == "__main__":
    # Chemin vers votre dataset original
    original_dataset_path = "train.csv"
    # Chemin pour sauvegarder le dataset traduit
    translated_dataset_path = "train_french.csv"
    
    # Traduction du dataset
    df_french = translate_dataset_to_french(original_dataset_path, translated_dataset_path)
    
    # Entraînement du modèle
    model, tokenizer = train_french_model(translated_dataset_path)
    
    print("Modèle français entraîné avec succès !") 