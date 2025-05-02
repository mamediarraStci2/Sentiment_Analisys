import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
import os
from transformers import pipeline

# Créer un répertoire pour les données NLTK avec les bonnes permissions
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Télécharger les données NLTK nécessaires
nltk.download('punkt', download_dir=nltk_data_dir)

# Dictionnaire de mots négatifs en français avec leurs poids
FRENCH_NEGATIVE_WORDS = {
    'déteste': 5, 'haine': 5, 'haïr': 5, 'méchant': 5, 'mechant': 5, 'méchante': 5, 'mechante': 5,
    'horrible': 4, 'terrible': 4, 'énervé': 4, 'fâché': 4, 'triste': 3, 'déçu': 3, 'désolé': 3,
    'malheureux': 3, 'désagréable': 3, 'insupportable': 4, 'insatisfait': 3, 'dégoûté': 4,
    'déplaisant': 3, 'désespéré': 4, 'désolant': 3, 'désastreux': 4, 'catastrophique': 4,
    'affreux': 4, 'atroce': 4, 'épouvantable': 4, 'exécrable': 4, 'odieux': 4, 'détestable': 4,
    'abominable': 4, 'horripilant': 4, 'exaspérant': 4, 'agaçant': 3, 'énervant': 3,
    'irritant': 3, 'fâcheux': 3, 'regrettable': 3, 'décevant': 3, 'déplorable': 3,
    'pitoyable': 3, 'médiocre': 3, 'nul': 3, 'pourri': 4, 'merdique': 4, 'dégueulasse': 4,
    'infect': 4, 'immondice': 4, 'saleté': 4, 'ordure': 4, 'crapule': 4, 'salaud': 4,
    'connard': 4, 'imbécile': 3, 'idiot': 3, 'stupide': 3, 'crétin': 3, 'abruti': 3,
    'débile': 3, 'con': 3, 'niais': 3, 'bête': 3, 'malin': 3, 'malicieux': 3,
    'perfide': 4, 'traître': 4, 'fourbe': 4, 'hypocrite': 4, 'menteur': 4, 'trompeur': 4,
    'dangereux': 4, 'menaçant': 4, 'agressif': 4, 'violent': 4, 'brutal': 4, 'cruel': 4,
    'sadique': 4, 'méprisable': 4, 'ignoble': 4, 'vile': 4, 'bas': 3, 'lâche': 4,
    'peureux': 3, 'faible': 3, 'incompétent': 3, 'incapable': 3, 'inutile': 3,
    'inutilité': 3, 'raté': 3, 'échec': 3, 'perdu': 3, 'désastre': 4, 'catastrophe': 4,
    'calamité': 4, 'malheur': 4, 'tragédie': 4, 'drame': 4, 'problème': 3, 'difficulté': 3,
    'obstacle': 3, 'barrière': 3, 'blocage': 3, 'empêchement': 3, 'handicap': 3,
    'désavantage': 3, 'inconvénient': 3, 'nuisance': 3, 'gêne': 3, 'dérangement': 3,
    'trouble': 3, 'perturbation': 3, 'désordre': 3, 'chaos': 4, 'confusion': 3,
    'désorganisation': 3, 'déséquilibre': 3, 'instabilité': 3, 'incertitude': 3,
    'doute': 3, 'suspicion': 3, 'méfiance': 3, 'crainte': 3, 'peur': 3, 'angoisse': 4,
    'anxiété': 4, 'stress': 3, 'tension': 3, 'pression': 3, 'souci': 3, 'inquiétude': 3,
    'préoccupation': 3, 'tourment': 4, 'souffrance': 4, 'douleur': 4, 'mal': 3,
    'maladie': 3, 'blessure': 3, 'plaie': 3, 'cicatrice': 3, 'traumatisme': 4,
    'choc': 3, 'crise': 3, 'urgence': 3, 'danger': 4, 'risque': 3, 'menace': 4,
    'péril': 4, 'perte': 3, 'manque': 3, 'absence': 3, 'vide': 3, 'trou': 3,
    'faille': 3, 'défaut': 3, 'faiblesse': 3, 'vulnérabilité': 3, 'fragilité': 3,
    'sensibilité': 3, 'susceptibilité': 3, 'irritabilité': 3, 'agacement': 3,
    'exaspération': 4, 'énervement': 3, 'colère': 4, 'fureur': 4, 'rage': 4,
    'haine': 5, 'rancune': 4, 'ressentiment': 4, 'vengeance': 4, 'revanche': 4
}

# Dictionnaire de mots positifs en français avec leurs poids
FRENCH_POSITIVE_WORDS = {
    'amour': 5, 'adorer': 5, 'aimer': 5, 'merveilleux': 4, 'fantastique': 4, 'superbe': 4,
    'génial': 4, 'excellent': 4, 'parfait': 4, 'magnifique': 4, 'splendide': 4, 'formidable': 4,
    'extraordinaire': 4, 'incroyable': 4, 'fabuleux': 4, 'merveilleux': 4, 'sublime': 4,
    'heureux': 4, 'joyeux': 4, 'content': 3, 'satisfait': 3, 'ravi': 4, 'enchanté': 4,
    'émerveillé': 4, 'ébloui': 4, 'enthousiaste': 4, 'passionné': 4, 'passionnant': 4,
    'intéressant': 3, 'captivant': 4, 'fascinant': 4, 'attrayant': 3, 'séduisant': 3,
    'charmant': 3, 'agréable': 3, 'plaisant': 3, 'délicieux': 4, 'savoureux': 3,
    'exquis': 4, 'succulent': 3, 'délectable': 3, 'appétissant': 3, 'gourmand': 3,
    'gourmet': 3, 'gastronomique': 3, 'culinaire': 3, 'artisanal': 3, 'authentique': 3,
    'naturel': 3, 'biologique': 3, 'écologique': 3, 'durable': 3, 'responsable': 3,
    'éthique': 3, 'moral': 3, 'vertueux': 3, 'noble': 3, 'digne': 3, 'honorable': 3,
    'respectable': 3, 'estimable': 3, 'appréciable': 3, 'valable': 3, 'valide': 3,
    'légitime': 3, 'juste': 3, 'équitable': 3, 'impartial': 3, 'objectif': 3,
    'rationnel': 3, 'logique': 3, 'cohérent': 3, 'pertinent': 3, 'approprié': 3,
    'adéquat': 3, 'convenable': 3, 'satisfaisant': 3, 'acceptable': 3, 'tolérable': 3,
    'supportable': 3, 'endurable': 3, 'durable': 3, 'stable': 3, 'solide': 3,
    'robuste': 3, 'résistant': 3, 'fort': 3, 'puissant': 3, 'efficace': 3,
    'productif': 3, 'performant': 3, 'compétent': 3, 'capable': 3, 'talentueux': 3,
    'doué': 3, 'gifted': 3, 'brillant': 4, 'intelligent': 3, 'sage': 3,
    'savant': 3, 'érudit': 3, 'cultivé': 3, 'instruit': 3, 'éduqué': 3,
    'diplômé': 3, 'qualifié': 3, 'expérimenté': 3, 'professionnel': 3, 'expert': 3,
    'spécialiste': 3, 'maître': 3, 'virtuose': 4, 'génie': 4, 'prodigieux': 4
}

# Dictionnaire de mots neutres en français avec leurs poids
FRENCH_NEUTRAL_WORDS = {
    'normal': 2, 'ordinaire': 2, 'habituel': 2, 'courant': 2, 'commun': 2, 'banal': 2,
    'standard': 2, 'classique': 2, 'traditionnel': 2, 'conventionnel': 2, 'régulier': 2,
    'fréquent': 2, 'quotidien': 2, 'journalier': 2, 'hebdomadaire': 2, 'mensuel': 2,
    'annuel': 2, 'périodique': 2, 'systématique': 2, 'méthodique': 2, 'organisé': 2,
    'structuré': 2, 'planifié': 2, 'programmé': 2, 'prévu': 2, 'anticipé': 2,
    'attendu': 2, 'espéré': 2, 'souhaité': 2, 'désiré': 2, 'voulu': 2,
    'choisi': 2, 'sélectionné': 2, 'retenu': 2, 'adopté': 2, 'approuvé': 2,
    'validé': 2, 'confirmé': 2, 'certifié': 2, 'garanti': 2, 'assuré': 2,
    'sécurisé': 2, 'protégé': 2, 'conservé': 2, 'maintenu': 2, 'préservé': 2,
    'gardé': 2, 'retenu': 2, 'conservé': 2, 'stocké': 2, 'archivé': 2,
    'enregistré': 2, 'sauvegardé': 2, 'copié': 2, 'dupliqué': 2, 'reproduit': 2,
    'imité': 2, 'copié': 2, 'plagié': 2, 'reproduit': 2, 'dupliqué': 2,
    'multiplié': 2, 'augmenté': 2, 'accru': 2, 'développé': 2, 'étendu': 2,
    'élargi': 2, 'agrandi': 2, 'aggravé': 2, 'intensifié': 2, 'renforcé': 2,
    'consolidé': 2, 'stabilisé': 2, 'équilibré': 2, 'harmonisé': 2, 'coordonné': 2,
    'organisé': 2, 'structuré': 2, 'planifié': 2, 'programmé': 2, 'prévu': 2,
    'anticipé': 2, 'attendu': 2, 'espéré': 2, 'souhaité': 2, 'désiré': 2,
    'voulu': 2, 'choisi': 2, 'sélectionné': 2, 'retenu': 2, 'adopté': 2
}

# Charger le modèle et le vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Charger le modèle de sentiment analysis en français
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def preprocess(text):
    # Gérer les valeurs manquantes
    if not text:
        return ""
        
    # Convertir en minuscules
    text = text.lower()
    
    # Gérer les contractions avec apostrophe
    text = text.replace("j'", "je ")
    text = text.replace("t'", "te ")
    text = text.replace("l'", "le ")
    text = text.replace("m'", "me ")
    text = text.replace("s'", "se ")
    text = text.replace("d'", "de ")
    text = text.replace("n'", "ne ")
    text = text.replace("c'", "ce ")
    
    # Supprimer les URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Supprimer les mentions et hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Supprimer la ponctuation sauf les apostrophes
    text = re.sub(r'[^\w\s\']', '', text)
    
    # Tokenisation
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    
    # Supprimer les stopwords et ajouter des poids aux mots selon leur sentiment
    try:
        stop_words = set(stopwords.words('french'))
        processed_tokens = []
        for word in tokens:
            if word not in stop_words:
                # Si le mot est dans un de nos dictionnaires, le répéter selon son poids
                if word in FRENCH_NEGATIVE_WORDS:
                    weight = FRENCH_NEGATIVE_WORDS[word]
                    processed_tokens.extend([word] * weight)
                elif word in FRENCH_POSITIVE_WORDS:
                    weight = FRENCH_POSITIVE_WORDS[word]
                    processed_tokens.extend([word] * weight)
                elif word in FRENCH_NEUTRAL_WORDS:
                    weight = FRENCH_NEUTRAL_WORDS[word]
                    processed_tokens.extend([word] * weight)
                else:
                    processed_tokens.append(word)
        tokens = processed_tokens
    except LookupError:
        pass
    
    # Ne pas utiliser le stemming pour préserver les mots originaux
    return ' '.join(tokens)

def predict_sentiment(text):
    # Prétraitement
    cleaned = preprocess(text)
    # Vectorisation
    vect = vectorizer.transform([cleaned])
    # Prédiction
    sentiment = model.predict(vect)[0]
    # Probabilités
    probabilities = model.predict_proba(vect)[0]
    return sentiment, probabilities

# Interface utilisateur
st.set_page_config(
    page_title="Analyse de Sentiment Twitter",
    page_icon="🐦",
    layout="centered"
)

st.title("Analyse de Sentiment Twitter 🐦")

# Ajouter une description
st.markdown("""
Cette application analyse le sentiment d'un texte (tweet, commentaire, etc.) et détermine s'il est :
- 😊 Positif
- 😐 Neutre
- 😔 Négatif

Entrez votre texte ci-dessous et cliquez sur 'Analyser' !
""")

# Zone de texte pour l'entrée
tweet = st.text_area(
    "Entrez votre texte :",
    height=150,
    placeholder="Exemple : Tu es vraiment méchant !"
)

# Bouton d'analyse
if st.button("Analyser"):
    if tweet:
        # Afficher un spinner pendant l'analyse
        with st.spinner('Analyse en cours...'):
            sentiment, probabilities = predict_sentiment(tweet)
            
            # Personnaliser l'affichage selon le sentiment
            if sentiment == "positive":
                st.success(f"Sentiment : POSITIF 😊")
            elif sentiment == "negative":
                st.error(f"Sentiment : NÉGATIF 😔")
            else:
                st.info(f"Sentiment : NEUTRE 😐")
            
            # Afficher les probabilités
            st.write("---")
            st.write("**Probabilités de chaque sentiment :**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positif", f"{probabilities[2]*100:.1f}%")
            with col2:
                st.metric("Neutre", f"{probabilities[1]*100:.1f}%")
            with col3:
                st.metric("Négatif", f"{probabilities[0]*100:.1f}%")
            
            # Afficher le texte prétraité
            st.write("---")
            st.write("**Détails du traitement :**")
            st.write("Texte prétraité :", preprocess(tweet))
    else:
        st.warning("Veuillez saisir un texte d'abord.") 