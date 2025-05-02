from transformers import pipeline
import streamlit as st

# Charger le modèle de sentiment analysis en français
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiment(text):
    # Charger le modèle
    classifier = load_model()
    
    # Analyser le sentiment
    result = classifier(text)[0]
    
    # Convertir le score en sentiment
    score = int(result['label'].split()[0])
    if score <= 2:
        sentiment = "Négatif"
    elif score == 3:
        sentiment = "Neutre"
    else:
        sentiment = "Positif"
    
    return sentiment, result['score']

# Interface Streamlit
st.title("Analyse de Sentiment en Français")
st.write("Entrez un texte en français pour analyser son sentiment")

# Zone de texte pour l'entrée
text = st.text_area("Votre texte:", height=100)

if st.button("Analyser"):
    if text:
        sentiment, score = analyze_sentiment(text)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Score de confiance: {score:.2f}")
        
        # Afficher une barre de progression colorée
        if sentiment == "Positif":
            st.progress(score, text="Positif")
        elif sentiment == "Négatif":
            st.progress(score, text="Négatif")
        else:
            st.progress(score, text="Neutre")
    else:
        st.warning("Veuillez entrer un texte à analyser") 