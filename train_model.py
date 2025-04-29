import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer  # Utiliser SnowballStemmer pour le français
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Télécharger les ressources NLTK nécessaires
print("Téléchargement des ressources NLTK...")
nltk.download('punkt')
nltk.download('stopwords')
print("Ressources NLTK téléchargées.")

# Dictionnaire de mots négatifs en français avec leurs poids
FRENCH_NEGATIVE_WORDS = {
    'déteste': 5, 'haine': 5, 'haïr': 5, 'méchant': 4, 'horrible': 4, 'terrible': 4,
    'énervé': 4, 'fâché': 4, 'triste': 3, 'déçu': 3, 'désolé': 3, 'malheureux': 3,
    'désagréable': 3, 'insupportable': 4, 'insatisfait': 3, 'dégoûté': 4, 'déplaisant': 3,
    'désespéré': 4, 'désolant': 3, 'désastreux': 4, 'catastrophique': 4, 'affreux': 4,
    'atroce': 4, 'épouvantable': 4, 'exécrable': 4, 'odieux': 4, 'détestable': 4,
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
    'adore': 5, 'aime': 4, 'excellent': 5, 'parfait': 5, 'magnifique': 5, 'superbe': 5,
    'merveilleux': 5, 'fantastique': 5, 'extraordinaire': 5, 'incroyable': 5, 'génial': 5,
    'formidable': 5, 'super': 4, 'cool': 4, 'sympa': 4, 'agréable': 4, 'plaisant': 4,
    'content': 4, 'heureux': 4, 'joyeux': 4, 'ravi': 4, 'enchanté': 4, 'satisfait': 4,
    'brillant': 4, 'intelligent': 4, 'sage': 4, 'beau': 4, 'charmant': 4, 'adorable': 4,
    'mignon': 4, 'doux': 4, 'gentil': 4, 'aimable': 4, 'généreux': 4, 'attentionné': 4,
    'bienveillant': 4, 'compatissant': 4, 'compréhensif': 4, 'patient': 4, 'calme': 3,
    'paisible': 3, 'serein': 3, 'tranquille': 3, 'reposant': 3, 'relaxant': 3,
    'apaisant': 3, 'réconfortant': 4, 'rassurant': 4, 'encourageant': 4, 'motivant': 4,
    'inspirant': 4, 'stimulant': 4, 'dynamique': 4, 'énergique': 4, 'vif': 4,
    'rapide': 3, 'efficace': 4, 'compétent': 4, 'capable': 4, 'talentueux': 4,
    'doué': 4, 'expert': 4, 'professionnel': 4, 'qualifié': 4, 'expérimenté': 4,
    'fiable': 4, 'sûr': 4, 'solide': 4, 'stable': 3, 'durable': 3, 'résistant': 3,
    'fort': 4, 'puissant': 4, 'vigoureux': 4, 'robuste': 4, 'sain': 4, 'prospère': 4,
    'riche': 4, 'abondant': 4, 'généreux': 4, 'luxueux': 4, 'précieux': 4,
    'valeureux': 4, 'noble': 4, 'honorable': 4, 'respectable': 4, 'admirable': 4,
    'exemplaire': 4, 'modèle': 4, 'idéal': 4, 'optimal': 4, 'impeccable': 4,
    'irréprochable': 4, 'excellent': 5, 'exceptionnel': 5, 'remarquable': 5,
    'spectaculaire': 5, 'sensationnel': 5, 'fabuleux': 5, 'prodigieux': 5,
    'sublime': 5, 'divin': 5, 'céleste': 5, 'paradisiaque': 5, 'magique': 5,
    'enchanteur': 5, 'féerique': 5, 'merveilleux': 5, 'miraculeux': 5
}

# Dictionnaire de mots neutres en français avec leurs poids
FRENCH_NEUTRAL_WORDS = {
    'normal': 1, 'moyen': 1, 'ordinaire': 1, 'standard': 1, 'habituel': 1,
    'régulier': 1, 'commun': 1, 'courant': 1, 'typique': 1, 'classique': 1,
    'conventionnel': 1, 'traditionnel': 1, 'usuel': 1, 'banal': 1, 'quelconque': 1,
    'neutre': 1, 'indifférent': 1, 'impartial': 1, 'objectif': 1, 'équilibré': 1,
    'modéré': 1, 'tempéré': 1, 'mesuré': 1, 'raisonnable': 1, 'acceptable': 1,
    'passable': 1, 'suffisant': 1, 'adéquat': 1, 'convenable': 1, 'correct': 1,
    'approprié': 1, 'pertinent': 1, 'adapté': 1, 'conforme': 1, 'régulier': 1,
    'stable': 1, 'constant': 1, 'uniforme': 1, 'homogène': 1, 'égal': 1,
    'similaire': 1, 'comparable': 1, 'équivalent': 1, 'analogue': 1, 'semblable': 1,
    'pareil': 1, 'identique': 1, 'même': 1, 'tel': 1, 'ainsi': 1,
    'donc': 1, 'alors': 1, 'ensuite': 1, 'puis': 1, 'après': 1,
    'avant': 1, 'pendant': 1, 'durant': 1, 'lors': 1, 'tandis': 1,
    'car': 1, 'parce': 1, 'puisque': 1, 'comme': 1, 'si': 1,
    'mais': 1, 'ou': 1, 'et': 1, 'donc': 1, 'or': 1,
    'ni': 1, 'soit': 1, 'cependant': 1, 'néanmoins': 1, 'toutefois': 1,
    'pourtant': 1, 'malgré': 1, 'bien': 1, 'mal': 1, 'peut-être': 1,
    'probablement': 1, 'possiblement': 1, 'éventuellement': 1, 'certainement': 1, 'sûrement': 1,
    'vraiment': 1, 'réellement': 1, 'effectivement': 1, 'véritablement': 1, 'assurément': 1
}

def load_data(path_csv):
    # Charger les données avec pandas
    df = pd.read_csv(path_csv)
    
    # Afficher les informations sur les données
    print("Nombre total d'échantillons:", len(df))
    print("Distribution des sentiments:")
    print(df['sentiment'].value_counts())
    
    return df['text'], df['sentiment']

def preprocess(text):
    # Gérer les valeurs manquantes
    if pd.isna(text):
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

def train(texts, labels):
    print("Prétraitement des textes...")
    # Appliquer le prétraitement
    cleaned = [preprocess(t) for t in texts]
    
    print("Vectorisation des textes...")
    # Vectorisation avec plus de features et n-grammes
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),  # Utiliser des trigrammes
        min_df=2,  # Ignorer les termes qui apparaissent moins de 2 fois
        max_df=0.95,  # Ignorer les termes qui apparaissent dans plus de 95% des documents
        analyzer='word'  # Utiliser des mots complets
    )
    X = vectorizer.fit_transform(cleaned)
    
    # Séparation en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    print("Entraînement du modèle...")
    # Modèle avec class_weight pour gérer le déséquilibre des classes
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.1,  # Régularisation plus forte
        solver='liblinear',  # Meilleur pour les petits datasets
        multi_class='ovr'  # One-vs-rest pour une meilleure performance sur les classes déséquilibrées
    )
    model.fit(X_train, y_train)
    
    # Évaluation
    print("\nÉvaluation du modèle sur l'ensemble de test:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Sauvegarde des artefacts
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("\nModèle et vectorizer sauvegardés.")

if __name__ == '__main__':
    try:
        texts, labels = load_data('data/train.csv')
        train(texts, labels)
    except Exception as e:
        print(f"Erreur lors de l'exécution : {str(e)}") 