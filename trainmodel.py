import pandas as pd
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from snowballstemmer import TurkishStemmer
import zeyrek
import re


tqdm.pandas()

# Load data
df = pd.read_csv("MacYorumOnerici.csv")
texts = df["comment_text"].dropna().progress_apply(str)  
texts = texts.tolist()  


lemmatizer = zeyrek.MorphAnalyzer()
stemmer = TurkishStemmer()
stop_words = set(stopwords.words('turkish'))


def preprocess_sentence(sentence):

    sentence = sentence.lower()


    

    tokens = word_tokenize(sentence)
    

    filtered_tokens = [
        token for token in tokens 
        if token.isalpha() 
        and token not in stop_words
        and len(token) > 2  # Remove very short tokens
    ]
    

    lemmatized_tokens = []
    for token in filtered_tokens:
        try:
            analysis = lemmatizer.analyze(token)
            if analysis and analysis[0]:
                lemma = analysis[0][0][1]  # Get the first lemma
                lemmatized_tokens.append(lemma)
        except:
          
            lemmatized_tokens.append(token)
    

    stemmed_tokens = [stemmer.stemWord(token) for token in filtered_tokens]
    
    return lemmatized_tokens, stemmed_tokens

sentences = []
for text in tqdm(texts, desc="Extracting sentences"):
    sentences.extend(sent_tokenize(text))

print(f"\nSample sentences: {sentences[:2]}\n")


tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in tqdm(sentences, desc="Processing texts"):
    lemmatized, stemmed = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized)
    tokenized_corpus_stemmed.append(stemmed)

print("\nLemmatized sample:", tokenized_corpus_lemmatized[:2])
print("Stemmed sample:", tokenized_corpus_stemmed[:2])


lemmatized_df = pd.DataFrame({
    'original_sentence': sentences,
    'processed_tokens': [' '.join(tokens) for tokens in tokenized_corpus_lemmatized]
})

stemmed_df = pd.DataFrame({
    'original_sentence': sentences,
    'processed_tokens': [' '.join(tokens) for tokens in tokenized_corpus_stemmed]
})

# Save to CSV files
lemmatized_df.to_csv('lemmatized.csv', index=False, encoding='utf-8-sig')
stemmed_df.to_csv('stemmed.csv', index=False, encoding='utf-8-sig')

print("\nFiles saved successfully:")
print("- lemmatized.csv")
print("- stemmed.csv")


# Word2Vec parametreleri
parameters = [
    {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow', 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow', 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]

# Model eğitme ve kaydetme fonksiyonu
def train_and_save_model(corpus, params, model_name):
    model = Word2Vec(
        corpus, 
        vector_size=params['vector_size'],
        window=params['window'],
        min_count=1,
        sg=1 if params['model_type'] == 'skipgram' else 0,
        workers=4  # Çoklu işlemci desteği
    )
    model.save(f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']}.model")
    print(f"{model_name}_{params['model_type']}_window{params['window']}_dim{params['vector_size']} model saved!")

# Modelleri eğit
print("Lemmatized modeller eğitiliyor...")
for param in tqdm(parameters, desc="Lemmatized modeller"):
    train_and_save_model(tokenized_corpus_lemmatized, param, "lemmatized_model")

print("\nStemmed modeller eğitiliyor...")
for param in tqdm(parameters, desc="Stemmed modeller"):
    train_and_save_model(tokenized_corpus_stemmed, param, "stemmed_model")