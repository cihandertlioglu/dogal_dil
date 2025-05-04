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

# Turkish NLP tools
lemmatizer = zeyrek.MorphAnalyzer()
stemmer = TurkishStemmer()
stop_words = set(stopwords.words('turkish'))

# Enhanced preprocessing function for Turkish legal texts
def preprocess_sentence(sentence):
    # 1. Lowercase: Convert everything to lowercase for uniformity
    sentence = sentence.lower()

    # 2. Remove special characters and numbers but keep Turkish specific characters
    sentence = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', sentence)
    
    # 3. Tokenize the sentence into words
    tokens = word_tokenize(sentence)
    
    # 4. Remove stopwords and short tokens
    filtered_tokens = [
        token for token in tokens 
        if token.isalpha() 
        and token not in stop_words
        and len(token) > 2  # Remove very short tokens
    ]
    
    # 5. Lemmatization using Zeyrek (morphological analysis)
    lemmatized_tokens = []
    for token in filtered_tokens:
        try:
            analysis = lemmatizer.analyze(token)
            if analysis and analysis[0]:
                lemma = analysis[0][0][1]  # Get the first lemma
                lemmatized_tokens.append(lemma)
        except:
            # Fallback to original token if lemmatization fails
            lemmatized_tokens.append(token)
    
    # 6. Stemming using SnowballStemmer
    stemmed_tokens = [stemmer.stemWord(token) for token in filtered_tokens]
    
    return lemmatized_tokens, stemmed_tokens

# Process sentences
sentences = []
for text in tqdm(texts, desc="Extracting sentences"):
    sentences.extend(sent_tokenize(text))

print(f"\nSample sentences: {sentences[:2]}\n")

# Tokenization, lemmatization, and stemming
tokenized_corpus_lemmatized = []
tokenized_corpus_stemmed = []

for sentence in tqdm(sentences, desc="Processing texts"):
    lemmatized, stemmed = preprocess_sentence(sentence)
    tokenized_corpus_lemmatized.append(lemmatized)
    tokenized_corpus_stemmed.append(stemmed)

print("\nLemmatized sample:", tokenized_corpus_lemmatized[:2])
print("Stemmed sample:", tokenized_corpus_stemmed[:2])

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Ön işlenmiş token listelerini tekrar metne çeviriyoruz
lemmatized_texts = [' '.join(tokens) for tokens in tokenized_corpus_lemmatized]

lemmatized_texts[:3]

# TF-IDF vektörizerı başlatıyoruz
vectorizer = TfidfVectorizer()

# TF-IDF matrisini oluşturuyoruz
#terim frekansları, belge frekanslarıni hesplar
#TF-IDF vektörlerine dönüştürür
tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)

## Kelimeleri alalım
#F-IDF vektörleştirme işleminde kullanılan tüm kelimelerin essiz bir listesini döndürur
feature_names = vectorizer.get_feature_names_out()

# TF-IDF matrisini pandas DataFrame'e çevir-gorunurluk acisindan- calismasi kolay
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# İlk birkaç satırı gösterelim-ilk 5 cümle
print(tfidf_df.head())

#Her satır bir cümleyi temsil eder
#Her sütun bir kelimeyi temsil eder
#Hücreler ise o kelimenin o cümledeki TF-IDF skorudur - her cumle icin degisir-bakiniz:slaytlar
# İlk cümle için TF-IDF skorlarını al
first_sentence_vector = tfidf_df.iloc[0]

# Skorlara göre sırala (yüksekten düşüğe)
top_5_words = first_sentence_vector.sort_values(ascending=False).head(5)

# Sonucu yazdır
print("İlk cümlede en yüksek TF-IDF skoruna sahip 5 kelime:")
print(top_5_words)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# gol kelimesinin vektörünü alalım
gol_index = feature_names.tolist().index('gol')  # 'gol' kelimesinin indeksini bul

# gol kelimesinin TF-IDF vektörünü alıyoruz ve 2D formatta yapıyoruz
gol_vector = tfidf_matrix[:, gol_index].toarray()

# Tüm kelimelerin TF-IDF vektörlerini alıyoruz
tfidf_vectors = tfidf_matrix.toarray()

# Cosine similarity hesaplayalım
similarities = cosine_similarity(gol_vector.T, tfidf_vectors.T)

# Benzerlikleri sıralayalım ve en yüksek 5 kelimeyi seçelim
similarities = similarities.flatten()
top_5_indices = similarities.argsort()[-6:][::-1]  # 6. en büyükten başlıyoruz çünkü kendisi de dahil

# Sonuçları yazdıralım
for index in top_5_indices:
    print(f"{feature_names[index]}: {similarities[index]:.4f}")