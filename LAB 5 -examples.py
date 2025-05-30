#Przykład 1

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')

tekst = "Przetwarzanie języka naturalnego jest fascynujące i rozwija się dynamicznie."

# Tokenizacja - dzielenie tekstu na słowa
tokeny = word_tokenize(tekst)
print("Tokeny:", tokeny)

# Stemming - redukcja słów do ich podstawowej formy
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokeny]
print("Stemmed:", stemmed_words)

#Przykład 2

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

tekst = "Microsoft został założony przez Billa Gatesa."
tokeny = word_tokenize(tekst)
pos_tagged = pos_tag(tokeny)

print("Części mowy:", pos_tagged)


#Przykład 1
import spacy

nlp = spacy.load("en_core_web_sm")

tekst = "Microsoft został założony przez Billa Gatesa."
doc = nlp(tekst)

for ent in doc.ents:
    print(ent.text, ent.label_)


#Przykład 2
import spacy

nlp = spacy.load("en_core_web_sm")

tekst = "Kocham programowanie i pisanie kodu."
doc = nlp(tekst)

print("Lematyzacja:")
for token in doc:
    print(token.text, "->", token.lemma_)
