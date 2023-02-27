from flask import Flask, render_template, request
from gensim.models import LdaModel
import gensim
import spacy
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/infer', methods=['POST'])
def infer_conversation():
    input_text = request.form['input_text']

    # Preprocessing with the input text
    preprocessed_text = preprocess_text(input_text)
    preprocessed_text = apply_trigrams(preprocessed_text)
    preprocessed_text = apply_lemmatization(preprocessed_text)

    # Infer topic distribution using trained LDA model
    bow = lda_model.id2word.doc2bow(preprocessed_text)
    topic_distribution = lda_model.get_document_topics(bow)

    # Get the topic with the highest confidence
    max_topic = max(topic_distribution, key=lambda x: x[1])

    # retrieve the topic wordcloud image path
    image_path = f"./static/wordclouds/topic{max_topic[0]}.png"

    return render_template('index.html', image_path=image_path, confidence=max_topic[1])


def preprocess_text(doc):
    # Regex to remove URLs
    doc = re.sub(r'\bhttps?://\S+\b', '', doc) 

    # Regex to remove mentions, e.g. @UberSupport
    # doc = re.sub(r'@\w+\b', '', doc) 

    # Regex to remove signatures starting with ('^', '-' or '*')
    doc = re.sub(r'\B[-^*&]\s*\w+', '', doc) 
    
    # Regex to remove any symbols, emojis or non-western charachters
    doc = re.sub(r'[^a-zA-Z0-9\s,.?!;:()]+', '', doc) 
    
    # Function to convert document into lowercase, de-accents and tokenize
    doc = gensim.utils.simple_preprocess(doc, deacc=True, min_len=2)

    return doc


def apply_trigrams(doc):
    doc = [token for token in doc if token not in stopwords]
    doc = trigram_mod[bigram_mod[doc]]
    return doc


def apply_lemmatization(doc):
    doc = nlp(" ".join(doc))
    doc = [token.lemma_ for token in doc if token.pos_ in allowed_postags and len(token) > 2]
    return doc


if __name__ == '__main__':
    # Load Natural Languuege processing model
    nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
    stopwords = nlp.Defaults.stop_words
    stopwords.update(["hi", "hello", "hey", "et"])
    allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]

    # Load Phrases models
    bigram_mod = gensim.models.Phrases.load("./saved_models/bigram_mod.pkl")
    trigram_mod = gensim.models.Phrases.load("./saved_models/trigram_mod.pkl")

    # Load saved LDA model
    lda_model = LdaModel.load('./saved_models/LDAmodel')


    app.run(debug=True)