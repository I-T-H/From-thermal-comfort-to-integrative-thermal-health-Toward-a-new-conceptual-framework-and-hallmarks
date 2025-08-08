import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import string
import nltk
from gensim.models.coherencemodel import CoherenceModel
import os

# Download necessary NLTK tools if not already downloaded
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text, custom_stopwords=None):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    lemmatizer = WordNetLemmatizer()
    words_with_pos = pos_tag(words)
    words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in words_with_pos
        if word not in stop_words and len(word) > 1
    ]
    return words

def compute_coherence_score(lda_model, texts, dictionary, coherence_type='c_v'):
    coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence=coherence_type)
    return coherence_model.get_coherence()

def find_optimal_topics(processed_texts, dictionary, corpus, start=8, end=20, step=1):
    scores = []
    for num_topics in range(start, end + 1, step):
        print(f"Training LDA model with {num_topics} topics...")
        lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20, random_state=42)
        coherence_score = compute_coherence_score(lda_model, processed_texts, dictionary)
        scores.append((num_topics, coherence_score))
        print(f"Number of topics: {num_topics}, Coherence score: {coherence_score:.4f}")
    best_num_topics, best_score = max(scores, key=lambda x: x[1])
    print(f"\nOptimal number of topics: {best_num_topics}, Coherence score: {best_score:.4f}")
    return best_num_topics, best_score

def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder does not exist: {model_path}")
    lda_model_path = os.path.join(model_path, "lda_model.gensim")
    dictionary_path = os.path.join(model_path, "dictionary.txt")
    corpus_path = os.path.join(model_path, "corpus.mm")
    if not os.path.exists(lda_model_path):
        raise FileNotFoundError(f"LDA model file missing: {lda_model_path}")
    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"Dictionary file missing: {dictionary_path}")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus file missing: {corpus_path}")
    lda_model = gensim.models.LdaModel.load(lda_model_path)
    dictionary = corpora.Dictionary.load_from_text(dictionary_path)
    corpus = corpora.MmCorpus(corpus_path)
    print(f"Loaded LDA model, dictionary, and corpus from: {model_path}")
    return lda_model, dictionary, corpus

def save_document_topics(lda_model, corpus, output_path):
    """
    Save the topic distribution of each document to a CSV file.
    :param lda_model: Trained LDA model
    :param corpus: Document corpus
    :param output_path: Save path
    """
    print("\nSaving document topic distributions to file...")
    doc_topics = []
    for doc_id, doc in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(doc)
        doc_topics.append([doc_id] + [prob for _, prob in topic_distribution])
    df = pd.DataFrame(doc_topics)
    df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')
    print(f"Document topic distributions saved to: {output_path}")

def lda_analysis(file_path, custom_stopwords=None, save_results=False, model_path="lda_model"):
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "lda_model.gensim")):
        print("Detected saved model. Loading...")
        lda_model, dictionary, corpus = load_saved_model(model_path)
    else:
        print("No saved model detected. Starting training...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File path does not exist: {file_path}")
        data = pd.read_csv(file_path, encoding='utf-8-sig')
        if 'Abstract' not in data.columns:
            raise ValueError("Column 'Abstract' not found in the dataset. Please check the file format!")
        abstracts = data['Abstract'].dropna().tolist()
        processed_texts = [preprocess_text(text, custom_stopwords) for text in abstracts]
        dictionary = corpora.Dictionary(processed_texts)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]
        best_num_topics, best_score = find_optimal_topics(processed_texts, dictionary, corpus, start=8, end=15, step=1)
        lda_model = gensim.models.LdaModel(corpus, num_topics=best_num_topics, id2word=dictionary, passes=20, random_state=42)
        if save_results:
            os.makedirs(model_path, exist_ok=True)
            lda_model.save(os.path.join(model_path, "lda_model.gensim"))
            dictionary.save_as_text(os.path.join(model_path, "dictionary.txt"))
            corpora.MmCorpus.serialize(os.path.join(model_path, "corpus.mm"), corpus)
            print(f"\nLDA model, dictionary, and corpus saved to folder: {model_path}")
    print("\nIdentified topics and keywords:")
    for idx, topic in lda_model.print_topics(num_words=10):
        print(f"Topic {idx + 1}: {topic}")
    if save_results:
        output_path = os.path.join(model_path, "document_topics.csv")
        save_document_topics(lda_model, corpus, output_path)
    return lda_model, corpus, dictionary

if __name__ == "__main__":
    custom_stopwords = ['study', 'result', 'data', 'analysis', "95", "wbgt", "ieq", "ci", "cwi", "degrees", "min", "conditions", "research", "air", "using", "used", "patients", "admissions", "model","water","005","population","high"]
    abstract_file = "filtered_abstract"
    model_path = "model"
    lda_model, corpus, dictionary = lda_analysis(
        file_path=abstract_file,
        custom_stopwords=custom_stopwords,
        save_results=True,
        model_path=model_path
    )
