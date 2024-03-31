import nltk
import re
import os
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
from nltk import ngrams, collocations
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora, models
from gensim.models import Word2Vec
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

# Downloading necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')
nltk.download('vader_lexicon')

def create_dataset_directory():
    base_dir = 'dataset_'
    n = int(time.time())  # Using timestamp to ensure a unique name
    dataset_dir = f"{base_dir}{n}"
    os.makedirs(dataset_dir, exist_ok=True)  # Creates the directory if it does not exist
    return dataset_dir

def plot_frequency_graph(data, title, dataset_dir, filename):
    plt.figure(figsize=(10, 6))
    
    # Tratamento especial para n-gramas: converte cada n-grama de tupla para string
    if title == "Top n-grams":
        terms = [' '.join(term[0]) for term in data]  # Converte cada n-grama (tupla) em string
    else:
        terms = [term[0] for term in data]
    
    counts = [count[1] for count in data]
    
    sns.barplot(x=counts, y=terms)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(dataset_dir, filename))

def analise_exploratoria(texto, dataset_dir):
    original_stdout = sys.stdout  # Saves the standard output
    with open(os.path.join(dataset_dir, 'saida.txt'), 'w') as f:
        sys.stdout = f  # Changes the standard output to the file

        # Text preprocessing
        texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto.lower())
        palavras = nltk.word_tokenize(texto_limpo)
        stop_words = set(nltk.corpus.stopwords.words('portuguese'))
        palavras_filtradas = [p for p in palavras if p not in stop_words]

        # Word count and visualization
        contagem_palavras = Counter(palavras_filtradas)
        freq_palavras = contagem_palavras.most_common(20)
        print("\n-------Most frequent words-------")
        print(freq_palavras)
        plot_frequency_graph(freq_palavras, "Most Frequent Words", dataset_dir, "frequent_words.png")

        # Word Cloud
        nuvem = WordCloud(width=800, height=400).generate(' '.join(palavras_filtradas))
        plt.figure(figsize=(10, 6))
        plt.imshow(nuvem, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(os.path.join(dataset_dir, 'word_cloud.png'))

        # n-grams analysis
        n = 3
        ngramas = ngrams(palavras_filtradas, n)
        freq_ngramas = Counter(ngramas)
        top_ngramas = freq_ngramas.most_common(20)
        print(f"\n-------Top {n}-grams-------")
        print(top_ngramas)
        plot_frequency_graph(top_ngramas, "Top n-grams", dataset_dir, "ngrams.png")

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        sentimentos = sia.polarity_scores(' '.join(palavras_filtradas))
        print("\n-------Sentiment Analysis-------")
        print(sentimentos)

        # Co-occurrence Network
        finder = collocations.BigramCollocationFinder.from_words(palavras_filtradas)
        scored = finder.score_ngrams(nltk.collocations.BigramAssocMeasures().raw_freq)
        rede = nx.Graph()
        for ng, score in scored:
            w1, w2 = ng
            rede.add_edge(w1, w2, weight=score)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(rede, seed=42)
        nx.draw(rede, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2500, font_size=10)
        plt.savefig(os.path.join(dataset_dir, 'cooccurrence_network.png'))

        # Word Embeddings and TSNE
        model = Word2Vec(sentences=[palavras_filtradas], vector_size=100, window=5, min_count=1, workers=4)
        word_vectors = np.array([model.wv[w] for w in model.wv.index_to_key])

        n_samples = len(word_vectors)
        perplexity_value = min(30, n_samples - 1)  # Ensures perplexity is less than the number of samples

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
        embeddings_2d = tsne.fit_transform(word_vectors)

        plt.figure(figsize=(12, 8))
        for i, word in enumerate(model.wv.index_to_key):
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
            plt.text(embeddings_2d[i, 0]+0.03, embeddings_2d[i, 1]+0.03, word, fontsize=9)
        plt.savefig(os.path.join(dataset_dir, 'embeddings_tsne.png'))

        sys.stdout = original_stdout  # Restores the standard output

if __name__ == '__main__':
    # Reading the content from a text file
    file_name = 'texto_analise.txt'
    with open(file_name, 'r', encoding='utf-8') as file:
        text_example = file.read()
    
    # Creating a unique dataset directory for this execution
    dataset_dir = create_dataset_directory()
    
    # Performing the exploratory analysis with the text from the file
    analise_exploratoria(text_example, dataset_dir)
