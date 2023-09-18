import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
pio.templates.default = "plotly"


def display_temporal_changing_embedding(keyword, models, top_n=2, title="Word embeddings", subtitle="", use_tsne=False,
                                        use_plotly=True):
    # get most similar words for each timestep
    most_similar_list = [
        [tupel[0] for tupel in model.wv.most_similar(keyword, topn=top_n)] + [keyword]
        for model in models
    ]

    # merge them into a list for the viz-function
    word_list = [word for sublist in most_similar_list for word in sublist]

    # Iff use_tsne is true, perform t-SNE, otherwise perform PCA
    if use_tsne:
        vectors_2d_list = [perform_tsne(model.wv, words) for model, words in zip(models, most_similar_list)]
    else:
        vectors_2d_list = [perform_pca(model.wv, words) for model, words in zip(models, most_similar_list)]

    vectors_2d = np.vstack(vectors_2d_list)

    # Iff use_plotly is true, use plotly for plotting, otherwise matplotlib
    if use_plotly:
        plot_with_plotly(vectors_2d, word_list, keyword, title=title, subtitle=subtitle)
    else:
        plot_with_matplotlib(vectors_2d, word_list, keyword, title=title, subtitle=subtitle)


def display_static_embedding(model, words, keyword, title="Words in the embedding space", subtitle=None, use_tsne=False,
                             use_plotly=False):
    # we do not want to change mutable parameters
    word_list = words.copy()

    # keywords not in words will otherwise not get visualized
    if keyword not in word_list:
        word_list.append(keyword)

    # Iff use_tsne is true, perform t-SNE, otherwise perform PCA
    if use_tsne:
        vectors_2d = perform_tsne(model.wv, word_list)
    else:
        vectors_2d = perform_pca(model.wv, word_list)

    # Iff use_plotly is true, use plotly for plotting, otherwise matplotlib
    if use_plotly:
        plot_with_plotly(vectors_2d, word_list, keyword, title=title, subtitle=subtitle)
    else:
        plot_with_matplotlib(vectors_2d, word_list, keyword, title=title, subtitle=subtitle)


def plot_with_plotly(vectors, labels, keyword, title="Words in the embedding space", subtitle=None,
                     plot_in_notebook=True):
    # Create a list to hold scatter plot data points
    data = []
    text_offset = 0.05  # Offset for positioning text labels

    # Iterate through each vector and label to create scatter plot data
    for i, (x, y) in enumerate(vectors):
        # keyword is red, other words are blue
        text_color = 'red' if labels[i].startswith(keyword) else 'black'
        # Append a scatter plot trace for the current label
        data.append(go.Scatter(x=[x], y=[y], mode='markers+text', text=[labels[i]], textfont=dict(color=text_color),
                               marker=dict(color=text_color), textposition='top center'))

    # Define the layout settings for the plot
    layout = go.Layout(
        title=title,
        titlefont=dict(size=22),
        title_x=0.5,  # Center the title horizontally
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        showlegend=False,
        annotations=[
            dict(
                x=0.5,
                y=1.08,
                xref='paper',
                yref='paper',
                text=subtitle,
                showarrow=False,
            )
        ],
    )

    # Create a Figure object with the data and layout
    fig = go.Figure(data=data, layout=layout)

    # Display the plot in a notebook or save to an HTML file
    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(fig, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(vectors, labels, keyword, title="Words in the embedding space", subtitle=None):
    # Create a scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(vectors[:, 0], vectors[:, 1], edgecolors='k', c='orange')

    # Add word labels to the scatter plot
    for word, (x, y) in zip(labels, vectors):
        color = 'red' if word == keyword else 'orange'
        plt.text(x + 0.02, y + 0.02, word, color=color)

    # Set plot titles
    plt.suptitle(title, fontsize=14)
    if subtitle:
        plt.title(subtitle, fontsize=10)


# ------ Dimensionality reduction ------ #

def perform_pca(model, words):
    # Convert the word vectors of the specified words into a NumPy array

    word_vectors = np.array([model[w] for w in words])

    # 1. Apply PCA to reduce dimensionality to 2 components
    # 2. Perform the PCA transformation on the word vectors and select the first 2 dimensions
    return PCA().fit_transform(word_vectors)[:, :2]


def perform_tsne(model, words):
    # Convert the word vectors of the specified words into a NumPy array
    word_vectors = np.array([model[w] for w in words])

    # perplexity must not exceed the number of samples
    perplexity = min(5, len(words) - 1)

    # 1. Apply t-SNE with 2 components, using perplexity adjusted based on the number of words
    # 2. Perform the t-SNE transformation on the word vectors
    tsne_vectors = TSNE(n_components=2, perplexity=perplexity, random_state=1040).fit_transform(word_vectors)
    # Fit the scaler on t-SNE vectors
    scaler = StandardScaler()
    tsne_vectors = scaler.fit_transform(tsne_vectors)

    return tsne_vectors
