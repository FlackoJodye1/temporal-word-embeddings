import random
import numpy as np
from pprint import pprint

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot

pio.templates.default = "plotly"


# ------------------ Cosine plotting ------------------ #
def plot_cosine_similarity_tppmi_2(target_word, test_words, tppmi_model, selected_timesteps=None,
                                   elections=None, event_name="Elections",
                                   y_upper=0.8, save_path=None):
    # Use a baseline from provided test words
    words = test_words.copy()

    if selected_timesteps is None:
        selected_timesteps = [date for date in tppmi_model.dates]

    print(selected_timesteps)

    try:
        cosine_similarities = {
            key: {
                word: model.cosine_similarity(target_word, word) for word in words
            }
            for key, model in tppmi_model.ppmi_models.items() if key in selected_timesteps
        }
    except ValueError as e:
        print("All words need to be in the vocab of all timesteps.")
        print(e.args[0])
        return

    similarity_values = {word: [cosine_similarities[str(t)][word] for t in selected_timesteps] for word in words}

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')  # Set the figure's background color to white
    ax.set_facecolor('white')  # Set the axes' background color to white

    for word in words:
        ax.plot(selected_timesteps, similarity_values[word], marker='o', label=word)

    if elections:
        for term_end in elections:
            ax.axvline(x=term_end, color='gray', linestyle='--',
                       label=event_name if 'Elections' not in ax.get_legend_handles_labels()[1] else "")

    ax.set_ylabel('Cosine Similarity')
    ax.set_title(f'Cosine Similarity of "{target_word}" with Selected Test Words Over Time')
    ax.set_xticks(selected_timesteps)
    ax.set_xticklabels(selected_timesteps, rotation=90)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)  # Set grid lines
    ax.set_ylim(0, y_upper)
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    # Save the figure as an SVG file if a save path is provided
    if save_path:
        plt.savefig(save_path, format='svg')


def plot_cosine_similarity_tppmi(target_word, test_words, tppmi_model, selected_timesteps=None, event=None,
                                 event_name="",
                                 y_upper=1.1):
    # used for plotting a baseline
    words = test_words.copy()
    words.insert(0, target_word)

    if selected_timesteps is None:
        selected_timesteps = [date for date in tppmi_model.dates]

    try:
        cosine_similarities = {
            key: {
                word: model.cosine_similarity(target_word, word) for word in words
            }
            for key, model in tppmi_model.ppmi_models.items() if key in selected_timesteps
        }
    except ValueError as e:
        print("All words need to be in the vocab of all timesteps.")
        print(e.args[0])
        return

    similarity_values = {word: [cosine_similarities[str(t)][word] for t in selected_timesteps] for word in words}

    # Plotting
    plt.figure(figsize=(10, 6))
    for word in words:
        if word == target_word:
            plt.plot(selected_timesteps, similarity_values[word], marker='o', label=word, color="red")
        else:
            plt.plot(selected_timesteps, similarity_values[word], marker='o', label=word)

    if event:
        plt.axvline(x=event, color='gray', linestyle='--', label=event_name)

    plt.xlabel('Month')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity of "{target_word}" with selected Test-Words')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.ylim(0, y_upper)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.9))
    plt.tight_layout()
    plt.show()


def plot_cosine_similarity_cade(target_word, test_words, models, event=None, event_name=""):
    # used for plotting a baseline
    words = test_words.copy()
    words.insert(0, target_word)

    timesteps = [timestep.split("_")[1] for timestep in models.keys()]

    # Compute cosine similarities, assign 0 if a word is not in the model's vocabulary
    cosine_similarities = {
        key.split("_")[1]: {
            word: model.wv.similarity(target_word, word)
            if (word in model.wv.vocab and target_word in model.wv.vocab)
            else 0
            for word in words
        }
        for key, model in models.items()
    }

    similarity_values = {word: [cosine_similarities[t][word] for t in timesteps] for word in words}

    # Plotting
    plt.figure(figsize=(10, 6))
    for word in words:
        if word == target_word:
            plt.plot(timesteps, similarity_values[word], marker='o', label=word, color="red")
        else:
            plt.plot(timesteps, similarity_values[word], marker='o', label=word)

    if event:
        plt.axvline(x=event, color='gray', linestyle='--', label=event_name)

    plt.xlabel('Month')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarity of "{target_word}" with selected Test-Words')
    plt.grid(True)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------ 2D plotting ------------------ #

def plot_word_vectors_cade(models, words, perplexity=3, range=None, use_tsne=False):
    embeddings_list = {f"{word}_{key.split('_')[1]}": model.wv.get_vector(word) for key, model
                       in models.items() for word in words}

    embeddings_matrix = np.array(list(embeddings_list.values()))

    if use_tsne:
        embeddings_reduced = perform_tsne(embeddings_matrix, perplexity)
    else:
        embeddings_reduced = PCA(n_components=2).fit_transform(embeddings_matrix)

    word_vectors_dict = {word: reduced_embedding for word, reduced_embedding
                         in zip(embeddings_list.keys(), embeddings_reduced)}

    plot_word_vectors(word_vectors_dict, range=range)


def plot_word_vectors_tppmi(model, words, range=None, use_tsne=False, perplexity=3):
    embeddings_list = list(model.get_tppmi(words, smooth=True).values())

    concatenated_df = pd.concat(embeddings_list, ignore_index=False)

    embeddings_matrix = np.array(embeddings_list)
    embeddings_matrix = embeddings_matrix.reshape(-1, embeddings_matrix.shape[-1])

    if use_tsne:
        embeddings_reduced = perform_tsne(embeddings_matrix, perplexity)
    else:
        embeddings_reduced = PCA(n_components=2).fit_transform(embeddings_matrix)

    word_vectors_dict = {word: reduced_embedding for word, reduced_embedding
                         in zip(concatenated_df.index, embeddings_reduced)}

    plot_word_vectors(word_vectors_dict, range=range)


def plot_word_vectors(word_vectors_dict, range=None):
    if range is None:
        range = [-50, 50]
    unique_words = set([word.split("_")[0] for word in word_vectors_dict.keys()])
    color_map = px.colors.qualitative.Plotly  # Get a set of plotly colors
    word_color_dict = dict(zip(unique_words, color_map))

    data = []  # List to store scatter plot and line plot data
    prev_coords = {}  # Dictionary to store previous point's coordinates for each color

    for word, vectors_dict in word_vectors_dict.items():
        word_key = word.split("_")[0]
        pc1_values = vectors_dict[0]
        pc2_values = vectors_dict[1]
        color = word_color_dict[word_key]

        # Create scatter plot data for the point
        marker_trace = go.Scatter(
            x=[pc1_values],
            y=[pc2_values],
            mode='markers+text',  # Include text labels with markers
            name=word_key,
            showlegend=False,
            marker=dict(color=color),
            text=[word_key],  # Set the text label for the point
            textposition="top center"  # Position of the text label
        )
        data.append(marker_trace)

        # Check if there's a previous point for the same color
        if color in prev_coords:
            prev_pc1, prev_pc2 = prev_coords[color]
            # Create line plot data connecting the previous point and the current point
            line_trace = go.Scatter(
                x=[prev_pc1, pc1_values],
                y=[prev_pc2, pc2_values],
                mode='lines',
                showlegend=False,
                line=dict(color=color, width=1)
            )
            data.append(line_trace)
        else:
            # Create a specific symbol marker for the first point of each line
            start_marker_trace = go.Scatter(
                x=[pc1_values],
                y=[pc2_values],
                mode='markers',
                showlegend=False,
                marker=dict(symbol="diamond", size=10, color=color),
            )
            data.append(start_marker_trace)

        # Store the current point's coordinates as the previous coordinates for the color
        prev_coords[color] = (pc1_values, pc2_values)

    layout = dict(
        title="2D Visualization of Word Vectors",
        xaxis=dict(title="PC1", range=range),
        yaxis=dict(title="PC2", range=range),
        legend=dict(x=1.02, y=1.0)
    )

    # Create a Figure object with the data and layout
    fig = go.Figure(data=data, layout=layout)
    # Display the plot in a notebook or save to an HTML file

    init_notebook_mode(connected=True)
    iplot(fig, filename='word-embedding-plot')


def plot_temporal_changing_embedding(models, keyword, top_n=2, title="Word embeddings", subtitle="", use_tsne=False,
                                     use_plotly=True, range=None):
    """ 2D visualization of word-embeddings.
    Function can use either pca or tsne to reduce dimensions of the word vectors.
    Function can either use matplotlib or plotly for plotting"""
    # get most similar words for each timestep
    most_similar_list_for_reduction = [
        [tupel[0] for tupel in model.wv.most_similar(keyword, topn=top_n)] + [keyword]
        for model in models.values()
    ]

    most_similar_list_for_plotting = [
        [tupel[0] for tupel in model.wv.most_similar(keyword, topn=top_n)] + [
            keyword + "_" + str(key).split("_")[1]]
        for key, model in models.items()
    ]

    # merge them into a list for the viz-function
    word_list = [word for sublist in most_similar_list_for_plotting for word in sublist]

    # Iff use_tsne is true, perform t-SNE, otherwise perform PCA
    if use_tsne:
        vectors_2d_list = [perform_tsne(model.wv, words) for model, words in
                           zip(models.values(), most_similar_list_for_reduction)]
    else:
        vectors_2d_list = [perform_pca(model.wv, words) for model, words in
                           zip(models.values(), most_similar_list_for_reduction)]

    vectors_2d = np.vstack(vectors_2d_list)

    # Iff use_plotly is true, use plotly for plotting, otherwise matplotlib
    if use_plotly:
        plot_with_plotly(vectors_2d, word_list, keyword, title=title, subtitle=subtitle)
    else:
        plot_with_matplotlib(vectors_2d, word_list, keyword, title=title, subtitle=subtitle)


def plot_static_embedding(model, words, keyword, title="Words in the embedding space", subtitle=None, use_tsne=False,
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


def print_most_similar_cade(models, target_word, top_n=3):
    """
    Print the top N most similar words to a given target word across different models.

    :param models: A dictionary of models with keys as model identifiers.
    :param target_word: The word for which similar words are to be found.
    :param top_n: Number of top similar words to return. Defaults to 3.
    """

    if not isinstance(models, dict):
        raise ValueError("Models should be a dictionary")

    print(f"Word: {target_word}")
    for model_name, model in models.items():
        month = model_name.split('_')[1].capitalize()
        try:
            similar_words = model.wv.most_similar(target_word, topn=top_n)
            print(f"Month: {month}")
            pprint(similar_words)
        except KeyError:
            print(f"Month: {month}\n{target_word} not in vocab")
        print("--------------------------------")


# ------------------------------------------------------------------------- #
# ------------------------- Auxiliary functions --------------------------- #
# ------------------------------------------------------------------------- #


# ------------------ Methods for plotting ------------------ #

def plot_with_plotly(vectors, labels, keyword, title="Words in the embedding space", subtitle=None,
                     plot_in_notebook=True):
    # Create a list to hold scatter plot data points
    data = []
    text_offset = 0.05  # Offset for positioning text labels
    keyword_vectors = []

    # Iterate through each vector and label to create scatter plot data
    for i, (x, y) in enumerate(vectors):

        if labels[i].startswith(keyword):
            # keyword is red, other words are blue
            keyword_vectors.append((x, y))
            text_color = 'red'
        else:
            text_color = 'black'

        # Append a scatter plot trace for the current label
        data.append(go.Scatter(x=[x], y=[y], mode='markers+text', text=[labels[i]], textfont=dict(color=text_color),
                               marker=dict(color=text_color), textposition='top center'))

    # Iterate through keyword_vectors to draw lines between points
    for i in range(len(keyword_vectors) - 1):
        x_values = [keyword_vectors[i][0], keyword_vectors[i + 1][0]]
        y_values = [keyword_vectors[i][1], keyword_vectors[i + 1][1]]
        if i == 0:
            # Use a specific symbol for the first point of the line
            data.append(go.Scatter(x=[x_values[0]], y=[y_values[0]], mode='markers',
                                   marker=dict(symbol="diamond", size=10, color='red')))
        data.append(go.Scatter(x=x_values, y=y_values, mode='lines', line=dict(color='red')))

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


# ------------ Methods of dimensionality reduction ------------ #

def perform_pca(model, words):
    # Convert the word vectors of the specified words into a NumPy array

    word_vectors = np.array([model[w] for w in words])

    # 1. Apply PCA to reduce dimensionality to 2 components
    # 2. Perform the PCA transformation on the word vectors and select the first 2 dimensions
    return PCA().fit_transform(word_vectors)[:, :2]


def perform_tsne(embedding_matrix, perplexity):
    PERPLEXITY = perplexity

    # Initialize t-SNE with the specified perplexity and random state
    tsne = TSNE(n_components=2, perplexity=PERPLEXITY, random_state=1040)

    # Fit and transform the target_word_vectors using t-SNE
    tsne_vectors = tsne.fit_transform(embedding_matrix)

    # Fit the scaler on t-SNE vectors
    scaler = StandardScaler()
    tsne_vectors_scaled = scaler.fit_transform(tsne_vectors)

    return tsne_vectors_scaled
