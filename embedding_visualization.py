import random
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot

pio.templates.default = "plotly"


def plot_cosine_similarity_tppmi(target_word, test_words, tppmi_model, selected_months=None, event=None, event_name=""):
    # used for plotting a baseline
    words = test_words.copy()
    words.insert(0, target_word)

    if selected_months is None:
        selected_months = [f"{date.month:02d}" for date in tppmi_model.dates]

    try:
        cosine_similarities = {
            key: {
                word: model.cosine_similarity(target_word, word) for word in words
            }
            for key, model in tppmi_model.ppmi_models.items() if key in selected_months
        }
    except ValueError as e:
        print("All words need to be in the vocab of all timesteps.")
        print(e.args[0])
        return

    similarity_values = {word: [cosine_similarities[str(t)][word] for t in selected_months] for word in words}

    '''
    print("Similarity Value")
    print(similarity_values)'''


    # Plotting
    plt.figure(figsize=(10, 6))
    for word in words:
        if word == target_word:
            plt.plot(selected_months, similarity_values[word], marker='o', label=word, color="red")
        else:
            plt.plot(selected_months, similarity_values[word], marker='o', label=word)

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


def plot_cosine_similarity(target_word, test_words, models, event=None, event_name=""):
    # used for plotting a baseline
    words = test_words.copy()
    words.insert(0, target_word)

    timesteps = [timestep.split("_")[1] for timestep in models.keys()]

    cosine_similarities = {
        key.split("_")[1]: {
            word: model.wv.similarity(target_word, word) for word in words
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


def plot_word_vectors_tppmi(word_vectors_dict):
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
        xaxis=dict(title="PC1", range=[-50, 50]),
        yaxis=dict(title="PC2", range=[-50, 50]),
        legend=dict(x=1.02, y=1.0)
    )

    # Create a Figure object with the data and layout
    fig = go.Figure(data=data, layout=layout)
    # Display the plot in a notebook or save to an HTML file

    init_notebook_mode(connected=True)
    iplot(fig, filename='word-embedding-plot')


def plot_temporal_changing_embedding(keyword, models, top_n=2, title="Word embeddings", subtitle="", use_tsne=False,
                                     use_plotly=True):
    """ 2D visualization of word-embeddings.
    Function can use either pca or tsne to reduce dimensions of the word vectors.
    Function can either use matplotlib or plotly for plotting"""
    # get most similar words for each timestep
    most_similar_list_for_reduction = [
        [tupel[0] for tupel in model.wv.most_similar(keyword, topn=top_n)] + [keyword]
        for model in models.values()
    ]

    most_similar_list_for_plotting = [
        [tupel[0] + "_" + str(key).split("_")[1] for tupel in model.wv.most_similar(keyword, topn=top_n)] + [
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


# ------------------ Variants of plotting ------------------ #

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


# ------------ Variants of dimension reduction ------------ #

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
    perplexity = 2

    # 1. Apply t-SNE with 2 components, using perplexity adjusted based on the number of words
    # 2. Perform the t-SNE transformation on the word vectors
    tsne_vectors = TSNE(n_components=2, perplexity=perplexity, random_state=1040).fit_transform(word_vectors)
    # Fit the scaler on t-SNE vectors
    scaler = StandardScaler()
    tsne_vectors = scaler.fit_transform(tsne_vectors)

    return tsne_vectors


# currently not used
def sample_from_most_similar(model_pre, model_post, keyword, sample_size=6):
    # we only calculate the 100 most similar words
    sample_size = min(20, sample_size)

    similar_words_pre = model_pre.wv.most_similar(keyword, topn=20)
    similar_words_post = model_post.wv.most_similar(keyword, topn=20)

    sampled_words_pre = random.sample(similar_words_pre, sample_size // 2)
    sampled_words_post = random.sample(similar_words_post, sample_size // 2)
    sampled_words = sampled_words_pre + sampled_words_post

    return [key for key, _ in sampled_words]
