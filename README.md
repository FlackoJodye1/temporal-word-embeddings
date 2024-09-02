# Temporal Word Embeddings

## Datasets

### Social-Media Dataset

The datasets contain social-media posts from various platforms on the topic of education and focus mostly on the US.
It was used for sentiment-analysis since each observation has a sentiment attribute attached to it.
The recorded data spans 11 months, beginning on June 1, 2022, and ending on April 28, 2023.

### New-York-Times Dataset

This dataset contains 99,872 articles from the New York Times, published between January 1990 and July 2016.
It was introduced by Yao et al. 2018 in their paper "Dynamic Word Embeddings for Evolving Semantic Discovery"

## Project Organisation

------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── ppmi-matrices  <- PPMI-Matrices used to create the TPMMI-Model
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── test           <- Testsets for the quantitative evalutation
    │
    ├── models             <- Trained and serialized models: TWEC, StaticWord2Vec
    │
    ├── notebooks
    │   ├── analysis-qualitative       <- Visualiziations in 2D and cosine-similarity plots
    │   ├── analysis-quantitative      <- MP@K, MRR@K evalutations of models on NYT-Data, Sensitivity-Analysis of TPPMI
    │   ├── preprocessing              <- Preprocessing of NYT-Data & Social-Media-Data
    │   └── training                   <- Training the models
    │
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── dataset           
    │   │   └── download.py     <- Script to download the data if available
    │   │
    │   ├── packages        <- Implementation of the TPPMI-Model as a package
    │   │   └── ppmi_model.py     <- Implementation of the ppmi model
    │   │   └── tppmi_creation.ipynb     <- Notebook to create the tppmi model
    │   │   └── tppmi_creation.py     <- Utility functions that are used in the tppmi_creation notebook
    │   │   └── tppmi_functions.py     <- Additional utility functions
    │   │   └── tppmi_model.py     <- Implementation of the tppmi model
    │   │   └── util.py     <- Additional utility functions
    │   │
    │   ├── test            <- Utility functions for the quantitative-evaluation
    │   │
    │   └── visualizations  <- Utility functions for the qualitative-evaluation


--------

## Setup


### Installation

To clone the repository to your local machine run:

* git clone https://github.com/FlackoJodye1/temporal-word-embeddings.git

For the installation of the necessary packages you need install the dependencies specified in
the environment.yml. To do so use the following command:

* conda env create -f environment.yml

### Auxilliary datasets

1. To get the pantheon-entity-dataset, download it from the page here:
https://www.nature.com/articles/sdata201575

2. Store it in the data/raw directory.

3. Now run the preprocessing-pantheon-dataset.ipynb

### Download Datasets & Models

The datasets & models are stored outside the repository on my personal Google Drive.
To download them, you need to run the download.py script.
This will create 2 directories (data & model) with everything necessary in it, to run the notebooks.
(Can take up to 10 minutes)

* python src/dataset/download.py

## How to reprocude

1. Installation as described above
2. Get the auxiliary dataset as described above
3. Get the nyt-dataset (unfortunately no longer publicly available)
4. Run the preprocessing notebooks:
- Run preprocess-pantheon-dataset.ipynb
- Run filter-testsets.ipynb
- Run preprocess-nyt-data.ipynb
8. Run the training notebooks:
- Run create-ppmi-nyt-data.ipynb
- Run train-models-nyt-data.ipynb
9. Run model-comparison.ipynb

## Notebooks

### Preprocessing

#### preprocessing-social-media-data.ipynb
Preprocessing the data and saving the relevant (and cleaned) data into csv files.
It also splits the data into time intervals needed to observe the effect of impactful events on the embeddings.
(no need to run it again)

#### preprocessing-nyt-data.ipynb

Preprocessing the data from the New-York Times dataset

#### preprocessing-pantheon-dataset.ipynb

Preprocessing the pantheon-dataset which is used to filter the testsets

#### filter-testsets.ipynb

Creating a filtered_testset version of a supplied testset.
It will only include those testcases which token/name is of the entity PERSON

### Training

This directory contains notebooks for training the Word-Embedding models on the different datasets.
First the ppmi-models have to be created before they can be subsequently used to create the tppmi models 
in the train-models-... notebooks.

### Qualitative Analysis

These notebooks are not part of the quantitative experiment.

#### model-analysis-nyt-corpus

This notebook creates the graphs/visuals relating to the experiment on the NYT-Data.

#### model-analysis-monthly.ipynb
Loading the models, probing them with keywords and creating visualization,
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split by month.
It uses the social-media data and not the NYT-Dataset.

#### model-analysis-quarterly.ipynb
Loading the models, probing them with keywords and creating visualization,
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split into 4 quarters (Jun-Aug, Sep-Nov, Dez-Feb, Mar-April).
It uses the social-media data and not the NYT-Dataset.

#### nyt-data-model-comparison.ipynb

This notebook is used to explore the differences between the models and their output in more detail.
It uses the social-media data and not the NYT-Dataset.

### Quantitative Analysis

#### model-comparison

Reproduction of the Experiment as introduced by Valerio Di Carlo et al. in their paper 
"Training Temporal Word Embeddings with a Compass". The training set for this experiment is the NYT-Dataset and the test set 
was introduced by Yao et al. It contains temporal word analogies and contains approx. 11.000 testcases.
Each model has to calculate these analogies in their embedding space before the metrics MP@K, MRR@K are used to calculate scores.
The testset is filtered  to only include names of entity "PERSON" as described in the paper

* **Tested Models**: TWEC, TPPMI, StaticWord2Vec(Baseline) 

#### sensititvity_analysis_tppmi.ipynb

Evaluates TPPMI-Models based on the variation in the number of context-words they use,
which fundamentally represents their embedding dimension.
The context-words are randomly sampled from the most-common words in the corpus.
In the models that utilize 200, 500, and 1000 context-words,
I select samples from the top 2000 most frequent words. Meanwhile,
for the model with 5000 context-words, the sampling is done from the 10,000 most commonly used words.

* **Tested Numbers of dimensions**: 200, 500, 1000, 5000