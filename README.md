# Temporal Word Embeddings

Source code of my bachelors thesis on the topic of temporal word embeddings

To see the progress only the notebooks in the qualitative/quantitative-analysis directories need to be run.

The models are already build and are stored in the /model directory.

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
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── dataset           
    │   │   └── download.py     <- Script to download the data
    │   │
    │   ├── packages        <- Implementation of the TPPMI-Model as a package
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

### Download Datasets & Models

The datasets & models are stored outside the repository on my personal Google Drive.
To download them, you need to run the download.py script.
This will create 2 directories (data & model) with everything necessary in it, to run the notebooks.
(Can take up to 10 minutes)

* python src/dataset/download.py

## Notebooks

### Qualitative Analysis

#### model-analysis-monthly.ipynb
Loading the models, probing them with keywords and creating visualization,
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split by month.

#### model-analysis-quarterly.ipynb
Loading the models, probing them with keywords and creating visualization,
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split into 4 quarters (Jun-Aug, Sep-Nov, Dez-Feb, Mar-April).

### Quantitative Analysis

#### reproduction-twec-results

Reproduction of the Experiment as introduced by Valerio Di Carlo et al. in their paper 
"Training Temporal Word Embeddings with a Compass". The training set for this experiment is the NYT-Dataset and the test set 
was introduced by Yao et al. It contains temporal word analogies and contains approx. 11.000 testcases.
Each model has to calculate these analogies in their embedding space before the metrics MP@K, MRR@K are used to calculate scores.

* **Tested Models**: TWEC, TPPMI, StaticWord2Vec(Baseline) 

#### tppmi_sensititvity_analysis.ipynb

Evaluates TPPMI-Models based on the variation in the number of context-words they use,
which fundamentally represents their embedding dimension.
The context-words are randomly sampled from the most-common words in the corpus.
In the models that utilize 200, 500, and 1000 context-words,
I select samples from the top 2000 most frequent words. Meanwhile,
for the model with 5000 context-words, the sampling is done from the 10,000 most commonly used words.

* **Tested Numbers of dimensions**: 200, 500, 1000, 5000

### Preprocessing

#### preprocessing-social-media-data.ipynb
Preprocessing the data and saving the relevant (and cleaned) data into csv files.
It also splits the data into time intervals needed to observe the effect of impactful events on the embeddings.
(no need to run it again)

#### preprocessing-nyt-data.ipynb

Preprocessing the data from the New-York Times dataset 

### Training

This directory contains notebooks for training the Word-Embedding models on the different datasets.