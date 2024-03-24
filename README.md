# Temporal Word Embeddings

Source code of my bachelors thesis on the topic of temporal word embeddings

To see the progress only the ***model-analysis.ipynb*** needs to be run.

The models are already build and are stored in the /model directory. There are two models per event which are trained
separately on the corpus split at the given date. Alignment was enforced via the compass.

## Dataset(s)

The datasets contain social-media posts from various platforms on the topic of education and focus mostly on the US.
It was used for sentiment-analysis since each observation has a sentiment attribute attached to it.
The recorded data spans 11 months, beginning on June 1, 2022, and ending on April 28, 2023.

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

## Files

### data-preprocessing.ipynb
Preprocessing the data and saving the relevant (and cleaned) data into csv files.
It also splits the data into time intervals needed to observe the effect of impactful events on the embeddings.
(no need to run it again)

### data-analysis.ipynb
Loading the data from the csv files, analysing the data and creating the models/word-embeddings.
The models are then stored in the model folder, which is part of repo
(no need to run it again)

### model-analysis.ipynb
Loading the models, probing them with keywords and creating visualization, 
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split by month.

### model-analysis-quarterly.ipynb
Loading the models, probing them with keywords and creating visualization,
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split into 4 quarters (Jun-Aug, Sep-Nov, Dez-Feb, Mar-April).

### tppmi_sensititvity_analysis.ipynb

Evaluates TPPMI-Models based on the variation in the number of context-words they use, 
which fundamentally represents their embedding dimension.
The context-words are randomly sampled from the most-common words in the corpus.
In the models that utilize 200, 500, and 1000 context-words, 
I select samples from the top 2000 most frequent words. Meanwhile, 
for the model with 5000 context-words, the sampling is done from the 10,000 most commonly used words.

* Tested Numbers of dimensions: 200, 500, 1000, 5000 

## Developments examined

* Elon Musk Twitter aquisition (27.10.2022)
* Shootings in the US
  * May: Uvalde Robb Elementary School shooting (May 24, 2022)
  * November: Colorado Springs nightclub shooting (November 19, 2022)
  * March: Nashville school shooting (March 27, 2023)
* President(s) (Trump, Biden etc.)

They are handpicked and one can clearly see the difference in the resulting embeddings of their keywords.