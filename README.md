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

### Download Models

The Models are stored outside the repository on my personal Google Drive.
The sharable links are stored in the drive-urls.json. To download the models one has to run the download_models.py
script, parameters allows one to specify which models should be downloaded.

* **Static Word2Vec model**: python download_models.py --static
* **Cade models**: python download_models.py --cade
* **PPMI models**: python download_models.py --ppmi
* **All**: python download_models.py --static --cade --ppmi

## Files

### data-preprocessing.ipynb
Preprocessing the data and saving the relevant (and cleaned) data into csv files.
It also splits the data into time intervals needed to observe the effect of impactful events on the embeddings.
(no need to run it again)

### data-analysis.ipynb
Loading the data from the csv files, analysing the data and creating the models/word-embeddings.
The models are then stored in the model folder, which is part of repo
(no need to run it again)

### model-analysis.ipnyb
Loading the models, probing them with keywords and creating visualization, 
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split by month.

### model-analysis-quarterly.ipnyb
Loading the models, probing them with keywords and creating visualization,
which compare the embeddings before and after impactful events occured.
The models are trained on a corpus that is split into 4 quarters (Jun-Aug, Sep-Nov, Dez-Feb, Mar-April).

## Developments examined

* Elon Musk Twitter aquisition (27.10.2022)
* Shootings in the US
  * May: Uvalde Robb Elementary School shooting (May 24, 2022)
  * November: Colorado Springs nightclub shooting (November 19, 2022)
  * March: Nashville school shooting (March 27, 2023)
* President(s) (Trump, Biden etc.)

They are handpicked and one can clearly see the difference in the resulting embeddings of their keywords.