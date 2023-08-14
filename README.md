# temporal-word-embeddings

Source code of my bachelors thesis on the topic of temporal word embeddings

To see the progress only the ***model-analysis.ipynb*** needs to be run.

The models are already build and are stored in the /model directory. There are two models per event which are trained
separately on the corpus split at the given date. Alignment was enforced via the compass.

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
Loading the models, probing them with keywords and creating visualization, which compare the embeddings before and after impactful events occured.
(This is the only notebook that needs to run, since the models are already build)

## Impactful Events 

* Release of Brittney Grinner (08.12.2022)
* Elon Musk Twitter aquisition (27.10.2022)
* Attack on Paul Pelosi (28.10.2022)
* Colorado Springs nightclub shooting. (19.11.2022)
  
They are handpicked and one can clearly see the difference in the resulting embeddings of their keywords. 



