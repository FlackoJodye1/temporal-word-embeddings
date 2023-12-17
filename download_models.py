import gdown

DATA_DIR_LINK = "https://drive.google.com/drive/folders/1TWC7-3uuUKA7qlEQTdPDo0XUh5ZEwG-k"
MODEL_DIR_LINK = "https://drive.google.com/drive/folders/1U7c4M-62hosM1xpXwPy-lqOKr3Ni8W2P"

gdown.download_folder(DATA_DIR_LINK)
gdown.download_folder(MODEL_DIR_LINK)
