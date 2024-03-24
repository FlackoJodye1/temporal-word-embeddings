import gdown

DATA_URL = "https://drive.google.com/drive/folders/19d-_jJDaW2Vd50mcnOQWiGJZ-Q5OBbWZ"
MODEL_URL = "https://drive.google.com/drive/folders/1QFFl3yh-yaJKQUd2l5JL69zjfhfkIjfY"

gdown.download_folder(DATA_URL, remaining_ok=True)
gdown.download_folder(MODEL_URL, remaining_ok=True)
