import os
import json
import gdown
import argparse

# Define the JSON filename as a constant
JSON_FILENAME = "drive-urls.json"


def load_model_data(json_filename):
    # Load model data from the JSON file
    with open(json_filename, "r") as json_file:
        model_data = json.load(json_file)
    return {entry["type"].lower(): entry["models"] for entry in model_data.get("models", [])}


def download_static_model(model_dict):
    if "static" in model_dict:
        static_download_data = model_dict.get("static", [])
        if not os.path.exists("model"):
            os.mkdir("model")
        # Create the "monthly" directory if it does not exist
        if not os.path.exists("model/static"):
            os.mkdir("model/static")
        for file_name, file_id in static_download_data[0].items():
            download_link = f"https://drive.google.com/uc?id={file_id}"
            destination_path = os.path.join("model/static", file_name)
            gdown.download(download_link, destination_path, quiet=False)
    else:
        print("Static model not found in the JSON data.")


def download_ppmi_models(model_dict):
    if "ppmi-models" in model_dict:
        ppmi_download_data = model_dict.get("ppmi-models", [])
        if not os.path.exists("data"):
            os.mkdir("data")
        if not os.path.exists("data/ppmi-matrices"):
            os.mkdir("data/ppmi-matrices")
        for data in ppmi_download_data:
            print(data)
            matrix_download_link = f"https://drive.google.com/uc?id={data['matrix_file_id']}"
            vocab_download_link = f"https://drive.google.com/uc?id={data['vocab_file_id']}"
            matrix_destination_path = f"data/ppmi-matrices/{data['name']}.npz"
            vocab_destination_path = f"data/ppmi-matrices/{data['name']}.pkl"
            gdown.download(matrix_download_link, matrix_destination_path)
            gdown.download(vocab_download_link, vocab_destination_path)
    else:
        print("PPMI models not found in the JSON data.")


def download_cade_models(model_dict):
    # Check if the "cade" model type exists in the model dictionary
    if "cade" in model_dict:
        cade_download_data = model_dict.get("cade", [])
        # Create the "model" directory if it does not exist
        if not os.path.exists("model"):
            os.mkdir("model")
        # Create the "monthly" directory if it does not exist
        if not os.path.exists("model/monthly"):
            os.mkdir("model/monthly")
        for data in cade_download_data:
            download_link = f"https://drive.google.com/uc?id={data['file_id']}"
            model_name = f"model/monthly/{data['name']}"
            # Download Cade models
            gdown.download(download_link, model_name, quiet=False)
    else:
        print("Cade models not found in the JSON data.")


if __name__ == "__main__":
    # Create an argument parser

    parser = argparse.ArgumentParser(description="Download models from Google Drive.")

    # Add an argument for model selection
    parser.add_argument("--static", action="store_true", help="Download static Word2Vec model")
    parser.add_argument("--cade", action="store_true", help="Download Cade models")
    parser.add_argument("--ppmi", action="store_true", help="Download PPMI models")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load model data from the JSON file
    model_dict = load_model_data(JSON_FILENAME)

    # Download the selected models based on command-line arguments
    if args.static:
        download_static_model(model_dict)

    if args.cade:
        download_cade_models(model_dict)

    if args.ppmi:  # Added this condition
        download_ppmi_models(model_dict)
