import sys
import requests
import os
from tqdm import tqdm
import torch

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)
if torch.backends.mps.is_available():
    device = torch.device("mps")
url = 'https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt'

checkpoints_dir = "./checkpoints/"

model_checkpoint_name = "vallex-checkpoint.pt"


def download_models_from_github():
    if not os.path.exists("./checkpoints"):
        model_url = "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt"
        whisper_url = "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
        os.makedirs("./whisper", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)

        print("Downloading token list...")
        os.makedirs("./utils/g2p", exist_ok=True)
        r = requests.get("https://raw.githubusercontent.com/korakoe/VALL-E-X/master/utils/g2p/bpe_69.json", stream=True)
        path = "./utils/g2p/bpe_69.json"
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()

        print("Downloading whisper...")
        r = requests.get(whisper_url, stream=True)
        path = './whisper/medium.pt'
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()

        print("Downloading VALL-E-X...")
        r = requests.get(model_url, stream=True)
        path = './checkpoints/vallex-checkpoint.pt'
        with open(path, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb", encoding='utf-8') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def main():
    if len(sys.argv) >= 3:
        file_id = sys.argv[1]
        destination = sys.argv[2]
    else:
        file_id = "TAKE_ID_FROM_SHAREABLE_LINK"
        destination = "DESTINATION_FILE_ON_YOUR_DISK"
    print(f"dowload {file_id} to {destination}")
    download_file_from_google_drive(file_id, destination)


if __name__ == "__main__":
    main()