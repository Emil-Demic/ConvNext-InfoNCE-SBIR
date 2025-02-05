import os
import time
import torch
import tqdm
import numpy as np

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from utils import init_model_transforms

register_heif_opener()

path_to_folder = "Photos"
files = os.listdir(path_to_folder)
files = [os.path.join(path_to_folder, file) for file in files]

model, transforms = init_model_transforms()

embeddings = []

start_time = time.time()

with torch.no_grad():
    for file in tqdm.tqdm(files):
        img = Image.open(file)
        img = ImageOps.exif_transpose(img)
        img = transforms(img).unsqueeze(0)
        embeddings.append(model(img).numpy()[0])

elapsed_time = time.time() - start_time
print("Time: ", elapsed_time)

embeddings = np.stack(embeddings)
files = np.stack(files)
np.save("embeddings.npy", embeddings)
np.save("files.npy", files)



