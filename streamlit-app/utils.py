import os
import sys

import torch
import torch.nn.functional as F

from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import SbirModel


def find_similar_embeddings(query_embedding, gallery_embeddings, top_k=10):
    query_embedding = torch.from_numpy(query_embedding).float()
    gallery_embeddings = torch.from_numpy(gallery_embeddings).float()
    similarities = F.pairwise_distance(query_embedding, gallery_embeddings)
    top_k_indices = torch.sort(similarities)[1][:top_k]
    return top_k_indices


def init_model_transforms():
    """
    Change the path to the pretrained model you wish to use. By default, the model trained
    on the unseen user train/test split is used
    """
    model = SbirModel(pretrained=False)
    model.load_state_dict(torch.load('../models/model_unseen.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    transforms = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RGB(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return model, transforms
