import tqdm
import torch

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB
from torchvision.transforms import InterpolationMode

from config import args
from model import SbirModel
from utils import compute_view_specific_distance, calculate_results, seed_everything
from data import DatasetFSCOCO

seed_everything()

transforms = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RGB(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

dataset_val = DatasetFSCOCO("fscoco", mode="val", transforms_sketch=transforms, transforms_image=transforms)

dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * 3, shuffle=False)

model = SbirModel(pretrained=False)
if args.cuda:
    model.cuda()

model_path = args.model_path
if model_path == '':
    if args.val_unseen:
        model_path = "models/model_unseen.pth"
    else:
        model_path = "models/model_normal.pth"

if args.cuda:
    model.load_state_dict(torch.load(model_path, weights_only=True))
else:
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))


with torch.no_grad():
    model.eval()

    sketch_output = []
    image_output = []
    for data in tqdm.tqdm(dataloader_val):
        if args.cuda:
            data = [d.cuda() for d in data]

        output = model(data)
        sketch_output.append(output[0].cpu())
        image_output.append(output[1].cpu())

    sketch_output = torch.concatenate(sketch_output)
    image_output = torch.concatenate(image_output)

    dis = compute_view_specific_distance(sketch_output.numpy(), image_output.numpy())

    top1, top5, top10 = calculate_results(dis, dataset_val.get_file_names())

