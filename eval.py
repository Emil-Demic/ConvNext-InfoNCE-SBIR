import tqdm
import torch

from torch.utils.data import DataLoader

from config import args
from model import SbirModel
from utils import compute_view_specific_distance, calculate_results, seed_everything, create_transforms
from data import DatasetFSCOCO

seed_everything()

transforms = create_transforms()

dataset_val = DatasetFSCOCO("fscoco", mode="val", transforms_sketch=transforms, transforms_image=transforms)

dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * 3, shuffle=False)

model = SbirModel(pretrained=False)

model_path = args.model_path
if model_path == '':
    if args.val_unseen:
        model_path = "models/model_unseen.pth"
    else:
        model_path = "models/model_normal.pth"

if args.cuda:
    model.cuda()
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

        output1 = model(data[0])
        output2 = model(data[1])

        sketch_output.append(output1.cpu())
        image_output.append(output2.cpu())

    sketch_output = torch.concatenate(sketch_output)
    image_output = torch.concatenate(image_output)

    dis = compute_view_specific_distance(sketch_output.numpy(), image_output.numpy())

    top1, top5, top10 = calculate_results(dis, dataset_val.get_file_names())

