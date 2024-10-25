import tqdm
import torch

from info_nce import InfoNCE
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB
from torchvision.transforms import InterpolationMode

from config import args
from model import SbirModel
from utils import compute_view_specific_distance, calculate_results, seed_everything
from data import DatasetFSCOCO

seed_everything()

transforms = Compose([
            RGB(),
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

dataset_train = DatasetFSCOCO("fscoco", mode="train", transforms_sketch=transforms, transforms_image=transforms)
dataset_val = DatasetFSCOCO("fscoco", mode="val", transforms_sketch=transforms, transforms_image=transforms)

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * 3, shuffle=False)

model = SbirModel(pretrained=True)
if args.cuda:
    model.cuda()

optimizer = Adam(model.parameters(), lr=args.lr)

loss_fn = InfoNCE(negative_mode="unpaired", temperature=args.temp)

best_res = 0
best_top1 = 0
no_improvement = 0
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader_train):
        if args.cuda:
            data = [d.cuda() for d in data]

        output = model(data)

        loss = loss_fn(output[0], output[1])

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 4:
            print(f'[{epoch:03d}, {i:03d}] loss: {running_loss / 5  :0.5f}')
            running_loss = 0.0

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

        print(f"EPOCH {str(epoch)}:")
        top1, top5, top10 = calculate_results(dis, dataset_val.get_file_names())

        if top10 > best_res:
            no_improvement = 0
            best_res = top10
            best_top1 = top1
            if args.save:
                torch.save(model.state_dict(), f"E{epoch}_model.pth")
        else:
            if args.save and top1 > best_top1 and top10 == best_res:
                best_top1 = top1
                torch.save(model.state_dict(), f"E{epoch}_model.pth")
            no_improvement += 1
            if no_improvement == 2:
                print("top10 metric has not improved for 2 epochs. Ending training.")
                break
