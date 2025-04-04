import tqdm
import torch

from info_nce import InfoNCE
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import args
from model import SbirModel
from utils import compute_view_specific_distance, calculate_results, seed_everything, create_transforms
from data import DatasetFSCOCO

seed_everything()

transforms = create_transforms()

dataset_train = DatasetFSCOCO("fscoco", mode="train", transforms_sketch=transforms, transforms_image=transforms)
dataset_val = DatasetFSCOCO("fscoco", mode="val", transforms_sketch=transforms, transforms_image=transforms)

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * 3, shuffle=False)

model = SbirModel(pretrained=True)
if args.cuda:
    model.cuda()

optimizer = Adam(model.parameters(), lr=args.lr)

loss_fn = InfoNCE(negative_mode="unpaired", temperature=args.temp)

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader_train):
        if args.cuda:
            data = [d.cuda() for d in data]

        output1 = model(data[0])
        output2 = model(data[1])

        loss = loss_fn(output1, output2)

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

            output1 = model(data[0])
            output2 = model(data[1])

            sketch_output.append(output1.cpu())
            image_output.append(output2.cpu())

        sketch_output = torch.concatenate(sketch_output)
        image_output = torch.concatenate(image_output)

        dis = compute_view_specific_distance(sketch_output.numpy(), image_output.numpy())

        print(f"EPOCH {str(epoch)}:")
        top1, top5, top10 = calculate_results(dis, dataset_val.get_file_names())

        if args.save:
            torch.save(model.state_dict(), f"E{epoch}_model.pth")

