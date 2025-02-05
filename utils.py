import os
import random
import numpy as np
import scipy.spatial.distance as ssd
import torch

from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB
from torchvision.transforms import InterpolationMode

from config import args


def create_transforms():
    transforms = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RGB(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms


def seed_everything():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)


def compute_view_specific_distance(sketch_feats, image_feats):
    return ssd.cdist(sketch_feats, image_feats, 'sqeuclidean')


def output_html(sketch_index, image_indices, file_names):
    tmp_line = "<tr>"
    tmp_line += "<td><image src='%s' width=256 /></td>" % (
        os.path.join("fscoco", "raster_sketches", str(file_names[sketch_index])))
    for i in image_indices:
        if i != sketch_index:
            tmp_line += "<td><image src='%s' width=256 /></td>" % (
                os.path.join("fscoco", "images", str(file_names[i])))
        else:
            tmp_line += "<td ><image src='%s' width=256   style='border:solid 2px red' /></td>" % (
                os.path.join("fscoco", "images", str(file_names[i])))

    return tmp_line + "</tr>"


def calculate_results(dist, file_names):
    top1 = 0
    top5 = 0
    top10 = 0
    tmp_line = ""
    for i in range(dist.shape[0]):
        rank = dist[i].argsort()
        if rank[0] == i:
            top1 = top1 + 1
        if i in rank[:5]:
            top5 = top5 + 1
        if i in rank[:10]:
            top10 = top10 + 1
        tmp_line += output_html(i, rank[:10], file_names) + "\n"
    num = dist.shape[0]
    print(f' top1: {top1 / float(num):.4f} ({top1})')
    print(f' top5: {top5 / float(num):.4f} ({top5})')
    print(f'top10: {top10 / float(num):.4f} ({top10})')

    html_content = """
       <html>
       <head></head>
       <body>
       <table>%s</table>
       </body>
       </html>""" % tmp_line
    with open(r"result.html", 'w+') as f:
        f.write(html_content)
    return top1, top5, top10
