import os
import random

import cv2
import numpy as np
import scipy.spatial.distance as ssd
import torch
from bresenham import bresenham

from config import args

def drawPNG(sketch_data, side=256, time_frac=None, skip_front=False):
    raster_image = np.ones((side, side), dtype=np.uint8)
    prevX, prevY = None, None
    start_time = sketch_data[0]['timestamp']
    end_time = sketch_data[-1]['timestamp']

    if time_frac:
        if skip_front:
            start_time += (end_time - start_time) * time_frac
        else:
            end_time -= (end_time - start_time) * time_frac

    for points in sketch_data:
        time = points['timestamp']
        if not (start_time <= time <= end_time):
            continue

        x, y = map(float, points['coordinates'])
        x = int(x * side)
        y = int(y * side)
        pen_state = list(map(int, points['pen_state']))
        if not (prevX is None or prevY is None):
            cordList = list(bresenham(prevX, prevY, x, y))
            for cord in cordList:
                if (cord[0] > 0 and cord[1] > 0) and (cord[0] < side and cord[1] < side):
                    raster_image[cord[1], cord[0]] = 0
            if pen_state == [0, 1, 0]:
                prevX = x
                prevY = y
            elif pen_state == [1, 0, 0]:
                prevX = None
                prevY = None
            else:
                raise ValueError('pen_state not accounted for')
        else:
            prevX = x
            prevY = y
    # invert black and white pixels and dialate
    raster_image = (1 - cv2.dilate(1 - raster_image, np.ones((3, 3), np.uint8), iterations=1)) * 255
    return raster_image


def seed_everything():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


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
