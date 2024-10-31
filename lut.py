#!/usr/bin/env python3
# pip3 install numpy colour-science imageio pillow

import argparse, os
from datetime import datetime
from typing import Union

import numpy as np
import imageio.v3 as iio
from colour.io.luts.iridas_cube import read_LUT_IridasCube, LUT3D, LUT3x1D
from PIL import Image


def replace_extension(path, new_extension):
    root, _ = os.path.splitext(path)
    return root + new_extension


# LUT

LUT = LUT3D | LUT3x1D


def apply_lut(img: np.ndarray, lut: LUT, color_size=255):
    im_array = np.asarray(img, dtype=np.float32) / color_size
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    dom_scale = None

    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        im_array = im_array * dom_scale + lut.domain[0]

    im_array = lut.apply(im_array)

    if is_non_default_domain:
        im_array = (im_array - lut.domain[0]) / dom_scale

    return np.clip(im_array * color_size, 0, color_size).astype(np.uint8)


def read_lut(lut_path: str, clip=False) -> LUT:
    '''
    Reads a LUT from the specified path, returning instance of LUT3D or LUT3x1D

    <lut_path>: the path to the file from which to read the LUT (
    <clip>: flag indicating whether to apply clipping of LUT values, limiting all values to the domain's lower and
        upper bounds
    '''
    lut: Union[LUT3x1D, LUT3D] = read_LUT_IridasCube(lut_path)
    lut.name = os.path.splitext(os.path.basename(lut_path))[0]  # use base filename instead of internal LUT name

    if clip:
        if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
            lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
        else:
            if len(lut.table.shape) == 2:  # 3x1D
                for dim in range(3):
                    lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
            else:  # 3D
                for dim in range(3):
                    lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

    return lut


# X3F


def parse_jpg(jpg_path: str, lut: LUT, scale: int = 1):
    with open(jpg_path, 'rb') as f:
        img = Image.open(f)

        if scale > 1:
            img = img.resize((int(img.width / scale), int(img.height / scale)))

        start = datetime.now()
        rgb = apply_lut(np.array(img), lut)
        print(f'{datetime.now() - start}: {jpg_path}')
        iio.imwrite(replace_extension(jpg_path, '.lut.jpg'), rgb, quality=98, subsampling='4:4:4')


# CLI

parser = argparse.ArgumentParser()
parser.add_argument('file', nargs='+', help='path of jpg files')
parser.add_argument('-l', '--lut', help='path to the .cube file')
parser.add_argument('-s', '--scale', type=int, help='scale down thumbnail', default=1)
args = parser.parse_args()

lut = read_lut(args.lut)
print('LUT:', lut.name)

for x in args.file:
    parse_jpg(x, lut, scale=args.scale)
