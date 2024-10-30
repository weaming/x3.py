#!/usr/bin/env python3
# pip3 install numpy==1.26.4 rawpy colour-science imageio pillow
# useful repos:
# https://github.com/duolanda/smart-lut-creator

import argparse, os
from datetime import datetime
from typing import Union
from io import BytesIO
import subprocess

import rawpy
from rawpy import ColorSpace, FBDDNoiseReductionMode, HighlightMode, Thumbnail
import numpy as np
import imageio.v3 as iio
from colour.io.luts.iridas_cube import read_LUT_IridasCube, LUT3D, LUT3x1D
from PIL import Image


def replace_extension(path, new_extension):
    root, _ = os.path.splitext(path)
    return root + new_extension


# LUT


# inspired by https://github.com/yoonsikp/pycubelut/blob/master/pycubelut.py
def apply_lut(img: np.ndarray, lut: Union[LUT3D, LUT3x1D], color_size=65535):
    im_array = np.asarray(img, dtype=np.float32) / color_size
    is_non_default_domain = not np.array_equal(lut.domain, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    dom_scale = None

    if is_non_default_domain:
        dom_scale = lut.domain[1] - lut.domain[0]
        im_array = im_array * dom_scale + lut.domain[0]

    im_array = lut.apply(im_array)

    if is_non_default_domain:
        im_array = (im_array - lut.domain[0]) / dom_scale

    return np.clip(im_array * color_size, 0, color_size).astype(np.uint16)


def read_lut(lut_path: str, clip=False) -> Union[LUT3D, LUT3x1D]:
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


# EXIF
# exiftool -G1 -a -s -EXIF:all SDQ01.X3F


def get_orientation(raw_path):
    process = subprocess.Popen(
        f'exiftool -G1 -a -s -EXIF:all "{raw_path}" | grep -v IFD1 | grep Orientation',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = process.communicate()
    orientation = str(output).split(':')[1].strip()

    # Counter ClockWise
    # 0=none, 3=180, 5=90CCW, 6=90CW
    if '270' in orientation:  # Rotate 270 CW
        return 5, 90
    elif '90' in orientation:
        return 6, 270
    elif '180' in orientation:
        return 3, 180
    return None, None


# X3F


def write_layers(layers):
    b = layers[:, :, 0]
    g = layers[:, :, 1]
    r = layers[:, :, 2]
    iio.imwrite(f'b.tiff', b)
    iio.imwrite(f'g.tiff', g)
    iio.imwrite(f'r.tiff', r)
    iio.imwrite(f'raw.tiff', layers)


def parse_x3f(raw_path: str, lut_path: str, preview=False, scale: int = 1, raw_to_jpg=False):
    lut = {}
    if lut_path:
        lut = read_lut(lut_path)

    # https://letmaik.github.io/rawpy/api/rawpy.Params.html
    # demosaic_algorithm=None, half_size=False, four_color_rgb=False, dcb_iterations=0, dcb_enhance=False,
    # fbdd_noise_reduction=FBDDNoiseReductionMode.Off, noise_thr=None, median_filter_passes=0, use_camera_wb=False,
    # use_auto_wb=False, user_wb=None, output_color=ColorSpace.sRGB, output_bps=8, user_flip=None, user_black=None,
    # user_sat=None, no_auto_bright=False, auto_bright_thr=None, adjust_maximum_thr=0.75, bright=1.0,
    # highlight_mode=HighlightMode.Clip, exp_shift=None, exp_preserve_highlights=0.0, no_auto_scale=False,
    # gamma=None, chromatic_aberration=None, bad_pixels_path=None

    with rawpy.imread(raw_path) as raw:
        # layers = d.raw_image_visible
        # write_layers(layers)
        color_size = 65535
        flip, angle = get_orientation(raw_path)
        if preview:
            thumb: Thumbnail = raw.extract_thumb()
            print('thumnail type:', thumb.format)
            img = Image.open(BytesIO(thumb.data))
            img = img.resize((int(raw.sizes.width / scale), int(raw.sizes.height / scale)), Image.LANCZOS)
            # resize then rotate
            if angle is not None:
                img = img.rotate(angle, expand=True)
            rgb = np.array(img)
            color_size = 255
        else:
            rgb: np.ndarray = raw.postprocess(
                # noise reduction
                fbdd_noise_reduction=FBDDNoiseReductionMode.Full,
                noise_thr=0.2,
                median_filter_passes=2,
                # WB
                use_camera_wb=True,  # important
                use_auto_wb=False,
                user_wb=None,
                chromatic_aberration=(1, 1),
                # quality
                output_color=ColorSpace.sRGB,
                output_bps=16,  # important
                # custom
                user_flip=flip,
                user_black=None,
                user_sat=None,
                # brightness
                # gamma=(1, 1),  # important
                # gamma=(2.222, 4.5),  # Rec.709
                gamma=(2.4, 12.92),  # Rec.2020
                # no_auto_bright=True,  # important
                auto_bright_thr=0.0001,
                bright=1,
                highlight_mode=HighlightMode.Blend,
                exp_shift=1.4,
                exp_preserve_highlights=1.0,  # important
                # resolution
                no_auto_scale=True,
            )
        if lut:
            print('LUT:', lut.name)
            start = datetime.now()
            rgb = apply_lut(rgb, lut, color_size=color_size)
            print(f'time cost for apply_lut: {datetime.now() - start}')
        if preview:
            rgb = rgb.astype(np.uint8)
            iio.imwrite(replace_extension(raw_path, '.jpg'), rgb)
        elif raw_to_jpg:
            rgb = np.clip(rgb.astype(np.float32) / color_size * 255, 0, 255).astype(np.uint8)
            iio.imwrite(replace_extension(raw_path, '.jpg'), rgb)
        else:
            iio.imwrite(replace_extension(raw_path, '.dng'), rgb)


# CLI

parser = argparse.ArgumentParser()
parser.add_argument('file', help='path to the .x3f raw file')
parser.add_argument('-l', '--lut', help='path to the .cube file')
parser.add_argument('-j', '--jpg', action='store_true', help='export raw as jpg')
parser.add_argument('-p', '--preview', action='store_true', help='use thumbnail instead of full resolution')
parser.add_argument('-s', '--scale', type=int, help='scale down thumbnail', default=4)
args = parser.parse_args()

parse_x3f(
    args.file,
    args.lut,
    preview=args.preview,
    scale=args.scale,
    raw_to_jpg=args.jpg,
)
