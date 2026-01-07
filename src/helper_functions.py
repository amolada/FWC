#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 11:41:11 2025

@author: Adolfo Molada Tebar
"""
import glob
import os

import geopandas
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import pyarrow
import pyarrow.feather as feather

# Files
# ---------

def extract_files_from_dir(dir_path, extension="png"):
    extension_to_find = "*." + extension
    files_paths = glob.glob(os.path.join(dir_path, extension_to_find)) # get all png files from dir (only .png)
    print("Number of files: ", len(files_paths)) 
    return files_paths

# Read
# ----

def read_shp(shp_path):
    gdf = geopandas.read_file(shp_path) 
    fields = list(gdf.columns)
    print(fields)
    return gdf

def read_sample_rgb(sample_path):
    img = Image.open(sample_path).convert("RGB")
    return np.array(img)

def load_feather_as_df(feather_file_path):
    df = pd.read_feather(feather_file_path)
    return df


# Save data
# ---------

def save_df_as_feather(df, feather_file_path):
    """
    Function to save a pd.DataFrame as .feather
    """
    table = pyarrow.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, feather_file_path)

def save_as_png(img, path):
    alpha = np.full((img.shape[0], img.shape[1]), 255, dtype=np.uint8)
    img_rgba = np.dstack([img, alpha])
    image_from_array = Image.fromarray(img_rgba.astype("uint8"))
    image_from_array.save(path)
