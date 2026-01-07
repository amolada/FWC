#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 11:41:11 2025

@author: Adolfo Molada Tebar
"""
import glob
import os

import geopandas
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import shapiro
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2gray, rgb2hsv, rgb2lab, rgb2ycbcr

# Clean NaN
# ---------

def find_nan(data):
    """ 
    Funtion to detect NaN values on a DataFrame
    """
    nan_detected = data.isna().any().any() # bool
    if nan_detected:
        filas_con_nan = data[data.isna().any(axis=1)]
        print("\nFilas con NaN:")
        print(filas_con_nan)
        # Posiciones exactas (fila, columna)
        nan_positions = [(idx, col)
                         for idx, row in data.iterrows() 
                         for col, val in row.items() 
                         if pd.isna(val)]

        print("\nPosiciones exactas de NaNs:")
        print(nan_positions)        
        data = data.dropna() # delete
    else:
        print("No hay NaN. No se elimina nada")
    return data

# Outliers

# Z-score

def test_distribucion_normal(data):
    """ 
    Function to apply the Shapirp-Wilk test
    """
    columna = data.columns
    for variable in columna:
        stat, p = shapiro(data[variable])
        if p < 0.05:
            print(f"La variable {variable} no sigue una distribución normal → Z-score no fiable")

# IQR

def analizar_outliers_iqr(df, factor=1.5):
    """
    Function for outlier detection using the IQR criterion
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    lim_inf = Q1 - factor * IQR
    lim_sup = Q3 + factor * IQR

    limites = pd.DataFrame({
        "lim_inf": lim_inf,
        "lim_sup": lim_sup
    })

    outliers = (df < lim_inf) | (df > lim_sup)
    n_outliers_por_variable = outliers.sum(axis=0)
    n_total = len(df)
    porcentaje_outliers = 100 * n_outliers_por_variable / n_total

    outliers_analysis = pd.DataFrame({
        'n_outliers': n_outliers_por_variable,
        'porcentaje_outliers': porcentaje_outliers
        }).sort_values('porcentaje_outliers', ascending=False)

    return outliers_analysis


# Métodos multivariantes

def detectar_outliers_multivariante(df, cols_drop, n_estimators=200, n_neighbors=20, contamination="auto"):
    """
    Function for outlier detection using Multivariante criteria
    IsolationForest -> global
    LocalOutlierFactor -> local
    """
    df_outliers = df.copy()
    X = df_outliers.drop(columns=cols_drop).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # IsolationForest -> Detección global
    iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso_forest.fit(X_scaled)
    # Etiqueta de outlier: -1 = outlier, 1 = normal
    df_outliers['outlier_global'] = iso_forest.predict(X_scaled)
    # Extraer IDs de puntos outliers globales
    ids_outliers_global = df_outliers.loc[df_outliers['outlier_global'] == -1, 'sample_id']
    
    # LocalOutlierFactor -> Detección local
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    df_outliers['outlier_local'] = lof.fit_predict(X_scaled)
    # Extraer IDs de puntos outliers locales
    ids_outliers_local = df_outliers.loc[df_outliers['outlier_local'] == -1, 'sample_id']

    return df_outliers, ids_outliers_global.tolist(), ids_outliers_local.tolist()

# Image

def central_pixel(img):
    """ 
    Function to get the RGB value of the central pixel of an image
    """
    w, h, _ = img.shape
    # Coordenadas del píxel central
    cx = w // 2
    cy = h // 2
    # Obtener valor RGB
    a, b, c = img[cy, cx]
    return (a, b, c)

def get_R_G_B_data(rgb_array):
    """ 
    Function to split the RGB channels of an image
    """
    R = rgb_array[:, :, 0].astype(float) # required for computing the espectral indexes
    G = rgb_array[:, :, 1].astype(float)
    B = rgb_array[:, :, 2].astype(float)    
    return R,G,B

def create_array_from_rgb_values(r, g, b):
    """
    Function to create an array form r,g,b values
    """
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

# Colour space conversion

def rgb_to_gray(rgb_array):
    """
    Function to transform between RGB/Gray
    """
    gray_array = rgb2gray(rgb_array)
    return gray_array.astype(np.uint8)

def rgb_to_hsv(rgb_array):
    """
    Function to transform between RGB/HSV
    """    
    #rgb_norm = rgb_array / 255.0 # norm
    hsv = rgb2hsv(rgb_array)
    return hsv

def rgb_to_lab(rgb_array):
    """
    Function to transform between RGB/CIELAB
    """        
    #rgb_norm = rgb_array / 255.0 # norm
    lab = rgb2lab(rgb_array)
    return lab

def rgb_to_ycbcr(rgb_array):
    """
    Function to transform between RGB/YCbCr
    """        
    #rgb_norm = rgb_array / 255.0 # norm
    ycbcr = rgb2ycbcr(rgb_array)
    return ycbcr


# Spectral indexes

stability_constant = 1e-6 # avoid NaN

def cive(R,G,B):
    """
    Function to compute CIVE (Color Index of Vegetation Extraction) 
    
    Ref:    
        Kataoka, Takashi, et al. "Crop growth estimation system using machine vision." 
        Proceedings 2003 IEEE/ASME international conference on advanced intelligent mechatronics 
        (AIM 2003). Vol. 2. IEEE, 2003. 
        https://doi.org/10.1109/AIM.2003.1225492 
    """
    return 0.441 * R - 0.811 * G + 0.385 * B + 18.787

def exg(R, G, B): 
    """
    Function to compute ExG (Excess Green)
    
    Ref:
        Meyer, G. E., & Neto, J. C. (2008). Verification of color vegetation indices 
        for automated crop imaging applications. Computers and electronics in agriculture, 
        63(2), 282-293.
        https://doi.org/10.1016/j.compag.2008.03.009                
    """
    return 2 * G - R - B

def exr(R, G, B): 
    """
    Function to compute ExR (Excess Red)
    
    Ref:
        Meyer, G. E., & Neto, J. C. (2008). Verification of color vegetation indices 
        for automated crop imaging applications. Computers and electronics in agriculture, 
        63(2), 282-293.
        https://doi.org/10.1016/j.compag.2008.03.009
    
    """
    return 1.4 * R - G

def exb(R, G, B):
    """
    Function to compute ExB (Excess Blue)
    
    Ref:
        Meyer, G. E., & Neto, J. C. (2008). Verification of color vegetation indices 
        for automated crop imaging applications. Computers and electronics in agriculture, 
        63(2), 282-293.
        https://doi.org/10.1016/j.compag.2008.03.009    
    """ 
    exb = 2 * B - R - G
    return exb

def gli(R, G, B):  
    """
    Function to compute GLI (Green Leaf Index)

    Ref: 
    
        Viña, Andrés, et al. "Comparison of different vegetation indices for the remote assessment
        of green leaf area index of crops." Remote sensing of environment 115.12 (2011): 3468-3478.
        https://doi.org/10.1016/j.rse.2011.08.010
        
        Anatoly A. Gitelson, Yoram J. Kaufman, Robert Stark, Don Rundquist,
        Novel algorithms for remote estimation of vegetation fraction,
        Remote Sensing of Environment, Volume 80, Issue 1, 2002,Pages 76-87,
        https://doi.org/10.1016/S0034-4257(01)00289-9.
    """
    return (2*G - R - B) / (2*G + R + B + stability_constant)


def ngrdi(R,G,B):
    """
    Function to compute NGRDI (Normalized Green-Red Difference Index)

    Ref: 
        Rodríguez-Pérez, José R., et al. "Evaluation of hyperspectral reflectance indexes to detect 
        grapevine water status in vineyards." American Journal of Enology and Viticulture 58.3 (2007): 302-317. 
        https://doi.org/10.5344/ajev.2007.58.3.302    
    """
    return (G - R) / (G + R + stability_constant)

def rgbvi(R,G,B):  
    """
    Function to compute RGBVI (Red Green Blue Vegetation Index)

    Ref: 
        Bendig, Juliane, et al. "Combining UAV-based plant height from crop surface models, visible, 
        and near infrared vegetation indices for biomass monitoring in barley." International Journal of 
        Applied Earth Observation and Geoinformation 39 (2015): 79-87. 
        https://doi.org/10.1016/j.jag.2015.02.012  
    """
    return (G*G - R*B) / (G*G + R*B + stability_constant)
    
def tgi(R,G,B):
    """
    Function to compute TGI (Triangular Greenness Index)

    Ref: 
        Hunt Jr, E. Raymond, et al. "Remote sensing leaf chlorophyll content using a visible band index." 
        Agronomy journal 103.4 (2011): 1090-1099. 
        https://doi.org/10.2134/agronj2010.0395 
    """    
    return -0.5 * (190 * (R - G) - 120 * (R - B))

def vari(R, G, B): 
    """
    Function to compute VARI (Visible Armospherically Resistant Index) = (G-R)/(G+R-B)
    
    Ref: 
        Gitelson, A. A., Kaufman, Y. J., Stark, R., & Rundquist, D. (2002). 
        Novel algorithms for remote estimation of vegetation fraction. 
        Remote sensing of Environment, 80(1), 76-87.
        https://doi.org/10.1016/S0034-4257(01)00289-9
    """
    return (G - R) / (G + R - B + stability_constant)


def vvi(R, G, B):  
    """
    Function to compute VVI (Visible Vegetation Index)
     
    Ref: 
        Louhaichi, M., Borman, M. M., & Johnson, D. E. (2001). 
        Spatially located platform and aerial photography for documentation of grazing 
        impacts on wheat. Geocarto International, 16(1), 65-70.
        https://doi.org/10.1080/10106040108542184
    """
    return G/(R+G+B + stability_constant)


# Perceptual Luminance

def compute_perceptual_luminance(R,G,B):
    """
    Function to compute the L perceptual luminance L = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    Ref: 
        Foley, James D. Computer graphics: principles and practice. Vol. 12110. 
        Addison-Wesley Professional, 1996.
    """
    L_perc = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return L_perc

# ratios

def compute_ratios(R, G, B):
    """
    Function to compute RGB band ratios
    """
    gr = G / (R + stability_constant)
    gb = G / (B + stability_constant)
    rb = R / (B + stability_constant)
    return gr, gb, rb


# Extrac features
# ---------------

def extract_features(RGB, sample_id=None, label=None, mode="sample"): # label=None para predict
    """
    Function to extract features from RGB values / RGB images
    """
    mode_implemented = ["sample", "predict"]
    # sample for training
    # predict to obtain the mask

    if mode not in mode_implemented:
        raise Exception
    
    RGB = RGB.astype(float) / 255.
    HSV = rgb_to_hsv(RGB)
    LAB = rgb_to_lab(RGB)
    YCBCR = rgb_to_ycbcr(RGB)
    
    if mode == "sample":
        # colour spaces
        R,G,B = central_pixel(RGB) # norm
        H,S,V = central_pixel(HSV)
        L,a,b = central_pixel(LAB)
        Y,CB,CR = central_pixel(YCBCR)
    else:
        R = RGB[:, :, 0]
        G = RGB[:, :, 1]
        B = RGB[:, :, 2]
        H = HSV[:, :, 0]
        S = HSV[:, :, 1]
        V = HSV[:, :, 2]
        L = LAB[:, :, 0]
        a = LAB[:, :, 1]
        b = LAB[:, :, 2]        
        Y = YCBCR[:, :, 0]
        CB = YCBCR[:, :, 1]
        CR = YCBCR[:, :, 2]        
    
    # spectral indexes
    exg_value = exg(R,G,B)
    exr_value = exr(R,G,B)
    exb_value = exb(R, G, B)    
    vvi_value = vvi(R,G,B)
    vari_value = vari(R,G,B)
    gli_value = gli(R,G,B)
    cive_value = cive(R,G,B)
    rgbvi_value = rgbvi(R,G,B)
    ngrdi_value = ngrdi(R,G,B)
    tgi_value = tgi(R,G,B)
        
    # perceptual luminance
    L_perc = compute_perceptual_luminance(R,G,B)        
        
    # ratios
    gr, gb, rb = compute_ratios(R, G, B)
    
   # Features
   # label,R,G,B,H,S,V,L,a,b,Y,CB,CR,exg_value,exr_value,exb_value,
   # vvi_value,vari_value,gli_value,cive_value,rgbvi_value,ngrdi_value,tgi_value,
   # L_perc,gr, gb, rb

    if mode == "sample":       
        if label is None:
            raise Exception
        if sample_id is None:
            raise Exception        
        else:
            features =[sample_id, int(label), 
                   R,G,B,H,S,V,L,a,b,Y,CB,CR,   
                   exg_value,exr_value,exb_value,vvi_value,vari_value,gli_value,cive_value,rgbvi_value,ngrdi_value,tgi_value,
                   L_perc,
                   gr, gb, rb]
    else:
        row, col, _ = RGB.shape
        dim = row*col    
        # Shape: (90,90) → reshape to (8100,)
        features = np.column_stack([
            R.ravel(),
            G.ravel(),
            B.ravel(),
            H.ravel(),
            S.ravel(),
            V.ravel(),
            L.ravel(),
            a.ravel(),
            b.ravel(),
            Y.ravel(),
            CB.ravel(),
            CR.ravel(),
            exg_value.ravel(),
            exr_value.ravel(),
            exb_value.ravel(),
            vvi_value.ravel(),
            vari_value.ravel(),
            gli_value.ravel(),
            cive_value.ravel(),
            rgbvi_value.ravel(),
            ngrdi_value.ravel(),
            tgi_value.ravel(),
            L_perc.ravel(),
            gr.ravel(), 
            gb.ravel(), 
            rb.ravel()])
    
    return features


# PCA

def compute_pca(df, label = "RGB", columns=['R', 'G', 'B'], n_components=2):
    # puede ser para aplicar PCA a cualquier conjunto de columnas
    # Extraer las columnas para PCA
    X = df[columns].values
    
    # Escalar los datos (recomendado para PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    # Crear nuevo DataFrame con los componentes
    pca_columns = [f'PCA_{label}_{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(components, columns=pca_columns, index=df.index)
    
    return df_pca





