#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 11:41:11 2025

@author: Adolfo Molada Tebar
"""

import os

import math
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler

import helper_functions

# Show image

def show_image_from_path(image_path):
    sample_png = helper_functions.read_sample_rgb(image_path)
    plt.imshow(sample_png)
    plt.show()

def show_image_from_array(image):
    plt.imshow(image)
    plt.show()

def show_mask(mask_path, nd=[0,128,255], labels=["Woody","Non woody","Bare soil/non vegetation"], colors = [[0, 150, 0],[144, 238, 144],[139, 69, 19]]):
    mask_basename = os.path.splitext(os.path.basename(mask_path))[0]
    # mask_rgb: array (90,90,3)
    mask_grey = helper_functions.read_sample_rgb(mask_path)
    mask = mask_grey[:, :, 0]
    # Crear imagen coloreada
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
    colored[mask == nd[0]] = colors[0] 
    colored[mask == nd[1]] = colors[1]
    colored[mask == nd[2]] = colors[2]

    # Figura con panel de leyenda más estrecho
    fig, (ax_img, ax_leg) = plt.subplots(
        1, 2,
        figsize=(9, 5),
        gridspec_kw={'width_ratios': [5, 0.8]}   # << más estrecho el panel derecho
        )
    # Imagen
    ax_img.imshow(colored)
    ax_img.set_title(f"Máscara {mask_basename}")
    ax_img.axis("off")

    # Leyenda
    legend_elements = [
        Patch(facecolor=np.array(colors[0])/255,    edgecolor='black', label=labels[0]),
        Patch(facecolor=np.array(colors[1])/255, edgecolor='black', label=labels[1]),
        Patch(facecolor=np.array(colors[2])/255, edgecolor='black', label=labels[2])]

    ax_leg.legend(handles=legend_elements,title="Clases",loc='upper right')
    ax_leg.axis("off")

    # --- Ajuste fino del espacio ---
    plt.subplots_adjust(wspace=0.1,   # << reduce el espacio horizontal entre paneles
                        right=0.95)   # << ajusta margen derecho para acercar leyenda
    plt.show()

# Plots

def plot_boxplot_all_features(df):
    plt.figure(figsize=(20,10))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("Boxplot de cada variable")
    plt.show()
    
# Outliers

def plot_iqr_outliers_analysis(outliers_analysis, figsize=(14,6), titulo='Número y porcentaje de outliers por variable (IQR)'):
    """
    Genera un gráfico combinado (barras + línea) a partir del DataFrame `outliers_analysis`.
    Los ticks de los ejes Y están en negro.
    """
    # Ordenamos por porcentaje descendente
    df_sorted = outliers_analysis.sort_values('porcentaje_outliers', ascending=False)

    variables = df_sorted.index
    n_outliers = df_sorted['n_outliers']
    porcentaje_outliers = df_sorted['porcentaje_outliers']

    # Crear figura y ejes
    fig, ax1 = plt.subplots(figsize=figsize)

    # Barras: número de outliers
    color_barras = 'skyblue'
    ax1.bar(variables, n_outliers, color=color_barras, label='Número de outliers')
    ax1.set_xlabel('Variable')
    ax1.set_ylabel('Número de outliers')
    ax1.tick_params(axis='y', labelcolor='black')  # Ticks en negro
    ax1.tick_params(axis='x', rotation=90)

    # Segundo eje: porcentaje
    ax2 = ax1.twinx()
    color_linea = 'salmon'
    ax2.plot(variables, porcentaje_outliers, color=color_linea, marker='o', label='Porcentaje de outliers')
    ax2.set_ylabel('Porcentaje de outliers (%)')
    ax2.tick_params(axis='y', labelcolor='black')  # Ticks en negro

    # Título y layout
    plt.title(titulo)
    fig.tight_layout()
    plt.show()


def plot_outliers_locales_using_pca(df_outliers, column_label=['sample_id', "label",'outlier_global', 'outlier_local']):
    # Seleccionar columnas numéricas
    cols_numericas = df_outliers.drop(columns=column_label).columns
    X = df_outliers[cols_numericas].copy()
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA a 2 componentes
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualización
    plt.figure(figsize=(10,7))
    # Todos los puntos en azul
    plt.scatter(X_pca[:,0], X_pca[:,1], c='blue', alpha=0.5, s=50, label='Puntos')
    # Outliers locales en rojo
    outlier_mask = df_outliers['outlier_local'] == -1
    plt.scatter(X_pca[outlier_mask,0], X_pca[outlier_mask,1],
            c='red', s=70, edgecolor='k', label='Outliers locales')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title("Representación de outliers locales")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_outliers_locales_por_clase_using_pca(df_outliers, column_label=['sample_id', "label",'outlier_global', 'outlier_local'], class_field_name = "label"):

    # Selección de columnas numéricas
    cols_numericas = df_outliers.drop(columns=column_label).columns
    X = df_outliers[cols_numericas].copy()
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # PCA a 2 componentes
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_plot = df_outliers.copy()
    df_plot['PC1'] = X_pca[:, 0] 
    df_plot['PC2'] = X_pca[:, 1]

    # Colores por clase
    class_colors = {
        0: '#1B5E20',  # Verde oscuro → Woody vegetation
        1: '#F9A825',  # Amarillo oscuro (mostaza) → Non-woody vegetation
        2: '#4E342E'   # Marrón oscuro → Non vegetation / bare soil
    }
    # Etiquetas descriptivas para la leyenda
    class_labels = {
        0: 'Woody',
        1: 'Non-Woody',
        2: 'Non-Wegetation/Bare soil'
    }
    
    # Visualización
    plt.figure(figsize=(11, 8))

    # Dibujar puntos por clase
    for clase, color in class_colors.items():
        mask = df_plot[class_field_name] == clase
        plt.scatter(
            df_plot.loc[mask, 'PC1'],
            df_plot.loc[mask, 'PC2'],
            c=color,
            alpha=0.6,
            s=45,
            label=class_labels[clase]
        )

    # Resaltar outliers locales
    outlier_mask = df_plot['outlier_local'] == -1
    plt.scatter(
        df_plot.loc[outlier_mask, 'PC1'],
        df_plot.loc[outlier_mask, 'PC2'],
        facecolors='none',
        edgecolors='red',
        s=90,
        linewidths=1.5,
        label='Outliers locales'
    )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Representación de outliers locales por clase (PCA 2D)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Metrics

def plot_confusion_matrix(conf_matrix, size=(6,4)):
    plt.figure(figsize=size)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(df_importances, model_name):
    df_importances = df_importances.sort_values(by="importance", ascending=True)
    # Gráfico de barras horizontales
    plt.figure(figsize=(14, len(df_importances)*0.3))  # Ajusta la altura según número de features
    bars = plt.barh(df_importances["feature"], df_importances["importance_pct"], color="skyblue")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance {model_name}")
    # Añadir los valores en porcentaje sobre cada barra
    for bar, imp_pct in zip(bars, df_importances["importance_pct"]):
        plt.text(imp_pct + 0.0005, bar.get_y() + bar.get_height()/2, f"{imp_pct:.2f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(model,X_test, y_test):
    y_proba_test = model.predict_proba(X_test) # "predict_proba" para extraer probabilidades vez de predicciones

    prob_1 = y_proba_test[:,0] 

    for i in range(y_proba_test.shape[1]):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba_test[:, i])
        auc_i = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Clase {i} (AUC={auc_i:.3f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC por clase')
    plt.legend()
    plt.grid()
    plt.show()

def plot_learning_curves(history, model_name):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Curva de aprendizaje {model_name}")
    plt.legend()
    plt.show()