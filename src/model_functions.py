import glob
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pickle
from PIL import Image

from scipy.stats import wilcoxon
import sklearn
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import classification_report, confusion_matrix

import display_functions
import features
import helper_functions

# Read/Save

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def load_model_data_standardizer(standalizer_path):
    with open(standalizer_path, 'rb') as file:
        model_data_standardizer = pickle.load(file)
    return model_data_standardizer

def save_model_data_standardizer(model_data_standardizer, path):
    with open(path, 'wb') as file:
        pickle.dump(model_data_standardizer, file)

# Evaluation

def compute_metrics(y_test,y_pred):
    #mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test,y_pred)
    #med = median_absolute_error(y_test,y_pred)
    #ev = explained_variance_score(y_test,y_pred)
    #accuracy = accuracy_score(y_test, y_pred)
    #epsilon = 1e-10 
    #mape = np.mean(np.abs((y_testing - y_pred_test) / (y_testing + epsilon))) * 100 # evitar división por 0
    return mse,rmse,r2 #ev,mae,med,accuracy

def compute_classification_report(y_test,y_pred):
    report = classification_report(y_test, y_pred)
    return report

def compute_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix

# Importance

def compute_feature_importance(model, feature_names, model_name, plot=True):
    importances = model.feature_importances_
    df_importances = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    df_importances["importance_pct"] = df_importances["importance"] * 100
    
    df_importances = df_importances.sort_values(by="importance", ascending=False)
    if plot:
        display_functions.plot_feature_importance(df_importances, model_name)
    return df_importances

# Wilcoxon

def apply_wilcoxon_test(scores_full, scores_reduced, alpha=0.05):

    scores_full = np.array(scores_full)
    scores_reduced = np.array(scores_reduced)

    stat, p_value = wilcoxon(
        scores_full,
        scores_reduced,
        alternative="two-sided"
    )

    mean_full = scores_full.mean()
    mean_reduced = scores_reduced.mean()

    print("===== TEST DE WILCOXON (muestras emparejadas) =====")
    print(f"Media modelo FULL     : {mean_full:.4f}")
    print(f"Media modelo REDUCIDO : {mean_reduced:.4f}")
    print(f"Estadístico W         : {stat:.4f}")
    print(f"p-value               : {p_value:.4e}")
    print(f"Nivel α               : {alpha}")

    if p_value < alpha:
        print(">>> RESULTADO: Se RECHAZA la hipótesis nula (H₀)")
        if mean_reduced > mean_full:
            print(">>> CONCLUSIÓN: El modelo REDUCIDO mejora SIGNIFICATIVAMENTE al FULL")
        else:
            print(">>> CONCLUSIÓN: El modelo FULL mejora SIGNIFICATIVAMENTE al REDUCIDO")
    else:
        print(">>> RESULTADO: NO se rechaza la hipótesis nula (H₀)")
        print(">>> CONCLUSIÓN: No hay diferencias estadísticamente significativas")

    return stat, p_value

# Predict for mask generation

def reshape_mask_predicted(mask_pred, nd_class=[0,128,255],size=(90,90), num_clases=3):
    #try:
    #    mask_reshape = mask_pred.reshape(size)
    #except:
    #    size_mod = (size[0], size[1], num_clases)
    #    mask_reshape = mask_pred.reshape(size_mod)
        #mask_reshape = mask_reshape[:,:,0]
    #    print(np.unique(mask_reshape))
    
    mask_reshape = mask_pred.reshape(size)
    mask_reshape[mask_reshape == 0] = nd_class[0]
    mask_reshape[mask_reshape == 1] = nd_class[1]
    mask_reshape[mask_reshape == 2] = nd_class[2]
    return mask_reshape

def mask_to_gray(mask):
    mask_grey = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_grey[:, :, 0] = mask
    mask_grey[:, :, 1] = mask
    mask_grey[:, :, 2] = mask
    return mask_grey

def predict_masks(image_paths, model, output_dir, standarized = None, model_name = None, mode="predict"):
    for path in image_paths:
        img = helper_functions.read_sample_rgb(path)
        #print(img.shape)
        sample_data = features.extract_features(img, label=None, mode=mode)
        #print(sample_data.shape)
        if standarized is not None:
            sample_data = standarized.transform(sample_data)
        mask_pred = model.predict(sample_data)
        if model_name=="MLP":
            mask_pred = np.argmax(mask_pred, axis=1)
        mask_reshape = reshape_mask_predicted(mask_pred)
        mask_grey = mask_to_gray(mask_reshape)
        # save as png
        mask_basename = os.path.basename(path)
        mask_out_path = os.path.join(output_dir, mask_basename)
        #print(mask_out_path)
        helper_functions.save_as_png(mask_grey, mask_out_path)

def predict_masks_fs(image_paths, model, model_features, dropped_features, output_dir, standarized = None, model_name = None, mode="predict"):
    for path in image_paths:
        img = helper_functions.read_sample_rgb(path)
        #print(img.shape)
        sample_data = features.extract_features(img, label=None, mode=mode)
        df = pd.DataFrame(sample_data, columns=model_features)
        sample_data_fs = df.drop(columns=dropped_features)
        
        #print(sample_data.shape)
        if standarized is not None:
            sample_data_fs = standarized.transform(sample_data_fs)
        mask_pred = model.predict(sample_data_fs)
        if model_name=="MLP":
            mask_pred = np.argmax(mask_pred, axis=1)
        mask_reshape = reshape_mask_predicted(mask_pred)
        mask_grey = mask_to_gray(mask_reshape)
        # save as png
        mask_basename = os.path.basename(path)
        mask_out_path = os.path.join(output_dir, mask_basename)
        #print(mask_out_path)
        helper_functions.save_as_png(mask_grey, mask_out_path)