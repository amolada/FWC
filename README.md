# FWC
TFM - Inteligencia Artificial para la Observación de la Tierra: Comparación de modelos para la Monitorización de la vegetación


Create a conda env


```
conda create -n fwc_env python=3.10

conda activate fwc_env

pip install torch torchvision numpy pandas pillow

pip install matplotlib seaborn geopandas pyarrow scikit-image

pip install scikit-learn

pip install xgboost

pip install shape lime

pip install ipykernel


```

Notebooks

01_Datos.ipynb

02_Preprocesamiento_de_datos.ipynb

03_RandomForest.ipynb

04_SVM.ipynb

05_XGBoost.ipynb

06_MLP.ipynb

07_XAI.ipynb

08_Generar_mascaras.ipynb

09_U-net_class.ipynb


Scripts

display.py

features.py

helper_functions.py

model_functions.py






