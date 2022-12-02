#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(42)

#Matplotlib and Seaborn parameters : 
from matplotlib import rcParams
rcParams['figure.figsize'] = 15,8 #Taille de la figure affichée
sns.set_style("darkgrid")#style de l'arrière plan de seaborn 
sns.set_palette("pastel")#Couleurs utilisées dans les graphiques


#Preprocessing
def check_nan(df):
    for i in df.columns.tolist():
        print("Valeurs nan dans "+str(i)+" : "+str(df[i].isna().sum()))
        
def check_unique(df):
    for i in df.columns.tolist():
        print("Valeurs uniques dans "+str(i)+" : "+str(df[i].nunique()))

def format_tags (x):
    list_dicts_temp =[{a[0]:a[1]} for a in filter(lambda z : len(z)>1 , [ y.split(":") for y in x.split(',')])]
    return {k:[d.get(k) for d in list_dicts_temp] for k in set().union(*list_dicts_temp)}

def filter_ingredients(liste_ingredients, ingredients_to_keep):
    return list(filter(lambda product : product in ingredients_to_keep, liste_ingredients)) 

