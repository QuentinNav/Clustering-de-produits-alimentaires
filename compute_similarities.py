import pandas as pd
import numpy as np
from tqdm import tqdm
import operator
from operator import itemgetter
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(42)
import math
from sklearn.utils import shuffle



import multiprocessing as mp
from itertools import repeat
from config import data_path

import warnings
warnings.filterwarnings("ignore")

#Matplotlib and Seaborn parameters :
from matplotlib import rcParams
rcParams['figure.figsize'] = 15,8 #Taille de la figure affichée
sns.set_style("darkgrid")#style de l'arrière plan de seaborn
sns.set_palette("pastel")#Couleurs utilisées dans les graphiques

df_ingredients_word2vec = pd.read_csv(data_path+"ingredients_word2vec.csv",sep="\t").set_index("ingredient")

#Similarity matrix between ingredients
def compute_similarity_matrix(df) :
    A_sparse = sparse.csr_matrix(df)
    similarities =cosine_similarity(A_sparse)
    return pd.DataFrame(data = similarities, index=list(df_ingredients_word2vec.index), columns = list(df_ingredients_word2vec.index))


df_similarities = compute_similarity_matrix(df_ingredients_word2vec)



def similarity_max_A_dans_B(A,B):
    return df_similarities.loc[A, B].max(axis=1).mean()


#Max pooling dans un seul sens
def find_similar_products_max_pooling(df_products, product_name):
    #On récupère la liste des ingrédients du produits
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]
    if len(liste_ingredients) == 0 :
        print("Produit non trouvé dans le dataframe")
    else :
        df_temp = df_products[df_products["product_name"]!=product_name]# On retire le produits testé de la liste

        #sum_max_similarities : somme des coefs max entre chaque produit et les ingrédients du produits testé

        sum_max_similarities =np.zeros(len(df_temp))
        for ingredient in tqdm(liste_ingredients) :

            liste_resultats = []
            dict_values = df_similarities[ingredient].to_dict()

            for liste_ingredients_to_compute in df_temp["liste_ingredients"] :
                f = operator.itemgetter(*liste_ingredients_to_compute)
                coefs = f(dict_values)
                liste_resultats.append(max(coefs) if type(coefs)==tuple else coefs)
            sum_max_similarities=np.add(sum_max_similarities, np.array(liste_resultats))



        results_similar_products = pd.DataFrame()
        results_similar_products["product_name"] = df_temp["product_name"]
        results_similar_products["liste_ingredients"] = df_temp["liste_ingredients"]
        results_similar_products["similarity"]= np.divide(sum_max_similarities,len(liste_ingredients))

    return results_similar_products

#Average pooling dans un seul sens
def find_similar_products_avg_pooling(df_products, product_name):
    #On récupère la liste des ingrédients du produits
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]
    if len(liste_ingredients) == 0 :
        print("Produit non trouvé dans le dataframe")
    else :
        df_temp = df_products[df_products["product_name"]!=product_name]# On retire le produits testé de la liste

        #sum_max_similarities : somme des coefs max entre chaque produit et les ingrédients du produits testé

        sum_max_similarities =np.zeros(len(df_temp))
        for ingredient in tqdm(liste_ingredients) :
            #serie_similarities_ingredient_teste = df_similarities[ingredient]
            #np.add(sum_max_similarities, df_temp["liste_ingredients"].apply(lambda liste : max_similarity_of_list(ingredient, liste,serie_similarities_ingredient_teste)))


            liste_resultats = []
            dict_values = df_similarities[ingredient].to_dict()

            for liste_ingredients_to_compute in df_temp["liste_ingredients"] :
                f = operator.itemgetter(*liste_ingredients_to_compute)
                coefs = f(dict_values)
                liste_resultats.append(np.mean(coefs) if type(coefs)==tuple else coefs)
            sum_max_similarities=np.add(sum_max_similarities, np.array(liste_resultats))



        results_similar_products = pd.DataFrame()
        results_similar_products["product_name"] = df_temp["product_name"]
        results_similar_products["liste_ingredients"] = df_temp["liste_ingredients"]
        results_similar_products["similarity"]= np.divide(sum_max_similarities,len(liste_ingredients))

    return results_similar_products


#Permet de calculer la similarité entre 2 produits
def find_similar_products_max_pooling_both_ways(df_products, product_name):
    #On récupère la liste des ingrédients du produits
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]
    if len(liste_ingredients) == 0 : #On vérifie que la liste des produits n'est pas vide
        print("Produit non trouvé dans le dataframe")
    else :
        df_temp = df_products[df_products["product_name"]!=product_name]# On retire le produits testé de la liste

        liste_similarities1 = []#Similarité des ingrédients du produit traité avec les ingrédients des autres produits (en faisant la moyenne d'un max pooling)
        liste_similarities2 = []#Similarité des ingrédients des autres produits avec ceux du produit traité (en faisant la moyenne d'un max pooling)
        for liste_ingredients_to_compute in tqdm(df_temp["liste_ingredients"]) :
            liste_similarities1.append(df_similarities.loc[liste_ingredients, liste_ingredients_to_compute].max(axis=1).mean())
            liste_similarities2.append(df_similarities.loc[liste_ingredients_to_compute, liste_ingredients].max(axis=1).mean())


        results_similar_products = pd.DataFrame()
        results_similar_products["product_name"] = df_temp["product_name"]
        results_similar_products["liste_ingredients"] = df_temp["liste_ingredients"]
        results_similar_products["similarity1"]= liste_similarities1
        results_similar_products["similarity2"]=liste_similarities2
        results_similar_products["mean_similarity"]=(results_similar_products["similarity1"] + results_similar_products["similarity2"])/2

    return results_similar_products.sort_values("mean_similarity",ascending=False)

#Même fonction que précédemment en améliorant le temps d'exécution
def find_similar_products_max_pooling_both_waysV2(df_products, product_name):
    #On récupère la liste des ingrédients du produits
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]
    if len(liste_ingredients) == 0 :
        print("Produit non trouvé dans le dataframe")
    else :
        df_temp = df_products[df_products["product_name"]!=product_name]# On retire le produits testé de la liste


        sum_max_similarities =np.zeros(len(df_temp))
        for ingredient in liste_ingredients :


            liste_similarities1 = []
            dict_values = df_similarities[ingredient].to_dict()

            for liste_ingredients_to_compute in df_temp["liste_ingredients"] :

                f = operator.itemgetter(*liste_ingredients_to_compute)
                coefs = f(dict_values)

                liste_similarities1.append(np.max(coefs) if type(coefs)==tuple else coefs)
            sum_max_similarities=np.add(sum_max_similarities, np.array(liste_similarities1))

        list_list_vect = [ [] for _ in range(len(df_temp["liste_ingredients"])) ]

        for ingredient in liste_ingredients:
            list_matrix_similarities = []
            dict_values= df_similarities[ingredient].to_dict()

            for index, liste_ingredients_to_compute in enumerate(df_temp["liste_ingredients"]) :
                f= operator.itemgetter(*liste_ingredients_to_compute)
                coefs = f(dict_values)

                list_list_vect[index].append(np.array(coefs))
                #list_matrix_similarities.append(np.array(coefs))


        #Getting max for each ingredients :
        list_mean_of_max_vects =[]
        for list_vect in list_list_vect :
            list_mean_of_max_vects.append(np.array([np.array(vect) for vect in list_vect]).max(axis=0,keepdims=True).mean())



        results_similar_products = pd.DataFrame()
        results_similar_products["product_name"] = df_temp["product_name"]
        results_similar_products["liste_ingredients"] = df_temp["liste_ingredients"]
        results_similar_products["similarity1"]= np.divide(sum_max_similarities,len(liste_ingredients))
        results_similar_products["similarity2"]=list_mean_of_max_vects
        results_similar_products["mean_similarity"]=(results_similar_products["similarity1"] + results_similar_products["similarity2"])/2

    return results_similar_products.sort_values("mean_similarity",ascending=False)



def similarities_both_ways(A,B, dicts_A) :
    list_vects=[]
    for dico in dicts_A :
        list_vects.append(np.array(operator.itemgetter(*B)(dico)))
    return np.array(list_vects).max(axis=1).mean(), np.array(list_vects).max(axis=0).mean()# similarité A dans B , B dans A

def similarities_both_ways_mean(A,B, dicts_A) :
    list_vects=[]
    for dico in dicts_A :
        list_vects.append(np.array(operator.itemgetter(*B)(dico)))
    return (np.array(list_vects).max(axis=1).mean()+ np.array(list_vects).max(axis=0).mean())/2# similarité A dans B , B dans A




#Même fonction que précédemment en améliorant le temps d'exécution
def find_similar_products_max_pooling_both_waysV3(df_products, product_name):
    #On récupère la liste des ingrédients du produits
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]
    if len(liste_ingredients) == 0 :
        print("Produit non trouvé dans le dataframe")
    else :
        df_temp = df_products[df_products["product_name"]!=product_name]# On retire le produits testé de la liste

        list_dicts = [df_similarities[ingredient].to_dict() for ingredient in liste_ingredients]
        df_temp[["similarity1","similarity2"]] =df_temp.apply(lambda row : similarities_both_ways(liste_ingredients,row.liste_ingredients,list_dicts),axis='columns', result_type='expand')
        df_temp["mean_similarity"] = (df_temp["similarity1"] + df_temp["similarity2"])/2

    return df_temp[["product_name","liste_ingredients","similarity1","similarity2","mean_similarity"]].sort_values("mean_similarity",ascending=False)


def find_similar_products_max_pooling_both_waysV3_multiprocessing(df_products, product_name):
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]

    df_temp = df_products[df_products["product_name"]!=product_name]# On retire le produits testé de la liste
    list_dicts = [df_similarities[ingredient].to_dict() for ingredient in liste_ingredients]

    with mp.Pool(mp.cpu_count()) as pool :
        df_temp[["similarity1","similarity2"]] = pool.starmap(similarities_both_ways, zip(repeat(liste_ingredients),df_temp["liste_ingredients"], repeat(list_dicts)))


    df_temp["mean_similarity"] = (df_temp["similarity1"] + df_temp["similarity2"])/2


    return df_temp[["product_name","liste_ingredients","similarity1","similarity2","mean_similarity"]].sort_values("mean_similarity",ascending=False)



#Mapping des produits :
def find_similar_products_max_pooling_both_waysV3_multiprocessing_maping(df_products, product_name):
    liste_ingredients = df_products.loc[df_products["product_name"]==product_name, "liste_ingredients"].iloc[0]

    df_temp = df_products.copy(deep=True)
    list_dicts = [df_similarities[ingredient].to_dict() for ingredient in liste_ingredients]

    with mp.Pool(mp.cpu_count()) as pool :
        df_temp[["similarity1","similarity2"]] = pool.starmap(similarities_both_ways, zip(repeat(liste_ingredients),df_temp["liste_ingredients"], repeat(list_dicts)))


    df_temp["mean_similarity"] = (df_temp["similarity1"] + df_temp["similarity2"])/2


    return df_temp.set_index("product_name")["mean_similarity"].to_dict()


#check if matrix is positive semidefinite :
def is_hermitian_positive_semidefinite(X):
    if X.shape[0] != X.shape[1]: # must be a square matrix
        return False

    if not np.all( X - X.T == 0 ): # must be a symmetric or hermitian matrix
        return False

    try: # Cholesky decomposition fails for matrices that are NOT positive definite.

        # But since the matrix may be positive SEMI-definite due to rank deficiency
        # we must regularize.
        regularized_X = X + np.eye(X.shape[0]) * 1e-14

        np.linalg.cholesky(regularized_X)
    except np.linalg.LinAlgError:
        return False

    return True


def x_coord_of_point(D, j):
    return ( D[0,j]**2 + D[0,1]**2 - D[1,j]**2 ) / ( 2*D[0,1] )

def coords_of_point(D, j):
    x = x_coord_of_point(D, j)
    #print("D[0,j]**2 : "+str( D[0,j]**2)+ " x**2 : "+str(x**2))
    return np.array([x, math.sqrt( (D[0,j]**2 - x**2) if (D[0,j]**2 - x**2)>0 else 0  )])

def calculate_positions(D):
    (m, n) = D.shape
    P = np.zeros( (n, 2) )
    tr = ( min(min(D[2,0:2]), min(D[2,3:n])) / 2)**2
    P[1,0] = D[0,1]
    P[2,:] = coords_of_point(D, 2)
    for j in range(3,n):
        P[j,:] = coords_of_point(D, j)
        if abs( np.dot(P[j,:] - P[2,:], P[j,:] - P[2,:]) - D[2,j]**2 ) > tr:
            P[j,1] = - P[j,1]
    return P





def selection_vecteurs_representation(df, seuil) :
    df=df.set_index("product_name")

    #Initialisation des variables :
    product_init = df.sample().index[0]
    list_products_names =[product_init]
    list_ingredients_list= [df.loc[df.index==product_init,"liste_ingredients"].iloc[0]]#list of list of ingredients_names

    list_dicts_list =[[df_similarities[ingredient].to_dict() for ingredient in list_ingredients_list[0]]]

    df_temp= shuffle(df)

    for i in tqdm(range(len(df_temp))) : # Pour chaque produit dans le dataframe
        #product_name = df_temp.index[i]
        product_ingredients= df_temp["liste_ingredients"].iloc[i]

        add=True

        for j in range(len(list_ingredients_list)) : #Pour chaque produit dans le vecteur
            sim = similarities_both_ways_mean(list_ingredients_list[j], product_ingredients,list_dicts_list[j])

            if sim >seuil :
                add = False
                break

        if add :
            list_products_names.append(df_temp.index[i])
            list_ingredients_list.append(product_ingredients)
            list_dicts_list.append([df_similarities[ingredient].to_dict() for ingredient in product_ingredients])


    return list_products_names, list_ingredients_list


def calcul_similarites_produits_vecteurs_produits_mp(df_products, list_products_vector,list_products_name): 

    df_results =df_products.set_index("product_name")[["liste_ingredients"]]

    for index, liste_ingredients in enumerate(tqdm(list_products_vector)) : #Pour chaque produit dans le vecteur de produit
        list_dicts = [df_similarities[ingredient].to_dict() for ingredient in liste_ingredients]#On récupère les dictionnaires de similarités associés à ses ingrédients
        with mp.Pool(mp.cpu_count()) as pool :
            df_results[list_products_name[index]] = pool.starmap(similarities_both_ways_mean, zip(repeat(liste_ingredients),df_products["liste_ingredients"], repeat(list_dicts)))#On calcul sa similarité avec tous les produits du dataframe

    return df_results
