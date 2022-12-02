import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(42)

from wordcloud import WordCloud
from collections import Counter
from matplotlib import gridspec
from sklearn.utils import shuffle 

#Matplotlib and Seaborn parameters : 
from matplotlib import rcParams
rcParams['figure.figsize'] = 15,8 #Taille de la figure affichée
sns.set_style("darkgrid")#style de l'arrière plan de seaborn 

def get_results(model) : 
    labels, tokens = [],[]
    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    return tokens, labels 

def plot_results(values,labels, title) : 
    x,y = [],[]
    for value in values : 
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(21, 20)) 
    plt.gcf().set_dpi(300)

    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom', fontsize=3)
    plt.title(title, fontsize=20)
    plt.show()

def plot_tsne(model): 
    tokens,labels = get_results(model)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    plot_results(new_values, labels,'Carte en 2 dimensions des vecteurs des ingrédients \nMéthode : TSNE' )

def plot_UMAP(model): 
    tokens, labels= get_results(model)

    reducer = umap.UMAP()
    data = np.array(tokens)
    scaled_data= StandardScaler().fit_transform(data)
    new_values = reducer.fit_transform(scaled_data)

    plot_results(new_values, labels, "Carte en 2 dimensions des vecteurs des ingrédients \nMéthode : UMAP")

def plot_PCA(model) :
    tokens,labels = get_results(model)

    data = np.array(tokens)
    scaled_data= StandardScaler().fit_transform(data)
    pca = PCA(n_components=2)
    new_values = pca.fit_transform(data)

    plot_results(new_values, labels, "Carte en 2 dimensions des vecteurs des ingrédients \nMéthode : ACP")



def kmeans_best_nb_clusters(values, limit=75, plot=False) : 

    scores_list =[]
    clusters = [k for k in range(2, limit+1)]
    for k in clusters : 
        model = KMeans(n_clusters=k)
        model.fit(values)
        pred = model.predict(values)
        score = silhouette_score(values, pred)
        scores_list.append(score)
    
    results = pd.DataFrame()
    results["nb_clusters"] = clusters
    results["score"] = scores_list
 
    if plot==True : 
        sns.lineplot(data= results, x ="nb_clusters", y="score")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Score de silhouette obtenu")
        plt.title("Score de silhoutte obtenu en fonction du nombre de clusters\n", fontsize=20)
        plt.show()
    
    return results

def clustering(model, nb_clusters= -1, plot=True): 
    tokens, labels= get_results(model)

    reducer = umap.UMAP(random_state=42)
    data = np.array(tokens)
    scaled_data= StandardScaler().fit_transform(data)
    new_values = reducer.fit_transform(scaled_data)

    best_nb= 2

    if nb_clusters==-1 :
        results_nb_clusters= kmeans_best_nb_clusters(new_values, limit=50, plot=plot)

        best_nb = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "nb_clusters"].iloc[0]
        best_score = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "score"].iloc[0]

        print(f"D'après le score de silhouette le meilleur nombre de clusters pour le clustering est de {best_nb} pour un score de {np.round(best_score,3)}")
    else : 
        best_nb=nb_clusters

    # fit du Kmeans : 
    kmeans = KMeans(n_clusters= best_nb)
    clusters=kmeans.fit_predict(new_values) 

    df_results_clusters = pd.DataFrame()
    df_results_clusters["ingredient"] = labels
    df_results_clusters["2Dvecteur_dim1"] = [value[0] for value in new_values]
    df_results_clusters["2Dvecteur_dim2"]= [value[1] for value in new_values]

    df_results_clusters["cluster"] = clusters 

    titre= f"Carte en 2 dimensions des vecteurs des ingrédients avec leurs clusters\nMéthodes : UMAP, Kmeans({best_nb} clusters) "
    plot_results_clusters(df_results_clusters, titre)


    list_columns=["init_dim_"+str(dim+1) for dim in range(len(model[list(model.wv.vocab.keys())[0]]))]
    df_results_clusters[list_columns]=df_results_clusters.apply(lambda row: model.wv[row.ingredient], axis='columns', result_type='expand') 

    return df_results_clusters

    #plot_results(new_values, labels, "Carte en 2 dimensions des vecteurs des ingrédients \nMéthode : UMAP")


def plot_results_clusters(df, titre): 
 
    plt.figure(figsize=(21, 20)) 
    plt.gcf().set_dpi(300)

    for cluster in df["cluster"].unique():

        df_temp = df[df["cluster"]==cluster] 
        plt.scatter(df_temp["2Dvecteur_dim1"], df_temp["2Dvecteur_dim2"], label = df_temp["cluster"].iloc[0])
        for i in range(len(df_temp)) :
            plt.annotate(df_temp["ingredient"].iloc[i],
                             xy =(df_temp["2Dvecteur_dim1"].iloc[i], df_temp["2Dvecteur_dim2"].iloc[i]), 
                             xytext=(5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom', fontsize=3)
    plt.title(titre, fontsize=20)
    plt.show()


def clustering_products(tokens,labels, nb_clusters= -1, plot=True,annotate=-1): 
    reducer = umap.UMAP(random_state=42)
    data = np.array(tokens)
    scaled_data= StandardScaler().fit_transform(data)
    new_values = reducer.fit_transform(scaled_data)

    best_nb= 2

    if nb_clusters==-1 :
        results_nb_clusters= kmeans_best_nb_clusters(new_values, limit=50, plot=plot)

        best_nb = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "nb_clusters"].iloc[0]
        best_score = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "score"].iloc[0]

        print(f"D'après le score de silhouette le meilleur nombre de clusters pour le clustering est de {best_nb} pour un score de {np.round(best_score,3)}")
    else : 
        best_nb=nb_clusters

    # fit du Kmeans : 
    kmeans = KMeans(n_clusters= best_nb)
    clusters=kmeans.fit_predict(new_values) 

    df_results_clusters = pd.DataFrame()
    df_results_clusters["product"] = labels
    df_results_clusters["2Dvecteur_dim1"] = [value[0] for value in new_values]
    df_results_clusters["2Dvecteur_dim2"]= [value[1] for value in new_values]

    df_results_clusters["cluster"] = clusters 

    titre= f"Carte en 2 dimensions des vecteurs des produits avec leurs clusters\nMéthodes : UMAP, Kmeans({best_nb} clusters) "
    plot_results_clusters_products(df_results_clusters, titre,annotate=annotate)

    return df_results_clusters

def plot_results_clusters_products(df, titre,annotate=-1): 
 
    plt.figure(figsize=(21, 20)) 
    plt.gcf().set_dpi(300)

    for cluster in df["cluster"].unique():

        df_temp = df[df["cluster"]==cluster] 
        plt.scatter(df_temp["2Dvecteur_dim1"], df_temp["2Dvecteur_dim2"], label = df_temp["cluster"].iloc[0])
        for i in range(len(df_temp)) :
            if i%annotate==0 and annotate !=-1 : 
                plt.annotate(df_temp["product"].iloc[i],
                                xy =(df_temp["2Dvecteur_dim1"].iloc[i], df_temp["2Dvecteur_dim2"].iloc[i]), 
                                xytext=(5, 2),
                                textcoords='offset points',
                                ha='right',
                                va='bottom', fontsize=3)
    plt.title(titre, fontsize=20)
    plt.show()

def grid_scatter_plot_clusters_product(df,title,  n_cols=2) : 
    nb_clusters= df["cluster"].nunique()
    n_rows= int(np.ceil(nb_clusters/n_cols))

    plt.gcf().set_dpi(300)
    fig=  plt.figure(figsize=(n_cols*10, n_rows*4+2))
    outer = gridspec.GridSpec(n_rows, n_cols,wspace=0.1, hspace=0.3)

    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)]) for j in range(nb_clusters)]

    index_cluster= 0
    for row in range(n_rows) :
        for col in range(n_rows) : 
            if index_cluster == nb_clusters : 
                break
            
            df_temp = df[df["cluster"]==index_cluster]
            inner =gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[index_cluster],wspace=0.05, hspace=0.1)

            #Scatter plot : 
            ax= plt.Subplot(fig,inner[0])
            ax.scatter( df_temp["2Dvecteur_dim1"], df_temp["2Dvecteur_dim2"],c=colors[index_cluster]  )
   
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"Cluster : {index_cluster}")
            #ax.set_xticks([])
            #ax.set_yticks([])
            fig.add_subplot(ax)


            #WordCloud : 
            ax=plt.Subplot(fig,inner[1])
            
            list_products = df_temp ["product"].tolist()
            #print(list_products)
            wordcloud= WordCloud(width = 600, height=400).generate(" ".join(list_products))

            ax.imshow(wordcloud)            
            ax.axis("off")
            ax.grid(None)
            ax.set_title(f"{len(df_temp)} produits ")
            fig.add_subplot(ax)
            index_cluster+=1

    plt.tight_layout()
    plt.show()

def kmeans_best_nb_clusters_products(values, min=10,limit=75, plot=False) : 

    scores_list =[]
    clusters = [k for k in range(min, limit+1,5)]
    for k in clusters : 
        model = KMeans(n_clusters=k)
        model.fit(values)
        pred = model.predict(values)
        score = silhouette_score(values, pred)
        scores_list.append(score)
    
    results = pd.DataFrame()
    results["nb_clusters"] = clusters
    results["score"] = scores_list
 
    if plot==True : 
        sns.lineplot(data= results, x ="nb_clusters", y="score")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Score de silhouette obtenu")
        plt.title("Score de silhoutte obtenu en fonction du nombre de clusters\n", fontsize=20)
        plt.show()
    
    return results


def clustering_products_bf_reduc(tokens,labels, nb_clusters= -1, plot=True,annotate=-1): 
    
    data = np.array(tokens)
    scaled_data= StandardScaler().fit_transform(data)
    

    best_nb= 2

    if nb_clusters==-1 :
        results_nb_clusters= kmeans_best_nb_clusters_products(scaled_data, limit=50, plot=plot)

        best_nb = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "nb_clusters"].iloc[0]
        best_score = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "score"].iloc[0]

        print(f"D'après le score de silhouette le meilleur nombre de clusters pour le clustering est de {best_nb} pour un score de {np.round(best_score,3)}")
    else : 
        best_nb=nb_clusters

    # fit du Kmeans : 
    kmeans = KMeans(n_clusters= best_nb)
    clusters=kmeans.fit_predict(scaled_data) 

    reducer = umap.UMAP(random_state=42)
    new_values = reducer.fit_transform(scaled_data)



    df_results_clusters = pd.DataFrame()
    df_results_clusters["product"] = labels
    df_results_clusters["2Dvecteur_dim1"] = [value[0] for value in new_values]
    df_results_clusters["2Dvecteur_dim2"]= [value[1] for value in new_values]

    df_results_clusters["cluster"] = clusters 

    titre= f"Carte en 2 dimensions des vecteurs des produits avec leurs clusters\nMéthodes : UMAP, Kmeans({best_nb} clusters) "
    plot_results_clusters_products(df_results_clusters, titre,annotate=annotate)

    return df_results_clusters


def clustering_products_bf_reducPCA(tokens,labels, nb_clusters= -1, plot=True,annotate=-1): 
    
    data = np.array(tokens)
    scaled_data= StandardScaler().fit_transform(data)
    

    best_nb= 2

    if nb_clusters==-1 :
        results_nb_clusters= kmeans_best_nb_clusters_products(scaled_data, limit=50, plot=plot)

        best_nb = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "nb_clusters"].iloc[0]
        best_score = results_nb_clusters.loc[results_nb_clusters["score"]==results_nb_clusters["score"].max(), "score"].iloc[0]

        print(f"D'après le score de silhouette le meilleur nombre de clusters pour le clustering est de {best_nb} pour un score de {np.round(best_score,3)}")
    else : 
        best_nb=nb_clusters

    # fit du Kmeans : 
    kmeans = KMeans(n_clusters= best_nb)
    clusters=kmeans.fit_predict(scaled_data) 

    pca = PCA(n_components=2)
    new_values = pca.fit_transform(data)


    df_results_clusters = pd.DataFrame()
    df_results_clusters["product"] = labels
    df_results_clusters["2Dvecteur_dim1"] = [value[0] for value in new_values]
    df_results_clusters["2Dvecteur_dim2"]= [value[1] for value in new_values]

    df_results_clusters["cluster"] = clusters 

    titre= f"Carte en 2 dimensions des vecteurs des produits avec leurs clusters\nMéthodes : ACP, Kmeans({best_nb} clusters) "
    plot_results_clusters_products(df_results_clusters, titre,annotate=annotate)

    return df_results_clusters