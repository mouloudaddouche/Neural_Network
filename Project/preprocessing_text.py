import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import arff
import os, shutil, sys


#********************** E/S **************************#

#CREATION D'UN DOSSIER TEMPORAIRE (POUR LES OUTILS DE VISUALISATION)
def create_tmp_folder():
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

#SUPPRESSION DU CONTENU DU DOSSIER TEMPORAIRE
def remove_tmp_folder_content():
    if os.path.exists('tmp'):
        shutil.rmtree('tmp')

#SUPPRESSION DU DOSSIER TEMPORAIRE
def remove_tmp_folder():
    if os.path.exists('tmp'):
        remove_tmp_folder_content()
        os.rmdir('tmp')

#SUPPRESSION DES FICHIERS GENERES AVEC MATPLOTLIB
def remove_generated_file( fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


#CHARGER LES DATASETS ARFF
def dataset_loader_arff(file_path):
    name_file = file_path.split('/')[-1]
    ext = name_file.split('.')[-1]
    raw_data = arff.loadarff(file_path)
    df = pd.DataFrame(raw_data[0])
    attributes = get_attributes(df)
    #pour l'encodage
    for att in attributes:
        if df[att].dtypes == "object":
            df[att] = df[att].str.decode("utf-8")
    #REMPLACER LES ? PAR NP.NAN
    #CREATION D'UNE LISTE ATTRIBUTES QUI CONTIENT QUE LES ATTRIBUTS NOMINAUX
    attributes = []
    atts = get_attributes(df)
    for att in atts:
        if is_nominal(att, df):
            attributes.append(att)
    for att in attributes:
        df[att]= df[att].replace('?', np.nan)
    df = df.replace('?', np.nan)
    return df

#CHARGER LES DATASETS CSV
def dataset_csv_loader(file_path, dataset):
    df=dataset
    name_file = file_path.split('/')[-1]
    ext = name_file.split('.')[-1]
    df = pd.read_csv(file_path, sep=',')
    return df

#*************************************** DATA TRANSFORMATION *****************************************************#
def min_max_normalize_dataset(dataset):
    df=dataset
    attributes = get_attributes(df)
    att_class=attributes[df.shape[1]-1]
    #POUR NE PAS MODIFIER L'ATTRIBUT CLASS
    del attributes[df.shape[1]-1] 
    etique_class=0
    values_class={}
    for val in df[att_class]:
        if not pd.isnull(val):
            if val not in values_class:
                values_class[val]=etique_class
                etique_class=etique_class+1
    for val in df[att_class]:
        if not pd.isnull(val):
                if val in values_class:
                    df[att_class] = df[att_class].replace(val, values_class[val])

    for att in attributes:
        if not is_nominal(att, df):
            x = pd.DataFrame({att: df[att]})
            min_max_scalar = MinMaxScaler()
            scaled = min_max_scalar.fit_transform(x)
            df[att] = pd.DataFrame(scaled)
        else:
        	#JE CREE UN DICTIONNAIRE VALUES 
        	#LA CLE EST LA VALEUR QUE PREND L'ATTRIBUT
        	#LA VALEUR EST SON ETIQUETTE, ON COMMENCE 0
            etique=0
            values = {}          
            for val in df[att]:         
                if not pd.isnull(val):
                    if val not in values:
                         values[val] = etique
                         etique = etique + 1
            for val in df[att]:
                if not pd.isnull(val):
                    if val in values:
                        df[att] = df[att].replace(val, values[val]/(etique-1))   	        

def decimal_normalize_dataset(dataset):
    df=dataset
    attributes = get_attributes(df)
    att_class=attributes[df.shape[1]-1]
    #POUR NE PAS MODIFIER L'ATTRIBUT CLASS
    del attributes[df.shape[1]-1] 
    etique_class=0
    values_class={}
    for val in df[att_class]:
        if not pd.isnull(val):
            if val not in values_class:
                values_class[val]=etique_class
                etique_class=etique_class+1
    for val in df[att_class]:
        if not pd.isnull(val):
                if val in values_class:
                    df[att_class] = df[att_class].replace(val, values_class[val])

    for att in attributes:
        if not is_nominal(att, df):
            p = return_max(att, df)
            q = len(str(abs(p)))
            df[att] = df[att] / 10 ** q
        else:
        	#JE CREE UN DICTIONNAIRE VALUES 
        	#LA CLE EST LA VALEUR QUE PREND L'ATTRIBUT
        	#LA VALEUR EST SON ETIQUETTE
            etique=0
            values = {}          
            for val in df[att]:         
                if not pd.isnull(val):
                    if val not in values:
                         values[val] = etique 
                         etique = etique + 1
            for val in df[att]:
                if not pd.isnull(val):
                    if val in values:
                        df[att] = df[att].replace(val, values[val]/(etique-1))


#*************************** LES VALEURS STATISTIQUES ******************************************#

def get_number_instances(dataset):
    df=dataset
    return df.shape[0]

def get_number_attributes(dataset):
    df=dataset
    return df.shape[1]

#RETOURNE LE NOMBRE DE VALEURS MANQUANTES TOTAL
def get_number_missing_values(dataset):
    df=dataset
    nb = 0
    att = get_attributes(df)
    for i in att:
        nb = nb + df[i].isnull().sum()
    return nb

def return_average( attribute, dataset):
    df=dataset
    return np.mean(df[attribute])

def return_stdv( attribute, dataset):
    df=dataset
    return np.std(df[attribute])

def return_mode( attribute, dataset):
    df=dataset
    return df[attribute].mode()[0]

def return_min( attribute, dataset):
    df=dataset
    return np.min(df[attribute])

def return_max( attribute, dataset):
    df=dataset
    return np.max(df[attribute])

#******************************** Traitement sur les attributs *************************************************#

# RETOURNE LES ATTRIBUTS DU DATASET
def get_attributes(dataset):
    df=dataset
    attributes = list(df.columns.values)
    return attributes

#RETOURNE LES INSTANCES
def get_instances(dataset): 
    df=dataset
    instances = {}
    i=1
    for inst in df.values:
        x = list(inst)
        instances[i] = ",".join(str(e) for e in x).split(",")
        i+=1
    return instances

#POUR SAVOIR SI L'ATTRIBUT EST NOMINAL OU PAS
def is_nominal( attribute, dataset):
    df=dataset
    if df[attribute].dtypes == "object":
        return True
    else:
        return False

#RETOURNE LE NOMBRE DE VALEURS MANQUANTES PAR ATTRIBUT
def get_number_missing_values_att( i, dataset):
    df=dataset
    atts = get_attributes(df)
    att = atts[i]
    nb = 0
    nb = df[att].isnull().sum()
    return nb


#RETOURNE LE NOMBRE DE VALEURS DISTINCTES DE L'ATTRIBUT i
def get_number_values_att( i, dataset):
    df=dataset
    vals = get_number_values_val_att(i, df)
    return len(vals)


#CREE UN DICTIONNAIRE CONTENANT LES VALEURS DE CHAQUE ATTRIBUT i AINSI QUE LEUR NOMBRE
def get_number_values_val_att( i, dataset):
    df=dataset
    values = {}
    atts = get_attributes(df)  
    att = atts[i]               
    for val in df[att]:         
        if not pd.isnull(val):
            if val in values:
                values[val] += 1
            else:
                values[val] = 1
    return values

#******************************** DATA CLEANING *************************************************#

# REMPLISSAGE DES VALEURS MANQUANTES PAR LA MOYENNE/LE MODE DE L'ATTRIBUT
def fill_missing_values(dataset):
    df=dataset
    attributes = get_attributes(df)
    for att in attributes:
        if df[att].isnull().sum() != 0:    #SI L'ATTRIBUT CONTIENT DES MISSING VALUES
            if is_nominal(att, df) == True:
                df[att] = df[att].replace(np.nan, str(return_mode(att, df)))
            else:
                df[att] = df[att].replace(np.nan, return_average(att, df))


# REMPLISSAGE DES VALEURS MANQUANTES SE BASANT SUR LA CLASSE
def fill_missing_values_class(dataset):
    df=dataset
    attributes = get_attributes(df)
    #AFFICHER LES CLASSES DU DATASET
    tmp= list(df[attributes[df.shape[1]-1]])
    classes = []
    for elt in tmp:
        if elt not in classes:
            classes.append(elt)
    for att in attributes:
        if df[att].isnull().sum() != 0:    #SI L'ATTRIBUT CONTIENT DES MISSING VALUES
            for item_class in classes:
                if is_nominal(att, df):
                    #get mode by class
                    values = df.loc[df[attributes[df.shape[1]-1]] == item_class, att]   #[attributes[df.shape[1]-1]]=='class'
                    valeur_mode = values.mode()[0]
                    df.loc[df[attributes[df.shape[1]-1]]== item_class, att] = df.loc[
                        df[attributes[df.shape[1]-1]] == item_class, att].replace(np.nan, valeur_mode)
                else:
                    #get moyenne by class
                    values = df.loc[df[attributes[df.shape[1]-1]] == item_class, att]
                    valeur_moyenne = np.mean(values)
                    df.loc[df[attributes[df.shape[1]-1]] == item_class, att] = df.loc[
                        df[attributes[df.shape[1]-1]] == item_class, att].replace(np.nan, valeur_moyenne)



#*********************************** VISUALISATION **********************************************#

#RETOURNE L'ATTRIBUT VIA L'INDEX
def get_att_by_index(i, dataset):
    df=dataset
    atts = get_attributes(df)
    att = atts[i]
    return att

def boxplot(dataset):
    df=dataset
    sns.boxplot(data=df, orient='h')
    plt.legend()
    text = "tmp/boxplot.png"
    remove_generated_file(text)
    plt.savefig(text, bbox_inches='tight')
    plt.clf()

def histogram_numeral( i, dataset):
    df=dataset
    attribute = get_att_by_index(i, df)
    liste = []
    for j in df[attribute]:
        if not pd.isnull(j):
            liste.append(j)
    sns.distplot(liste, color="skyblue", label='')
    text = "tmp/histo_" + str(i) + ".png"
    remove_generated_file(text)
    plt.savefig(text, bbox_inches='tight')
    plt.clf()

def histogram_nominal( i, dataset):
    df=dataset
    attribute = get_att_by_index(i,df)
    df[attribute].replace(np.nan, 'Missing')
    df[attribute].value_counts().plot(kind="bar", color=(0.7, 0.91, 1.0, 0.6), edgecolor=(0.2, 0, 1.0, 0.6), width=0.1, label='')
    plt.xticks(rotation=30)
    df[attribute].replace('Missing', np.nan)
    text = "tmp/histo_" + str(i) + ".png"
    remove_generated_file(text)
    plt.savefig(text, bbox_inches='tight')
    plt.clf()
