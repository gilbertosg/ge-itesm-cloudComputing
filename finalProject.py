# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:44:04 2016

@author: Gilberto Silva González
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import statsmodels.formula.api as smf
import statsmodels.api as sm



#%% Iniciación de los datos
df = pd.read_csv('engineering_data - blade_damage_assessment.csv')
df.dtypes
df.head(10)
df.dropna(axis = 0, inplace = True)
prev = len(df) #Número original de líneas

#%% 1. Convertir strings a NaNs en columnas numéricas
nc = list(df.columns.values)
col_int = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
col_temp = [6,7,8,9,10]
col_resto = [0,4,5,11,12,13,14,15,16]
#%%
def Cleansing(df, col_int, col_temp, col_resto):

    prev = len(df)
    
    nc = list(df.columns.values)
    
    for fila in range (0, len(col_int)):
        df[nc[col_int[fila]]] = pd.to_numeric(df[nc[col_int[fila]]], errors = 'coerce')

    df.dropna(axis = 0, inplace = True)   
    
    for fila in range (0, len(col_temp)):
        df = df.drop(df[(df[nc[col_temp[fila]]] < -273.15 )].index)
    
    for fila in range (0, len(col_resto)):
        df = df.drop(df[(df[nc[col_resto[fila]]] < 0)].index)
        
    cleansing = prev - len(df)
    
    return [df,cleansing]


# Exportar a .csv archivo "limpio"
# df.to_csv('data_cleansed.csv', index = False)
#%%
df, cleansed = Cleansing(df, col_int, col_temp, col_resto)

#%% 2.Agrupación por engine_id
num_flights = df[['engine_type','engine_id']].groupby('engine_id').count() #Número de vuelos por engine
#Medias por engine_id y engine_type
lista = list(df.columns.values)
del lista[0:4:1]
del lista[1]
ll = len(lista)
group = {}
for row in range (0,ll):
    group["var{0}".format(row)] = df.groupby(['engine_type', 'engine_id'])[lista[row]].mean()

#%% 3. Análisis Exploratorio
#a. 
num_engine_ids = len(set(df['engine_id']))
num_engine_types = len(set(df['engine_type']))
num_customers = len(set(df['customer']))
tot_flights = len(df)   
name_customers = (set(df['customer']))

#b.
summ_var = df[lista].describe()

#c. Boxplots
for x in range (0,ll):
    df.boxplot(column = lista[x], by = 'customer')
    
#c.2 Boxplots by category
for y in range (0,ll):
    df.boxplot(column = lista[y], by = 'category')

#%% 3.d
customer_groups = df.groupby('customer')

customer_ACC = customer_groups.get_group('ACC')
num_flights_ACC = customer_ACC[['engine_type', 'engine_id']].groupby('engine_id').count()
customer_ASI = customer_groups.get_group('ASI')
num_flights_ASI = customer_ASI[['engine_type', 'engine_id']].groupby('engine_id').count()
customer_DME = customer_groups.get_group('DME')
num_flights_DME = customer_DME[['engine_type', 'engine_id']].groupby('engine_id').count()
customer_FAR = customer_groups.get_group('FAR')
num_flights_FAR = customer_FAR[['engine_type', 'engine_id']].groupby('engine_id').count()
customer_SLA = customer_groups.get_group('SLA')
num_flights_SLA = customer_SLA[['engine_type', 'engine_id']].groupby('engine_id').count()

col_int = ['t_1', 't_4', 't_oil', 'fan_speed', 'thrust']

fig, ((ax1,ax2),(ax3,ax4),(ax5)) = plt.subplots(nrows=3, ncols=2, figsize=(11,9))

for f,ax in zip(col_int,(ax1,ax2,ax3,ax4,ax5)):
    # Plot density of ACC
    customer_ACC[f].plot(kind='kde', ax=ax, color='r', label='ACC')
    ax.set_title(f)
    ax.grid(True)
    # Plot density of ASI
    customer_ASI[f].plot(kind='kde', ax=ax, color='g', label='ASI')
    ax.set_title(f)
    ax.grid(True)
    # Plot density of DME
    customer_DME[f].plot(kind='kde', ax=ax, color='b', label='DME')
    ax.set_title(f)
    ax.grid(True)
    # Plot density of FAR
    customer_FAR[f].plot(kind='kde', ax=ax, color='c', label='FAR')
    ax.set_title(f)
    ax.grid(True)
    # Plot density of SLA
    customer_SLA[f].plot(kind='kde', ax=ax, color='k', label='SLA')
    ax.set_title(f)
    ax.grid(True)
    ax.legend(loc='best')    

#%% Función de variables importantes por cliente
def Important_Variables_Client(df_cl, num_flights_cl):
    """
    Función que genera una lista con las variables con mayor significancia entre 
    turbinas que han fallado contra las que no.
    Entradas: 
    - df_cl: Dataframe original segmentado por cliente.
    - num_flights_cl: número de vuelos por turbina por cliente.
    
    Salidas: 
        - Lista de las variables más significativas.
        - Resumen agrupado por turbinas de las variables más significativas.
        - Dataframe original con solamente las columnas más significativas.
        
    """
    
    lista = list(df.columns.values)
    del lista[0:4:1]
    del lista[1]
      
    failed = df_cl[(df_cl.category == "failed")]
    failed_summ = failed.groupby('engine_id').min()
    failed_summ2 = failed.groupby('engine_id').mean()
    failed_summ2.drop(['flight_id'], axis = 1, inplace = True)
    
    non_failed = df_cl[(df_cl.category == "non-failed")]
    non_failed_summ = non_failed.groupby('engine_id').max()
    non_failed_summ2 = non_failed.groupby('engine_id').mean()
    non_failed_summ2.drop(['flight_id'], axis = 1, inplace = True)

    if failed.empty:
        print('Este cliente no tiene turbinas con fallas')
        sys.exit()
    else:        
        lista.append('category')
        
        cat = (failed_summ['category'])
        failed_summ2['category'] = cat
        cat2 = (non_failed_summ['category'])
        non_failed_summ2['category'] = cat2
        
        df_summ = failed_summ2
        df2_cl  = df_summ.append(non_failed_summ2)
        
        df2_cl = df2_cl[lista]
        df2_cl = df2_cl.sort_index(axis = 0)
        num_flights_cl = num_flights_cl.sort_index(axis = 0)
        df2_cl['num_flights'] = num_flights_cl
        
        lista.remove('category')
        
        summ_failed = failed[lista].describe()
        summ_notFailed = non_failed[lista].describe()
        
        num_failed = df2_cl[(df2_cl.category == "failed")]
        num_nfailed = df2_cl[(df2_cl.category == "non-failed")]
        
        lista.append('num_flights')
        
        num_failed = num_failed['num_flights'].describe()
        num_nfailed = num_nfailed['num_flights'].describe()
    
        summ_failed['num_flights'] = num_failed
        summ_notFailed['num_flights'] = num_nfailed
    
        dist = np.ones((8,13))
        
        for j in range(0, len(summ_failed.columns)):
            for i in range (0, len(summ_failed)):
                dist[i,j] = abs((summ_failed[lista[j]][i] - summ_notFailed[lista[j]][i])/summ_notFailed[lista[j]][i])*100
            
        np.savetxt("Var.csv", dist, delimiter = ",")
        var = pd.read_csv('Var.csv', header = None)
        var.columns = lista
        var.index = list(summ_failed.index)
        var = var.drop(var.index[0])
        
        dist = np.ones((len(var), len(var.columns)))
        for j in range(0,len(var.columns)):
            for i in range(0,len(var)):
                if var.iat[i,j] > 10:
                    dist[i,j] = 1
                elif var.iat[i,j] < 10 and var.iat[i,j] > 5:
                    dist[i,j] = 0.5
                else:
                    dist[i,j] = 0
        
        np.savetxt("Var_mat.csv", dist, delimiter = ",")
        var_mat = pd.read_csv('Var_mat.csv', header = None)
        var_mat.columns = var.columns
        var_mat.index = list(var.index)
        
        dist = np.zeros((1, len(var_mat.columns)))
        for j in range(0, len(var_mat.columns)):
            if var_mat[[j]].sum().item() > 2:
                dist[0,j] = 1
            else:
                dist[0,j] = 0
                
        np.savetxt("Mat_Imp.csv", dist, delimiter = ",")
        mat_imp = pd.read_csv('Mat_Imp.csv', header = None)
        mat_imp.columns = var.columns
        
        mat_imp = mat_imp.loc[:, (mat_imp != 0).any(axis=0)]
        lista_imp_cl = list(mat_imp.columns.values)
        
        lista_imp_cl.append('category')
        
        df2_summ_cl = df2_cl[lista_imp_cl]
        
        for i in range (0, len(lista_imp_cl) - 1):
            if lista_imp_cl[i] == "num_flights":
                lista_imp_cl.remove('num_flights')
                
        df_cl_imp = df_cl[lista_imp_cl]
        
        
        return[lista_imp_cl, df2_summ_cl, df_cl_imp]
        
#%% Exportar archivos .csv con análisis previo
lista_imp_ACC, ACC_summ, ACC_imp = Important_Variables_Client(customer_ACC, num_flights_ACC)
lista_imp_ASI, ASI_summ, ASI_imp = Important_Variables_Client(customer_ASI, num_flights_ASI)
lista_imp_DME, DME_summ, DME_imp = Important_Variables_Client(customer_DME, num_flights_DME)
lista_imp_FAR, FAR_summ, FAR_imp = Important_Variables_Client(customer_FAR, num_flights_FAR)
lista_imp_SLA, SLA_summ, SLA_imp = Important_Variables_Client(customer_SLA, num_flights_SLA)
lista_imp    , summ_imp, tot_imp = Important_Variables_Client(df, num_flights)





#%% 5 Plots
# Scatterplots of Important
#a. Total
sns.set()
sns.pairplot(summ_imp, hue = "category", palette = "hls")

#b. DME
sns.set()
sns.pairplot(DME_imp, hue = "category", palette = "hls")


# Density
sns.set(style = "dark")
f, axes = plt.subplots(3,3, figsize = (9,9), sharex = True, sharey = True)

for ax, s in zip(axes.flat, np.linspace(0,3,10)):
    cmap = sns.cubehelix_palette(start = s, light = 1, as_cmap = True)
    sns.kdeplot(summ_imp['t_1'], summ_imp['damage'], cmap = cmap, shade = True, cut = 5, ax = ax)
    
f.tight_layout()

#%% 6. Correlation
# Matrix

Corr_Mat = df[lista].corr()
Corr_Mat = Corr_Mat[(Corr_Mat <> 1)]
Corr_Mat = Corr_Mat.fillna(value = 0)
Max_Corr = Corr_Mat.max()
Max_Var = Corr_Mat.idxmax()

# Damage
damage_corr = Corr_Mat['damage']
damage_corr = damage_corr ** 2
damage_corr = damage_corr.sort_values(axis = 0, ascending = False)

#%% 7. Models
# Modelo multivariado
mod_1 = smf.ols(formula = 'damage ~ core_speed + fan_speed + t_4 + thrust + C(customer) + t_1 + t_2 + t_3', data = df)
res_1 = mod_1.fit()
print res_1.summary()

pred = res_1.predict()
pred = pd.DataFrame(pred)
pred.columns = ['predict']
pred = pred.set_index(df.index)

df['predict'] = pred.predict

failed = df[(df.category == "failed")]
failed_summ = failed.groupby('engine_id').min()
failed_summ2 = failed.groupby('engine_id').mean()

non_failed = df[(df.category == "non-failed")]
non_failed_summ = non_failed.groupby('engine_id').max()
non_failed_summ2 = non_failed.groupby('engine_id').mean()

cat = (failed_summ['category'])
failed_summ2['category'] = cat
cat2 = (non_failed_summ['category'])
non_failed_summ2['category'] = cat2

df_summ = failed_summ2
df_summ  = df_summ.append(non_failed_summ2)
df_summ = df_summ.sort_index(axis = 0)


pred_engine = pd.DataFrame(df_summ['predict'])
pred_status = np.zeros((400,1))
pred_status = pred_status.astype(str)
pred_status = pd.DataFrame(pred_status)
pred_status.columns = ['category']

for i in range(0, len(pred_engine)):
    if pred_engine['predict'].iat[i] > 50:
        pred_status['category'].iat[i] = 'failed'
    else:
        pred_status['category'].iat[i] = 'non-failed'
        

    
pred_status = pred_status.set_index(df_summ.index)
df_summ['predict_category'] = pred_status['category']   

df_summ['precision'] = df_summ['category'] == df_summ['predict_category']

# Modelo logístico
failed_summ2['customer'] = failed_summ.customer
non_failed_summ2['customer'] = non_failed_summ.customer
df_log_summ = failed_summ2
df_log_summ = df_log_summ.append(non_failed_summ2)
df_log_summ = df_log_summ.sort_index(axis = 0)
failed_dummy = pd.get_dummies(df_log_summ['category'])['failed']
failed_dummy = pd.DataFrame(failed_dummy)

df_log_summ['failed_dummy'] = failed_dummy['failed']

cols_to_keep = ['failed_dummy','t_1', 't_4', 'thrust', 'fan_speed', 'core_speed']
dummy_ranks = pd.get_dummies(df_log_summ['customer'], prefix='customer')

data = df_log_summ[cols_to_keep]#.join(dummy_ranks.ix[:, 'customer_ASI':])

train_cols = data.columns[1:]

logit = sm.Logit(data['failed_dummy'], data[train_cols])

result = logit.fit()
print result.summary()

bb = df_log_summ[train_cols]
bb['failed_pred'] =     result.predict(bb)


#%% Prueba de modelo
datos_prueba = pd.read_csv('DF_prueba.csv')
col_int_prueba = [0,3,4,5,6,7,8,9,10,11,12,13,14]
col_temp_prueba = [4,5,6,7,8]
col_resto_prueba = [0,9,10,11,12]

Cleansing(datos_prueba, col_int_prueba, col_temp_prueba, col_resto_prueba)

datos_prueba['damage_pred'] = res_1.predict(datos_prueba)
engine_group = datos_prueba.groupby('engine_id').mean()
customer = datos_prueba.groupby('engine_id').min()

engine_group['customer'] = customer['customer']


damage_pred_engine = pd.DataFrame(engine_group['damage_pred'])
pred_stat = np.zeros((100,1))
pred_stat = pd.DataFrame(pred_stat)
pred_stat.columns = ['failed']

for i in range(0, len(engine_group)):
    if damage_pred_engine['damage_pred'].iat[i] > 50:
        pred_stat['failed'].iat[i] = 1
    else:
        pred_stat['failed'].iat[i] = 0

pred_stat = pred_stat.set_index(engine_group.index)
engine_group['predicted_category'] = pred_stat['failed']
failed_engines_test = engine_group['predicted_category']  
failed_engines_test = pd.DataFrame(failed_engines_test) 
failed_engines_test.to_csv('Failed_Engines.csv', index = True)

#%% Plots

customer_ACC = customer_ACC.groupby('engine_id').mean()
customer_ASI = customer_ASI.groupby('engine_id').mean()
customer_DME = customer_DME.groupby('engine_id').mean()
customer_FAR = customer_FAR.groupby('engine_id').mean()
customer_SLA = customer_SLA.groupby('engine_id').mean()


sns.set(style="darkgrid")
f, ax = plt.subplots(figsize = (10,10))

ax = sns.kdeplot(customer_ACC.t_1, customer_ACC.damage, cmap = "Reds", shade = True, shade_lowest = False)
ax = sns.kdeplot(customer_ASI.t_1, customer_ASI.damage, cmap = "Greens", shade = True, shade_lowest = False)
ax = sns.kdeplot(customer_DME.t_1, customer_DME.damage, cmap = "Blues", shade = True, shade_lowest = False)
ax = sns.kdeplot(customer_FAR.t_1, customer_FAR.damage, cmap = "Oranges", shade = True, shade_lowest = False)
ax = sns.kdeplot(customer_SLA.t_1, customer_SLA.damage, cmap = "Greys", shade = True, shade_lowest = False)

red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
green = sns.color_palette("Greens")[-2]
orange = sns.color_palette("Oranges")[-2]
grey = sns.color_palette("Greys")[-2]

ax.text(22.5, 10, "ACC", size=16, color=red)
ax.text(23.5, 50, "FAR", size=16, color=orange)
ax.text(25.3, 0, "SLA", size = 16, color = grey)
ax.text(25.3, 90, "ASI", size = 16, color = green)
ax.text(28, 90, "FAR", size = 16, color = blue)
