

Regina Rivero Perkins - A01017800

Gilberto Silva - A01018723

Johnny Turquie - A01021448


**GE, Data Science Lectures**

Blade type ABC123 was found to be deteriorated before the expected lifespan for engine EX-50. These blades are located in the high pressure turbine section of the engine.

It is of outmost importance to care and watch for the state of the blades of a motor in order to guarantee the well being of the plane, the flight and, most importantly, the passengers of each plane. A specific type of blade turned out to be deteriorated before the expected life of the motor it belongs to. With a set of data, it is possible to predict the conditions in which that blade is damaged more or less, in order to assess the context and provide planned maintenance to avoid accidents and extend the useful life of the blades of a motor.

**Objective:** To provide with a mathematical model that can predict, accurately, the current status of the blades inside a motor and provide with a solution for planned removals of engines, reduce maintenance pressure, increase engine availability and avoid delays and cancellations.

**Project**

1. Data Cleansing

For the data cleansing, we began by excluding all data lines that were incomplete and were of an incorrect type: strings in numerical columns. We continued by eliminating all those that included numbers that were unreal: negative values for pressure, temperatures under negative 273.15 Celsius degrees, negative velocities and negative vibrations.

-----------------------------------------------------------------------

#%% Convert strings to NaNs (function cleansing)

nc = list(df.columns.values)

col\_int = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

col\_temp = [6,7,8,9,10]

col\_resto = [0,4,5,11,12,13,14,15,16]

#%% Function cleansing

def Cleansing(df, col\_int, col\_temp, col\_resto):

  prev = len(df)

  nc = list(df.columns.values)

  for fila in range (0, len(col\_int)):

    df[nc[col\_int[fila]]] = pd.to\_numeric(df[nc[col\_int[fila]]], errors = &#39;coerce&#39;)

  df.dropna(axis = 0, inplace = True)

  for fila in range (0, len(col\_temp)):

    df = df.drop(df[(df[nc[col\_temp[fila]]] &lt; -273.15 )].index)

  for fila in range (0, len(col\_resto)):

     df = df.drop(df[(df[nc[col\_resto[fila]]] &lt; 0)].index)

  cleansing = prev - len(df)

  return [df,cleansing]

--------------------------------------------------------------

2. Data Pre-Processing

Assuming that each line is a different recorded flight, we were able to aggregate data by engine ID, calculate the number of flights per engine and create a table combining them.

3. Eploratory Analysis

Our analysis started with the understanding of each separate variable and their relations, getting to define the following:

| Different engine IDs | 400  |
| --- | --- |
| Number of engine type | 2   |
| Customers | 5  |
| Total flights | 1,787,418  |



With the scatter, and a set of box plots by customer for each of the variables, we can tell that there are some variables that are having no variation between the different customers. The variables that do not change significantly amongst customers are oil temperature and pressure, both compressor temperatures (inlet and outlet) and the results from both vibration sensors.

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/1-boxplots.png)



In the following scatter flows we were able to identify some very relevant dependencies amongst our variables:

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/2-scatterplots.png)


This set of scatter plots help us identify relationships between the variables, tendencies and how they group for failures and non-failures. There are some obvious relationships we can identify through the linearity of the graphs: t4 thrust, core speed fan speed, t3 t1. We can also begin to tell some hypotheses about the behaviour of fails and non-fails: the higher the fan shaft and core shaft rotating speed, the higher the probability the engine will fail.

To better assess the influence of variables, we developed an algorithm that evaluated each variable individually regarding their scope and variation for failure and for each customer. Thus, we were able to discriminate some information and to generate a new set of scatter plots focused on the client that has presented the most failures: Desert Middle East Airlines.

----------------------------------------------------------------

#%% Function to get significant variables of clients

def Important\_Variables\_Client(df\_cl, num\_flights\_cl):

&quot;&quot;&quot;

Get significant variables between turbines that have failed and that haven&#39;t

Inputs:

- df\_cl:Original dataframe segmented by client.

- num\_flights\_cl: num of flights by turbine and client

Outputs:

- List of most significant variables.

- Summary of vars

- Original dataframe with only most important columns

&quot;&quot;&quot;

lista = list(df.columns.values)

del lista[0:4:1]

del lista[1]

failed = df\_cl[(df\_cl.category == &quot;failed&quot;)]

failed\_summ = failed.groupby(&#39;engine\_id&#39;).min()

failed\_summ2 = failed.groupby(&#39;engine\_id&#39;).mean()

failed\_summ2.drop([&#39;flight\_id&#39;], axis = 1, inplace = True)

non\_failed = df\_cl[(df\_cl.category == &quot;non-failed&quot;)]

non\_failed\_summ = non\_failed.groupby(&#39;engine\_id&#39;).max()

non\_failed\_summ2 = non\_failed.groupby(&#39;engine\_id&#39;).mean()

non\_failed\_summ2.drop([&#39;flight\_id&#39;], axis = 1, inplace = True)

if failed.empty:

print(&#39;Este cliente no tiene turbinas con fallas&#39;)

sys.exit()

else:

lista.append(&#39;category&#39;)

cat = (failed\_summ[&#39;category&#39;])

failed\_summ2[&#39;category&#39;] = cat

cat2 = (non\_failed\_summ[&#39;category&#39;])

non\_failed\_summ2[&#39;category&#39;] = cat2

df\_summ = failed\_summ2

df2\_cl = df\_summ.append(non\_failed\_summ2)

df2\_cl = df2\_cl[lista]

df2\_cl = df2\_cl.sort\_index(axis = 0)

num\_flights\_cl = num\_flights\_cl.sort\_index(axis = 0)

df2\_cl[&#39;num\_flights&#39;] = num\_flights\_cl

lista.remove(&#39;category&#39;)

summ\_failed = failed[lista].describe()

summ\_notFailed = non\_failed[lista].describe()

num\_failed = df2\_cl[(df2\_cl.category == &quot;failed&quot;)]

num\_nfailed = df2\_cl[(df2\_cl.category == &quot;non-failed&quot;)]

lista.append(&#39;num\_flights&#39;)

num\_failed = num\_failed[&#39;num\_flights&#39;].describe()

num\_nfailed = num\_nfailed[&#39;num\_flights&#39;].describe()

summ\_failed[&#39;num\_flights&#39;] = num\_failed

summ\_notFailed[&#39;num\_flights&#39;] = num\_nfailed

dist = np.ones((8,13))

for j in range(0, len(summ\_failed.columns)):

for i in range (0, len(summ\_failed)):

dist[i,j] = abs((summ\_failed[lista[j]][i] - summ\_notFailed[lista[j]][i])/summ\_notFailed[lista[j]][i])\*100

np.savetxt(&quot;Var.csv&quot;, dist, delimiter = &quot;,&quot;)

var = pd.read\_csv(&#39;Var.csv&#39;, header = None)

var.columns = lista

var.index = list(summ\_failed.index)

var = var.drop(var.index[0])

dist = np.ones((len(var), len(var.columns)))

for j in range(0,len(var.columns)):

for i in range(0,len(var)):

if var.iat[i,j] &gt; 10:

dist[i,j] = 1

elif var.iat[i,j] &lt; 10 and var.iat[i,j] &gt; 5:

dist[i,j] = 0.5

else:

dist[i,j] = 0

np.savetxt(&quot;Var\_mat.csv&quot;, dist, delimiter = &quot;,&quot;)

var\_mat = pd.read\_csv(&#39;Var\_mat.csv&#39;, header = None)

var\_mat.columns = var.columns

var\_mat.index = list(var.index)

dist = np.zeros((1, len(var\_mat.columns)))

for j in range(0, len(var\_mat.columns)):

if var\_mat[[j]].sum().item() &gt; 2:

dist[0,j] = 1

else:

dist[0,j] = 0

np.savetxt(&quot;Mat\_Imp.csv&quot;, dist, delimiter = &quot;,&quot;)

mat\_imp = pd.read\_csv(&#39;Mat\_Imp.csv&#39;, header = None)

mat\_imp.columns = var.columns

mat\_imp = mat\_imp.loc[:, (mat\_imp != 0).any(axis=0)]

lista\_imp\_cl = list(mat\_imp.columns.values)

lista\_imp\_cl.append(&#39;category&#39;)

df2\_summ\_cl = df2\_cl[lista\_imp\_cl]

for i in range (0, len(lista\_imp\_cl) - 1):

if lista\_imp\_cl[i] == &quot;num\_flights&quot;:

lista\_imp\_cl.remove(&#39;num\_flights&#39;)

df\_cl\_imp = df\_cl[lista\_imp\_cl]

return[lista\_imp\_cl, df2\_summ\_cl, df\_cl\_imp]

----------------------------------------------------------------------------------



 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/3-scatterplots.png)



The second set of scatter plots allow us to visualize those variables that are having a real influence over damage and failure for the client that has presented the most failures over time.

4. Data Clustering

We found data clustering to be easier through heat maps as they visually represent data tendencies and allow us to have a better approach towards conclusions.

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/4-clusters.png)



We can see that centroids are located in the highest density points of the heat map, and that the clusters are very similar to the fails and non fails coloured in the heat map; the latter is despite the fact that heat maps tend to discriminate data that falls far away of the highest density areas. We can then confirm that the average core shaft speed is a very significant variable to predict average damage and, most importantly, failures.

5. Correlation and Variable Selection

With all the cleansed data, preliminary analysis, and the following correlation matrix we can identify and numerically assess the most related variables which are the core shaft and the fan shaft rotating speed, the ambient temperature and the compressor inlet temperature, the thrust and the exhaust gas temperature, the compressor inlet and outlet temperature and the core shaft rotating speed with the compressor outlet temperature. These are shown as the blue squares in the following table.

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/5-table1.png)



he fact that some variables have a high correlation index between them, indicate us that the inclusion of either of those variables might have similar impact on our model.

Along with the previous table, we can also identify those variables that influence the most in the damage that the engines present through their R-squared value:

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/6-table2.png)



With the latter information, we can begin to eliminate some variables that generate noise but that are not really relevant to achieve realistic and useful conclusions. With both tables presented in this section, we can confirm the preliminary assessment from the scatter plots: information about vibrations, oil temperature and oil pressure do not influence significantly on the resulting damage of an engine.

6. Data Classification and Regression Models

Our first approach to multivariate regression model was using the variables selected from the correlation matrix results. Thus, we obtained a regression model with an R-squared of 0.603; we can tell that the model is relatively useful to predict damage but still has room for error.

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/7-model1.png)



We decided to go for a different approach to increase the R-squared value for our model by including the customers as a dummy variable. Our boxplots showed the significant variation that each customer presents for the different variables we are studying. We can say that each customer flies under different ambient conditions, whether that might be the type of environment (city, jungle, desert) and weather or the flight length and the care and maintenance they pay for the engines. As we included the five customers as dummy variables, we increased the R-squared for our prediction model and are able to have better conclusions over unknown sets of data with our multivariate model.

 ![](https://raw.githubusercontent.com/gilbertosg/ge-itesm-cloudComputing/master/images/8-model2.png)



After testing this model, which considers the different customers, with the original cleansed data set, we were able to predict 100% of the times if an engine (through their IDs and mean values) was either going to fail or not. Even though the model is not quite accurate to predict the amount of damage, the most important thing is that it is able to determine whether an engine fails; this will allow the company to prepare down time for maintenance of their engines.

7. Final Comments and Conclusions

Our multivariate regression model demonstrated to be trustworthy as it was capable of predicting 100% of failure cases from the initial data set we studied. It is of outmost importance to completely understand the context in which the data is generated, collected and studied in order to take adequate decisions regarding data cleansing, processing and conclusions.

Using our multivariate regression model, our predictions for the additional data set are that eleven out of the hundred engines studied failed under the presented conditions.

For more information you can go visit:

[http://cloud.csf.itesm.mx/~ge-1018723/final/](http://cloud.csf.itesm.mx/~ge-1018723/final/)

And download the zip file with the source code, datasets and full report.


