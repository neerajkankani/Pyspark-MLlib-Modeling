#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
 
print("username:" , os.getlogin())
from datetime import date

today = date.today()
print("Today's date:", today)


# In[2]:


import findspark
findspark.init()


# In[3]:


from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import * 
import pyspark.sql.functions as F
from pyspark.sql.functions import col, asc,desc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline 
from sklearn.metrics import confusion_matrix

spark=SparkSession.builder .master ("local[*]").appName("part3").getOrCreate()


# In[4]:


sc=spark.sparkContext
sqlContext=SQLContext(sc)


# In[5]:


import os
os.getcwd()


# In[6]:


from platform import python_version

print(python_version())


# In[7]:


sc.version #spark version


# ## Read File

# In[8]:


df=spark.read  .option("header","True") .option("inferSchema","True") .option("sep",";") .csv("C:/Users/pranj/OneDrive - Oklahoma A and M System/Desktop/BAN 5753/Mini Project 2/Data/XYZ_Bank_Deposit_Data_Classification.csv")
print("There are",df.count(),"rows",len(df.columns),
      "columns" ,"in the data.") 


# ## Show Sample Data

# In[9]:


df.show(4)


# ## Data Types of Columns

# In[10]:


df.printSchema()


# # EDA - Total Numeric and categorical variables, Null values for dataset , Cardinality of all variables, Feature Distribution, Correlations, Bivariate analysis of target Vs input variables, Univariate patterns

# In[11]:


# Replacing '.' in column names with '_'
cols = [col.replace('.', '_') for col in df.columns]

df = df.toDF(*cols)


# In[12]:


# Identifying Numeric and Categorical variables

string_columns = [item[0] for item in df.dtypes if item[1].startswith('string')]
int_columns = [item[0] for item in df.dtypes if item[1].startswith('int')]
double_columns = [item[0] for item in df.dtypes if item[1].startswith('double')]

string_columns,int_columns,double_columns

numeric_variables = int_columns + double_columns

print("Numeric Variables are: ",numeric_variables)

print("categorical Variables are: ",string_columns)


# In[13]:


# Checking cardinality of all variables(unique values)

unique_values=[(column_name, df.select(column_name).distinct().count()) for column_name in df.columns]
print(unique_values)


# In[14]:


# Data statistics

df.select(numeric_variables).describe().toPandas().transpose()


# In[15]:


# Checking for Null values in dataset

from pyspark.sql.functions import isnan,when,count,col
df.select([count(when(isnan(c),c)).alias(c) for c in df.columns]).toPandas().head()


# In[17]:


# Distribution of Features

from matplotlib import cm
plt.style.use('Solarize_Light2')

## Plot Size
figure=plt.figure(figsize=(25,15)) 

## Plot Title
st = figure.suptitle("Distribution of Numeric Features", fontsize=40,
                    verticalalignment='center')

for col,num in zip(numeric_variables, range(1,11)):
    ax= figure.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
    plt.grid(False)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
plt.tight_layout()
st.set_y(0.95)
figure.subplots_adjust(top=0.85,hspace=0.4)

plt.show()


# In[18]:


figure=plt.figure(figsize=(25,15))
plt.style.use('Solarize_Light2')


## Plot Title
st = figure.suptitle("Distribution of Categorical Features", fontsize=40,
                    verticalalignment='center')


for col,num in zip(string_columns, range(1,13)):
    ax= figure.add_subplot(4,3,num)
    sdf = df.toPandas()[col].value_counts().reset_index()
    ax.bar(sdf['index'], sdf[col])

    def addlabels(sdf=sdf):
        for i in range(sdf.shape[0]):
            plt.text(i, sdf.iloc[i][col]//2, sdf.iloc[i][col], ha = 'center', fontdict={"color":'black'})

    addlabels()

    plt.grid(False)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)

#plt.tight_layout()
st.set_y(0.95)
figure.subplots_adjust(top=0.85,hspace=1)


# In[19]:


# Target Variable Distribution

df.groupby("y").count().show()


# In[20]:


# Bivariate analysis of input variables

sns.pairplot(df.toPandas())


# In[22]:


df2=df


# In[23]:


string_columns


# In[24]:


# String Indexer

stringIndexer1 = StringIndexer().setInputCols(string_columns).setOutputCols([col + '_index' for col in string_columns])


index_model=stringIndexer1.fit(df2)
index_df=index_model.transform(df2)
index_df = index_df.drop(*string_columns)

index_df.toPandas().tail(10)


# In[25]:


# List of all columns with indexed columns
index_df.columns


# In[26]:


import pandas as pd
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_columns', 12)


# In[27]:


# Vector Assembler

vector_assembler =  VectorAssembler().setInputCols(index_df.columns)                                     .setOutputCol("vectorised_features")

# Skipping the missinge ones
assembler_df = vector_assembler.setHandleInvalid("skip").transform(index_df)
assembler_df.toPandas().head()


# In[28]:


# Variable Correlations

from pyspark.ml.stat import Correlation

col_names = index_df.columns
matrix = Correlation.corr(assembler_df, "vectorised_features").collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = col_names, index=col_names) 
corr_matrix_df


# In[29]:


# Correlation Matrix

plt.figure(figsize=(16,5))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True)


# In[31]:


# bivariate analysis of target versus input variable

sns.scatterplot(data=index_df.toPandas(), x="age", y="y_index")


# In[30]:


index_cols = [col for col in index_df.columns if 'index' in col]
index_cols.remove('y_index')


# In[32]:


# One hot encoding

encoder = OneHotEncoder()         .setInputCols (index_cols)         .setOutputCols ([col + '_encoded' for col in index_cols])

encoder_model = encoder.fit(index_df)
encoder_df = encoder_model.transform(index_df)

encoder_df.toPandas().head()


# In[33]:


encoder_df.drop(*index_cols)


# In[34]:


all_cols = encoder_df.columns
all_cols.remove('y_index')


# In[35]:


# Vector Assembler

vector_assembler =  VectorAssembler().setInputCols(all_cols)                                     .setOutputCol("vectorised_features")

# Skipping the missinge ones
assembler_df = vector_assembler.setHandleInvalid("skip").transform(encoder_df)
assembler_df.toPandas().head()


# In[36]:


# Standard Scaler

scaler = StandardScaler()         .setInputCol ("vectorised_features")         .setOutputCol ("features")
        
scaler_model=scaler.fit(assembler_df)
scaler_df=scaler_model.transform(assembler_df)
pd.set_option('display.max_colwidth', 40)
scaler_df.select("vectorised_features","features").toPandas().head(5)


# In[82]:


scaler_df


# # Train / Test Split

# In[37]:


train, test = scaler_df.randomSplit([0.8, 0.2], seed = 100)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))


# In[38]:


train.groupby("y_index").count().show()
test.groupby("y_index").count().show()


# # Model Training

# ## Logistic Regression

# In[39]:


from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'y_index', maxIter=5)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions.select('y_index', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[40]:


# Logistic Regression Confusion Matrix

class_names=[1.0,0.0]
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[41]:


y_true = predictions.select("y_index")
y_true = y_true.toPandas()

y_pred = predictions.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[42]:


# Logistic Regression Accuracy

accuracy = predictions.filter(predictions.y_index == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ",accuracy)


# In[43]:


# Logistic Regression ROC Curve

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))


# In[44]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
evaluator.setLabelCol('y_index')


# In[45]:


print('Test Area Under ROC', evaluator.evaluate(predictions))


# In[46]:


# Logistic Regression Cross Validation and Parameter Tuning

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])# regularization parameter
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])# Elastic Net Parameter (Ridge = 0)
             .addGrid(lr.maxIter, [1, 5, 10])#Number of iterations
             .build())

cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, 
                    evaluator=evaluator, numFolds=5)

cvModel = cv.fit(train)


# In[47]:


## Evaluate Best Model
predictions = cvModel.transform(test)
print('Best Model Test Area Under ROC', evaluator.evaluate(predictions))


# ## Decision Tree

# In[48]:


from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'y_index')
dtModel = dt.fit(train)
predictions_dt = dtModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_dt.select('y_index', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[49]:


# Decision Tree Confusion Matrix

y_true = predictions_dt.select("y_index")
y_true = y_true.toPandas()

y_pred = predictions_dt.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[50]:


# Decision Tree Accuracy

accuracy_dt = predictions_dt.filter(predictions_dt.y_index == predictions_dt.prediction).count() / float(predictions_dt.count())
print("Accuracy DT : ",accuracy_dt)


# In[51]:


print('Test Area Under ROC', evaluator.evaluate(predictions_dt))


# ## Random Forest

# In[52]:


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'y_index', numTrees=10)
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_rf.select('y_index', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[53]:


# Random Forest Confusion Matrix

y_true = predictions_rf.select("y_index")
y_true = y_true.toPandas()

y_pred = predictions_rf.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[54]:


## Random Forest Accuracy

accuracy_rf = predictions_rf.filter(predictions_rf.y_index == predictions_rf.prediction).count() / float(predictions_rf.count())
print("Accuracy RF : ",accuracy_rf)


# In[72]:


# Decision Tree ROC Curve

trainingSummary_rf = rfModel.summary
roc_rf = trainingSummary_rf.roc.toPandas()
plt.plot(roc_rf['FPR'],roc_rf['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary_rf.areaUnderROC))


# In[55]:


print('Test Area Under ROC', evaluator.evaluate(predictions_rf))


# In[81]:


rfModel.featureImportances


# ## Gradient boosted trees

# In[56]:


from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(featuresCol = 'features', labelCol = 'y_index', maxIter=10)
gbtModel = gbt.fit(train)
predictions_gbt = gbtModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_gbt.select('y_index', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[57]:


# Gradient boosted trees Confusion Matrix

y_true = predictions_gbt.select("y_index")
y_true = y_true.toPandas()

y_pred = predictions_gbt.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[58]:


## Gradient boosted trees Accuracy

accuracy_gbt = predictions_gbt.filter(predictions_gbt.y_index == predictions_gbt.prediction).count() / float(predictions_gbt.count())
print("Accuracy GBT : ",accuracy_gbt)


# In[59]:


print('Test Area Under ROC', evaluator.evaluate(predictions_gbt))


# In[80]:


gbtModel.featuresCol


# In[75]:


gbtModel.featureImportances


# ## Support Vector Classifier

# In[60]:


from pyspark.ml.classification import LinearSVC

lsvc = LinearSVC(featuresCol = 'features', labelCol = 'y_index', maxIter=10, regParam=0.1)
lsvcModel = lsvc.fit(train)
predictions_lsvc = lsvcModel.transform(test)
#predictions_train = lrModel.transform(train)
predictions_lsvc.select('y_index', 'features',  'rawPrediction', 'prediction').toPandas().head(5)


# In[61]:


# Support Vector Classifier Confusion Matrix

y_true = predictions_lsvc.select("y_index")
y_true = y_true.toPandas()

y_pred = predictions_lsvc.select("prediction")
y_pred = y_pred.toPandas()

cnf_matrix = confusion_matrix(y_true, y_pred,labels=class_names)
#cnf_matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[62]:


# Support Vector Classifier Accuracy

accuracy_lsvc = predictions_lsvc.filter(predictions_lsvc.y_index == predictions_lsvc.prediction).count() / float(predictions_lsvc.count())
print("Accuracy LinearSVC : ",accuracy_lsvc)


# In[63]:


print('Test Area Under ROC', evaluator.evaluate(predictions_lsvc))


# In[64]:


gbt.explainParams().split("\n")


# ## K-Means Clustering

# In[65]:


from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

 # Trains a k-means model.
kmeans = KMeans().setK(2).setSeed(1)
model_km = kmeans.fit(scaler_df.select('features'))

# Make predictions
predictions_km = model_km.transform(scaler_df)

predictions_km.toPandas().head(10)


# In[66]:


# Evaluate clustering by computing Silhouette score

evaluator_km = ClusteringEvaluator()

silhouette = evaluator_km.evaluate(predictions_km)
print("Silhouette with squared euclidean distance = " + str(silhouette))


# In[67]:


# Shows the result

centers = model_km.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)


# In[83]:


pipeline_stages=Pipeline()                .setStages([GBTClassifier(featuresCol = 'features', labelCol = 'y_index', maxIter=10)])
pipeline_model=pipeline_stages.fit(train)


# In[84]:


predictions_gbt_pipe = pipeline_model.transform(test)
#predictions_train = lrModel.transform(train)
predictions_gbt_pipe.select('y_index', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)


# In[ ]:


path = "saved_model_file"
pipeline_model.save(path)


# In[ ]:


from pyspark.ml.pipeline import PipelineModel
l = PipelineModel.load(path)

