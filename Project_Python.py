import os
import pandas as pd
import seaborn as sns
from random import randrange, uniform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from scipy.stats import chi2_contingency

os.chdir("C:/Users/Rajashekar/Videos/project/Churn")
os.getcwd()

# In[104]:
#reading train data set 
churn_train=pd.read_csv("Train_data.csv",sep=",")

# In[115]:
#numeric columns for correlation analysis
numeric_cnames=['account length', 'area code','number vmail messages',
       'total day minutes', 'total day calls', 'total day charge',
       'total eve minutes', 'total eve calls', 'total eve charge',
       'total night minutes', 'total night calls', 'total night charge',
       'total intl minutes', 'total intl calls', 'total intl charge',
       'number customer service calls']

# In[116]:
churn_corr=churn_train.loc[:,numeric_cnames]

# In[121]:
#below is correlation matrix
f, ax = plt.subplots(figsize=(7, 5))
#Generate correlation matrix
corr = churn_corr.corr()
#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[125]:
#chisquare test for categorical variables
cat_cnames=['state','phone number','international plan', 'voice mail plan','Churn']

# In[131]:

#chi-square analysis for categorical variables
for i in cat_cnames:
    #print(i)
    chi2,p,dof,ex=chi2_contingency(pd.crosstab(churn_train['Churn'],churn_train[i]))
    #print(p,chi2,dof)

# In[132]:
#removal cnames
removal_cnames= ['phone number','total day charge','total eve charge','total night charge',
                'total intl charge']

# In[133]:
#reading data for Decision Tree analysis
churn_train_dt=pd.read_csv("Train_data.csv",sep=",")
churn_test_dt=pd.read_csv("Test_data.csv",sep=",")
churn_train_final_dt=churn_train_dt.drop(removal_cnames,axis=1)
churn_test_final_dt=churn_test_dt.drop(removal_cnames,axis=1)

# In[140]:
#Import Libraries for decision tree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# In[141]:
#replace target categories with Yes or No
churn_train_final_dt['Churn'] = churn_train_final_dt['Churn'].replace(' False', 'No')
churn_train_final_dt['Churn'] = churn_train_final_dt['Churn'].replace(' True', 'Yes')
churn_test_final_dt['Churn'] = churn_test_final_dt['Churn'].replace( ' True', 'Yes')
churn_test_final_dt['Churn'] = churn_test_final_dt['Churn'].replace( ' False', 'No')

# In[142]:
#objects types are not supported by decision tree so converting object types (Training Dataset)decision tree required types
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in churn_train_final_dt.columns:
    if churn_train_final_dt[column_name].dtype == object:
        churn_train_final_dt[column_name] = le.fit_transform(churn_train_final_dt[column_name])
    else:
         pass
#objects types are not supported by decision tree so converting object types (Test Dataset)decision tree required types        
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in churn_test_final_dt.columns:
    if churn_test_final_dt[column_name].dtype == object:
        churn_test_final_dt[column_name] = le.fit_transform(churn_test_final_dt[column_name])
    else:
         pass        


# In[156]:
#Applying DT regression
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(churn_train_final_dt.iloc[:,0:15],churn_train_final_dt.iloc[:,15])

# In[152]:
#predict new test cases
C50_Predictions = C50_model.predict(churn_test_final_dt.iloc[:,0:15])
#Create dot file to visualise tree  #http://webgraphviz.com/
dotfile = open("pt.dot", 'w')
df = tree.export_graphviz(C50_model, out_file=dotfile, feature_names =churn_train_final_dt.iloc[:,0:15].columns)

# In[174]:
#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(churn_test_final_dt.iloc[:,15], C50_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
print(CM)
#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Results
#Accuracy: 91.72165566886623
#FNR: 31.25


# In[160]:
#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 275).fit(churn_train_final_dt.iloc[:,0:15],churn_train_final_dt.iloc[:,15])

# In[161]:
RF_Predictions = RF_model.predict(churn_test_final_dt.iloc[:,0:15])
# In[162]:
#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(churn_test_final_dt.iloc[:,15], RF_Predictions)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#print(CM)
#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)
#False Negative rate 
(FN*100)/(FN+TP)
#Results
#Accuracy:95.62087582483504
#FNR: 29.910714285714285

# In[163]:
#Create logistic data. Save target variable first
churn_train_dt=pd.read_csv("Train_data.csv",sep=",")
churn_test_dt=pd.read_csv("Test_data.csv",sep=",")
churn_train_logit = pd.DataFrame(churn_train_dt['Churn'])
churn_test_logit = pd.DataFrame(churn_test_dt['Churn'])

# In[164]:
cnames=['account length', 'area code', 'number vmail messages',
       'total day minutes', 'total day calls', 
       'total eve minutes', 'total eve calls', 
       'total night minutes', 'total night calls', 
       'total intl minutes', 'total intl calls', 
       'number customer service calls']
# In[165]:
#Add continous variables
churn_train_logit = churn_train_logit.join(churn_train_dt[cnames])
churn_test_logit = churn_test_logit.join(churn_test_dt[cnames])
# In[166]:
##Create dummies for categorical variables
cat_names = ['state','international plan','voice mail plan',]
for i in cat_names:
    temp = pd.get_dummies(churn_train_dt[i], prefix = i)
    churn_train_logit = churn_train_logit.join(temp)
for i in cat_names:
    temp = pd.get_dummies(churn_test_dt[i], prefix = i)
    churn_test_logit = churn_test_logit.join(temp)    

# In[167]:
#objects types are not supported by decision tree so converting object types (Training Dataset)decision tree required types
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in churn_train_logit.columns:
    if churn_train_logit[column_name].dtype == object:
        churn_train_logit[column_name] = le.fit_transform(churn_train_logit[column_name])
    else:
         pass
#objects types are not supported by decision tree so converting object types (Test Dataset)decision tree required types        
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in churn_test_logit.columns:
    if churn_test_logit[column_name].dtype == object:
        churn_test_logit[column_name] = le.fit_transform(churn_test_logit[column_name])
    else:
         pass        

# In[169]:
#select column indexes for independent variables
train_cols = churn_train_logit.columns[1:68]
test_cols  = churn_test_logit.columns[1:68]

# In[49]:
#Built Logistic Regression
import statsmodels.api as sm
logit = sm.Logit(churn_train_logit['Churn'], churn_train_logit[train_cols]).fit()
logit.summary()

# In[50]:
#Predict test data
churn_test_logit['Actual_prob'] = logit.predict(churn_test_logit[test_cols])
churn_test_logit['Actual_Val'] = 1
churn_test_logit.loc[churn_test_logit.Actual_prob < 0.5, 'Actual_Val'] = 0

# In[53]:
#Build confusion matrix
CM = pd.crosstab(churn_test_logit['Churn'], churn_test_logit['Actual_Val'])
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#accuracy of model=86.98260347930415
#((TP+TN)*100)/(TP+TN+FP+FN)
#False Negative Score=75.44642857142857
#(FN*100)/(FN+TP)


# In[54]:
#reading data for KNN analysis
churn_train_dt=pd.read_csv("Train_data.csv",sep=",")
churn_test_dt=pd.read_csv("Test_data.csv",sep=",")
churn_train_final_dt=churn_train_dt.drop(removal_cnames,axis=1)
churn_test_final_dt=churn_test_dt.drop(removal_cnames,axis=1)

# In[55]:
#replace target categories with Yes or No
churn_train_final_dt['Churn'] = churn_train_final_dt['Churn'].replace(' False', 'No')
churn_train_final_dt['Churn'] = churn_train_final_dt['Churn'].replace(' True', 'Yes')
churn_test_final_dt['Churn'] = churn_test_final_dt['Churn'].replace( ' True', 'Yes')
churn_test_final_dt['Churn'] = churn_test_final_dt['Churn'].replace( ' False', 'No')

# In[56]:
#objects types are not supported by decision tree so converting object types (Training Dataset)decision tree required types
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in churn_train_final_dt.columns:
    if churn_train_final_dt[column_name].dtype == object:
        churn_train_final_dt[column_name] = le.fit_transform(churn_train_final_dt[column_name])
    else:
         pass
#objects types are not supported by decision tree so converting object types (Test Dataset)decision tree required types        
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column_name in churn_test_final_dt.columns:
    if churn_test_final_dt[column_name].dtype == object:
        churn_test_final_dt[column_name] = le.fit_transform(churn_test_final_dt[column_name])
    else:
         pass        

# In[86]:
#KNN implementation
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 1).fit(churn_train_final_dt.iloc[:,0:15],churn_train_final_dt.iloc[:,15])
# In[87]:
#predict test cases
KNN_Predictions = KNN_model.predict(churn_test_final_dt.iloc[:,0:15])

# In[88]:
#build confusion matrix
CM = pd.crosstab(churn_test_final_dt.iloc[:,15], KNN_Predictions)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]
#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)
#False Negative rate 
#(FN*100)/(FN+TP)
#Accuracy: 82.30353929214156
#FNR: 63.392857142857146

# In[439]:
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Naive Bayes implementation
NB_model = GaussianNB().fit(churn_train_final_dt.iloc[:,0:15],churn_train_final_dt.iloc[:,15])
# In[440]:
#predict test cases
NB_Predictions = NB_model.predict(churn_test_final_dt.iloc[:,0:15],)
# In[170]:
#Build confusion matrix
CM = pd.crosstab(churn_test_final_dt.iloc[:,15], NB_Predictions)
#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
#((TP+TN)*100)/(TP+TN+FP+FN)

#False Negative rate 
#(FN*100)/(FN+TP)

#Accuracy: 85.84283143371326
#FNR: 60.267857142857146


