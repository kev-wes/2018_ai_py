import os, csv, datetime
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn_pandas import DataFrameMapper
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
one_hot = OneHotEncoder(sparse=False)
label_encoder = LabelEncoder()
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Flatten
from keras.callbacks import ModelCheckpoint

def tfidf_extractor(text, ngram_range):

    vectorizer = TfidfVectorizer(norm='l2',
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(text)
    return features


path = os.path.join(os.path.dirname(os.path.realpath('C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email')))
    # Path of the source file (which is the dest file from the hashing function
src_file = os.path.join('C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email',
                        'email train may.csv')

df = pandas.read_csv(src_file, delimiter = ';')

#for i in df['created']:
    #print(datetime.datetime.fromtimestamp(int(i)))
    #print(i)
df['created_d'] = pandas.to_datetime(df['created'], unit = "ms")
df['created_day'] = (df['created_d'].dt.day)
df['created_month'] = (df['created_d'].dt.month)
df['created_year'] = (df['created_d'].dt.year)
df['created_weekday'] = (df['created_d'].dt.weekday)
df=df.drop(['updated', 'lastStatusTransistion'], axis = 1)
"""
df['updated_d'] = pandas.to_datetime(df['updated'], unit = "ms")
df['updated_day'] = (df['updated_d'].dt.day)
df['updated_month'] = (df['updated_d'].dt.month)
df['updated_year'] = (df['updated_d'].dt.year)
df['updated_weekday'] = (df['updated_d'].dt.weekday)

df['lastStatusTransistion_d'] = pandas.to_datetime(df['lastStatusTransistion'], unit = "ms")
df['lStT_day'] = (df['lastStatusTransistion_d'].dt.day)
df['lStT_month'] = (df['lastStatusTransistion_d'].dt.month)
df['lStT_year'] = (df['lastStatusTransistion_d'].dt.year)
df['lStT_weekday'] = (df['lastStatusTransistion_d'].dt.weekday)
"""
dest_path = os.path.join('C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email','email train may date.csv')
dest_path2 = os.path.join('C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email','results.csv')
dest_path3 = os.path.join('C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email','labels.csv')

#df.to_csv(dest_path, sep = ';')
###############################################Houng
df = df.replace(to_replace = np.nan,value = "-1")

"""
Model with rank: 1
Mean validation score: 0.531 (std: 0.008)
Parameters: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 3}


Model with rank: 1
Mean validation score: 0.530 (std: 0.005)
Parameters: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 1, 'min_samples_leaf': 1, 'min_samples_split': 10}

Model with rank: 1
Mean validation score: 0.440 (std: 0.003)
Parameters: {'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 10}

"""


txt_main_d = tfidf_extractor(df['txt_main'], (1,1))
txt_subj_d = tfidf_extractor(df['txt_subj'], (1,1))
pos_main_d = tfidf_extractor(df['pos_main'], (1,1))
pos_subj_d = tfidf_extractor(df['pos_subj'], (1,1))

one_hotc = pandas.get_dummies(df['created_weekday'])
one_hotc.columns = ["c_Monday", "c_Tuesday", "c_Wednesday", "c_Thursday", "c_Friday", "c_ Saturday", "c_Sunday"]
dfm = df.join(one_hotc)
"""
one_hotu = pandas.get_dummies(df['updated_weekday'])
one_hotu.columns = ["u_Monday", "u_Tuesday", "u_Wednesday", "u_Thursday", "u_Friday", "u_ Saturday", "u_Sunday"]
dfm = dfm.join(one_hotu)

one_hotl = pandas.get_dummies(df['lStT_weekday'])
one_hotl.columns = ["l_Monday", "l_Tuesday", "l_Wednesday", "l_Thursday", "l_Friday", "l_ Saturday", "l_Sunday"]
dfm = dfm.join(one_hotl)
"""
###############################################Houng
txt_comb_d = pandas.DataFrame(hstack([txt_main_d,txt_subj_d, pos_main_d, pos_subj_d]).todense())

dfm = dfm.drop(['txt_main', 'txt_subj','pos_main','pos_subj'], axis = 1)
#dfm = dfm.drop(['issueId', 'loadId', 'created','issueTypeName', 'issueKey','lastStatusTransistion','updated','transportOrderId', 'issueTypeId'],axis = 1)
#dfList = dfm[['assigneeUsername','priorityName','projectKey','reporterUsername', 'responsibleParty']]
#for i in dfList.columns.values:
#    dfList2[str(i)] = label_encoder.fit_transform(dfList[str(i)].astype(str))
#dfList2 = dfList
#dfList2['assigneeUsername'] = label_encoder.fit_transform(dfList['assigneeUsername'].astype(str))
#dfList2['priorityName'] = label_encoder.fit_transform(dfList['priorityName'].astype(str))
#dfList2['reporterUsername'] = label_encoder.fit_transform(dfList['reporterUsername'].astype(str))
#dfList2['responsibleParty'] = label_encoder.fit_transform(dfList['responsibleParty'].astype(str))

#dfListOneH = one_hot.fit_transform(dfList2)
#dfListC = one_hot.fit_transform(dfList)
#dfm = pandas.concat([dfm, txt_main_d], axis = 1)
#dfm = pandas.concat([dfm, txt_subj_d], axis = 1)
#dfm = pandas.concat([dfm, pos_main_d], axis = 1)
#dfm = pandas.concat([dfm, pos_subj_d], axis = 1)

#dfm.to_csv(dest_path, sep = ';')
#print(df[df['assignedGroup'].isnull()])
#df['created_day'] =  [datetime.datetime.fromtimestamp(i).day for i in df['created']]
#print([int(i) for i in df['assignedGroup']])
#print(df.head())

dfm = pandas.concat([dfm, txt_comb_d], axis = 1)


#please uncomment if running this the first time


#df.drop(columns = ["created", "updated", "lastStatusTransistion", "loadId"])
#df.to_csv(dest_path, sep = ';')
df2 = dfm

#df2.columns.values
#print(df2.shape)
"""
one_hot_asu = pandas.get_dummies(dfm['assigneeUsername'].astype(str), dummy_na = True, prefix = "a_")
dfm = pandas.concat([dfm, one_hot_asu], axis = 1)
dfm = dfm.drop(['assigneeUsername'], axis = 1)
"""
#print('asu',one_hot_asu)

one_hot_asg = pandas.get_dummies(dfm['assignedGroup'].astype(str), dummy_na = True, prefix = "b_")
dfm = pandas.concat([dfm,one_hot_asg], axis = 1)
dfm= dfm.drop(['assignedGroup'], axis = 1)


"""
one_hot_aspr = pandas.get_dummies(dfm['priorityName'].astype(str), dummy_na = True, prefix = "c_")
dfm = pandas.concat([dfm,one_hot_aspr], axis = 1)
dfm = dfm.drop(['priorityName'], axis = 1)"""


one_hot_aspk = pandas.get_dummies(dfm['projectKey'].astype(str), dummy_na = True, prefix = "d_")
dfm = pandas.concat([dfm,one_hot_aspk], axis = 1)
dfm = dfm.drop(['projectKey'], axis = 1)

"""
one_hot_asrU = pandas.get_dummies(dfm['reporterUsername'].astype(str), dummy_na = True, prefix = "e_")
dfm = pandas.concat([dfm,one_hot_asrU], axis = 1)  
dfm = dfm.drop(['reporterUsername'], axis = 1)


one_hot_asrP = pandas.get_dummies(dfm['responsibleParty'].astype(str), dummy_na = True, prefix = "f_")
dfm = pandas.concat([dfm,one_hot_asrP], axis = 1)  
dfm = dfm.drop(['responsibleParty'], axis = 1)"""


#one_hot_asrP = pandas.get_dummies(df['issueTypeCode'].astype(str), dummy_na = True, prefix = "f_")
#df2 = pandas.concat([df2,one_hot_asrP], axis = 1)
#df2 = df2.drop(['issueTypeCode'], axis = 1)

"""
one_hot_asrP = pandas.get_dummies(dfm['issueTypeId'].astype(str), dummy_na = True, prefix = "f_")
dfm = pandas.concat([dfm,one_hot_asrP], axis = 1)  
dfm = dfm.drop(['issueTypeId'], axis = 1)
"""


dateList = ["created_day","created_month","created_year","created_weekday"]#,"updated_day","updated_month", "updated_year", "updated_weekday", "lStT_day", "lStT_month", "lStT_year", "lStT_weekday"]
for i in dateList:
    one_hot_date = pandas.get_dummies(dfm[i].astype(str), dummy_na=True, prefix = (i+'_'))
    dfm = dfm.drop([i], axis = 1)
    dfm = pandas.concat([dfm, one_hot_date], axis = 1)

dfm = dfm.drop(['created_d'], axis = 1)#'updated_d', 'lastStatusTransistion_d'], axis = 1)

#issueTypeCode 	issueTypeId 	issueTypeName
#df2 = pandas.concat([df2,one_hot_asrP], axis = 1)

#one_hot_asu.columns = set(df['assigneeUsername'])

#df.column.values

#df2.to_csv(dest_path, sep = ';')

#df2.columns.values

dfm3 = dfm.drop(['assigneeUsername','issueTypeName', 'issueTypeId','issueKey', 'priorityName', 'responsibleParty', 'reporterUsername',  'loadId', 'issueId', 'created', 'transportOrderId'], axis = 1)
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

#sent_values = pandas.DataFrame(df5.loc[:,['neg_subj', 'neu_subj', 'posi_subj', 'compound_subj', 'neg_body',
#       'neu_body', 'posi_body', 'compound_body']])
#valueDF = pandas.concat((pandas.DataFrame(features), sent_values), axis = 1)
X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(dfm3.drop(['issueTypeCode'], axis = 1), dfm3['issueTypeCode'], dfm['ID'], test_size=0.33, random_state=42)
####################################################
#Kevin
####################################################

clf_rf = RandomForestClassifier(random_state=43)
clf_rf_5 = RandomForestClassifier(n_estimators=20, )
clr_rf_5 = clf_rf_5.fit(X_train,y_train)

print(clf_rf_5.score(X_test, y_test))