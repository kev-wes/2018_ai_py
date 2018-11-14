# coding: utf-8

# In[3]:


import os, csv, datetime
import pandas
import numpy as np

path = os.path.join(os.path.dirname(os.path.realpath(
    'C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email')))
# Path of the source file (which is the dest file from the hashing function
src_file = os.path.join(
    'C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email',
    'email train may.csv')

df = pandas.read_csv(src_file, delimiter=';')

# for i in df['created']:
# print(datetime.datetime.fromtimestamp(int(i)))
# print(i)
df['created_d'] = pandas.to_datetime(df['created'], unit="ms")
df['created_day'] = (df['created_d'].dt.day)
df['created_month'] = (df['created_d'].dt.month)
df['created_year'] = (df['created_d'].dt.year)
df['created_weekday'] = (df['created_d'].dt.weekday)

df['updated_d'] = pandas.to_datetime(df['updated'], unit="ms")
df['updated_day'] = (df['updated_d'].dt.day)
df['updated_month'] = (df['updated_d'].dt.month)
df['updated_year'] = (df['updated_d'].dt.year)
df['updated_weekday'] = (df['updated_d'].dt.weekday)

df['lastStatusTransistion_d'] = pandas.to_datetime(df['lastStatusTransistion'], unit="ms")
df['lStT_day'] = (df['lastStatusTransistion_d'].dt.day)
df['lStT_month'] = (df['lastStatusTransistion_d'].dt.month)
df['lStT_year'] = (df['lastStatusTransistion_d'].dt.year)
df['lStT_weekday'] = (df['lastStatusTransistion_d'].dt.weekday)

dest_path = os.path.join(
    'C:\\Users\\User\\Desktop\\Studium\\sciebo\\Master\\SoSe18\\Projektseminar\\Classifier\\Project\\Email',
    'email train may date.csv')
# df.to_csv(dest_path, sep = ';')

# print(df[df['assignedGroup'].isnull()])
# df['created_day'] =  [datetime.datetime.fromtimestamp(i).day for i in df['created']]
# print([int(i) for i in df['assignedGroup']])
# print(df.head())


# In[65]:


import pickle

filename = 'tfidf_vectorizer.sav'
tdidf_vect = pickle.load(open(filename, 'rb'))
# The FeatureHash.pys are saved
filename = 'tfidf_train_features.sav'
tdidf_feat = pickle.load(open(filename, 'rb'))

# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_extractor(text, ngram_range):
    # print(text)
    vectorizer = TfidfVectorizer(norm='l2',
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(text)
    return vectorizer, features


# df.columns.values


# In[12]:


for i in df.columns.values:
    print(i, len(set(df[i])))

# In[172]:


print(set(df['reporterUsername']))

# In[4]:


# please uncomment if running this the first time
"""
one_hotc = pandas.get_dummies(df['created_weekday'])
one_hotc.columns = ["c_Monday", "c_Tuesday", "c_Wednesday", "c_Thursday", "c_Friday", "c_ Saturday", "c_Sunday"]
df = df.join(one_hotc)

one_hotu = pandas.get_dummies(df['updated_weekday'])
one_hotu.columns = ["u_Monday", "u_Tuesday", "u_Wednesday", "u_Thursday", "u_Friday", "u_ Saturday", "u_Sunday"]
df = df.join(one_hotu)

one_hotl = pandas.get_dummies(df['lStT_weekday'])
one_hotl.columns = ["l_Monday", "l_Tuesday", "l_Wednesday", "l_Thursday", "l_Friday", "l_ Saturday", "l_Sunday"]
df = df.join(one_hotl)

#df.drop(columns = ["created", "updated", "lastStatusTransistion", "loadId"])
#df.to_csv(dest_path, sep = ';')
"""
df2 = df
# print(df2.shape)
one_hot_asu = pandas.get_dummies(df['assigneeUsername'], dummy_na=True)
df2 = pandas.concat([df2, one_hot_asu], axis=1)

print('df', df2.shape)
# print('asu',one_hot_asu)

one_hot_asg = pandas.get_dummies(df['assignedGroup'], dummy_na=True)
df2 = pandas.concat([df2, one_hot_asg], axis=1)
print('df', df2.shape)

# print(one_hot_aspr.shape)
one_hot_aspr = pandas.get_dummies(df['priorityName'], dummy_na=True)
df2 = pandas.concat([df2, one_hot_aspr], axis=1)
# print(one_hot_aspr.shape)
# print(df2.shape)

one_hot_aspk = pandas.get_dummies(df['projectKey'], dummy_na=True)
df2 = pandas.concat([df2, one_hot_aspk], axis=1)

# print(df2.shape)
one_hot_asrU = pandas.get_dummies(df['reporterUsername'], dummy_na=True)
df2 = pandas.concat([df2, one_hot_asrU], axis=1)
# print(one_hot_asrU.shape)
# print(df2.shape)

one_hot_asrP = pandas.get_dummies(df['responsibleParty'], dummy_na=True)
df2 = pandas.concat([df2, one_hot_asrP], axis=1)

# one_hot_asu.columns = set(df['assigneeUsername'])

# df.column.values
"""
#list(str(i) for i in df['assigneeUsername'])
"""
df2.to_csv(dest_path, sep=';')

# print(df2)


# In[15]:


df3 = df2.replace(to_replace=np.nan, value="-1")

# In[16]:


##Only for non-empty values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df3['txt_main'], df3['issueTypeCode'], test_size=0.33,
                                                    random_state=42)
vector, td_feat = tfidf_extractor(X_train, (1, 1))
print(td_feat.todense().shape)

# In[199]:


# df2[np.isnan(
# df2['txt_main'][~pandas.isnull(df2['txt_main'])]


# In[ ]:


# from scipy.sparse import csr_matrix
# print(tdidf_feat)
# tdidf_feat.todense().shape
# tdidf_feat.shape

