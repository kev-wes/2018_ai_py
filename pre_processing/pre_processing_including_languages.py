
# coding: utf-8

# In[80]:


# -*- coding: utf-8 -*-
# Needed project interpreters:
# keras
# tensorflow
import re
from keras.preprocessing.text import hashing_trick
import csv, os
import pynlpir
from langdetect import detect
# Needed project interpreters:
# keras
# tensorflow
from keras.preprocessing.text import hashing_trick
import spacy
from spacy.tokenizer import Tokenizer
import csv, os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentAn2 = SentimentIntensityAnalyzer()
import pandas, json
from pandas.io.json import json_normalize
import hashlib
from nltk.stem import PorterStemmer
from nltk.stem.snowball import GermanStemmer
from nltk.stem.snowball import FrenchStemmer
def token(column):
    list_of_words = [i.lower() for i in wordpunct_tokenize(column) if i.lower()] #not in stop]
    return list_of_words

contractions = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have",
"couldnt": "could not",
"couldn": "could not",
"wasnt": "was not",
"wasn": "was not",
"isnt": "is not",
"isn": "is not",
"doesnt": "does not",
"doesn": "does not",
"arent": "are not",
"aren": "are not",
"shouldnt": "should not",
"shouldn": "should not",
"wont": "could not",
"mightnt": "might not",
"mightn": "might not",
"aint": "should not"
}


# In[ ]:


def cleanPart(col):
    c = list(wordpunct_tokenize(col))
    for key, value in contractions.items():
        for x in c:
            if key == x:
                # print(c)
                index = c.index(x)
                c.remove(c[index])
                c.extend(wordpunct_tokenize(contractions[key]))
                # c = c.replace(x, contractions[key])
    # print(c)
    for i in c:
        for j in i:
            if j.lower() in stop2:
                i.replace(j, '')


    # list1 = [i for i in wordpunct_tokenize(col) if i not in stop]
    # list2 = [token(i) for i in col if i not in stop]
    # list2 = [i for i in wordpunct_tokenize(col.lower()) if i not in stop]
    port = PorterStemmer(mode='NLTK_EXTENSIONS')
    list2 = [port.stem(i) for i in c if i.lower() not in stop2 or i not in stop2]
    sentPol = sentAn2.polarity_scores(col)
    #sent = sentiment(list2)

    # print("s2", sentiment(col))
    P = nltk.pos_tag(list2)
    POS = [i[1] for i in P]
    #print(col, POS, sent, list2)
    return col, POS, list2, sentPol['neg'], sentPol['neu'], sentPol['pos'], sentPol['compound']
    # print(list2)
    # print([wordpunct_tokenize(col.lower())])


# In[176]:


def cleanL(c1, c2):
    lang = 'other'
    cL = [c1, c2]
    #cL = len([c1,c2])
    pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body = "", c1,"", c2, 0,0,0,0,0,0,0,0
    #print("Test", c2, detect(c2))
    #print(c2)
    if(detect(c2) == 'en'):
        print('EN')
        lang = 'en'
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj = cleanPart(c1)
            col_body, pos_body, list2_body,neg_body, neu_body, posi_body,compound_body = cleanPart(c2)
            #print('List2', list2_body, list2_subj)
        else:
            col_body, pos_body, list2_body,neg_body, neu_body, posi_body,compound_body = cleanPart(c2)
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = "", "", "", 0,0,0,0
    elif(detect(c2) == 'fr'):
        lang = 'fr'
        print('FR')
        if(len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = cleanFr(c1)
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body,compound_body = cleanFr(c2)
        else:
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = cleanFr(c2)
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body,compound_body = "", "", "", 0,0,0,0
    elif(detect(c2) == 'de'):
        lang = 'de'
        print('DE')
        if(len(cL) > 1):
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = cleanDe(c1)
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body,compound_body = cleanDe(c2)
        else:
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = cleanDe(c2)
            col_body, pos_body, list2_body, neg_body, neu_body, posi_body,compound_body = "", "", "", 0,0,0,0
    elif (detect(c2) == 'zh-cn'):
        lang = 'chn'
        print("CHN")
        col_subj, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj = cleanPart(c1)
        col_body, pos_body, list2_body, neg_body, neu_body, posi_body,compound_body = cleanCHN(c2)
        
        #return pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body
    else: 
        lang = "Other"
        print("Other", c2)
        if (len(cL) > 1):
            col_subj, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj = cleanPart(c1)
            col_body, pos_body, list2_body,neg_body, neu_body, posi_body,compound_body = cleanPart(c2)
            #print('List2', list2_body, list2_subj)
        else:
            col_body, pos_body, list2_body,neg_body, neu_body, posi_body,compound_body = cleanPart(c2)
            col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = "", "", "", 0,0,0,0
    return pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body, lang


# In[177]:


stop = set(stopwords.words('english'))
# print(len(stop))
strsplit = [i.split('\t')[0] for i in string.punctuation]
stop.remove('not')
stop.remove('same')
stop.update(
    ['.,', '.?', 'œ', 'donâ', '¤', 'Â', 'â', '€', '.', ',', '"', "'", '°', '?', '!', '^', '“', '°', '–', '\n', '\r',
     ':', ';', '`', '´', '(', ')', '[', ']', '{', '}', '-', '--', '----', '/', "'\'",
     '...'] + strsplit)  # remove it if you need punctuation

stopGen = set(
    ['.,', '.?', 'œ', 'donâ', '¤', 'Â', 'â', '€', '.', ',', '"', "'", '°', '?', '!', '^', '“', '°', '–', '\n', '\r',
     ':', ';', '`', '´', '(', ')', '[', ']', '{', '}',  '/', "'\'", 'dear', 'best'
     '...'] + strsplit)
#stopGen.update(['mit', 'freundlichen', 'grüßen', 'grüssen'])
#stopGen.update(['best', 'regards', 'dear'])

stop2 = [i.lower() for i in stop]
# print(stop2)
listR = []

def cleanFr(col):
    stopFr = stopwords.words('french')
    nlp = spacy.load('fr_core_news_sm')
    tokenPOS = nlp(col)
    pos = []
    tokens = []
    for t in tokenPOS:
        pos.append(t.pos_)
        if t.text not in stopFr:
            tokens.append(t.text)
    frStem = FrenchStemmer()
    list2 = [frStem.stem(i) for i in tokens if i.lower() not in stopFr]
    return col, pos, list2, 0,0,0,0
    #french POS tagging
def cleanDe(col):
    stopDe = stopwords.words('german')
    nlp = spacy.load('de_core_news_sm')
    tokenPOS = nlp(col)
    pos = []
    tokens = []
    for t in tokenPOS:
        pos.append(t.pos_)
        if t.text not in stopDe:
            tokens.append(t.text)
    geStem = GermanStemmer()
    list2 = [geStem.stem(i) for i in tokens if i.lower() not in stopDe]
    
    return col, pos, list2, 0,0,0,0
    #german POS tagging
def cleanCHN(col):
    stopwordsCHN = [line.rstrip() for line in open(os.path.join(path, 'Chinese\\chinese_stopwords.txt'),"r", encoding="utf-8")]
    stopwordsCHN = stopwordsCHN+[" ", "  "]
    #print(stopwordsCHN)
    pynlpir.open()
    tokenPOS = pynlpir.segment(col)
    list_body = [i[0] for i in tokenPOS if i[0] not in stopwordsCHN]
    #print(list_body)
    pos_body = [i[1] for i in tokenPOS if i[0] not in stopwordsCHN]
    return col, pos_body, list_body, 0,0,0,0
    


# In[186]:


def clean(col):
    #print(col.lower())
    #c = col.replace('  ', ' ')
    fromEmail = re.search("mailto:"+r'[\w\.-]+@[\w\.-]+', str(col).lower())
    if fromEmail:
        fromEmail = str(fromEmail.group(0)).lower()
    else:
        fromEmail = ""
        
    ccEmail = re.search("cc: "+r'[\w\.-]+@[\w\.-]+', str(col).lower())
    #print("cc", ccEmail)
    if ccEmail:
        ccEmail = str(ccEmail.group(0)).lower()
    else:
        ccEmail = ""
    toEmail = re.search("to: "+ r'([\w\.-]*)'+"; "+r'[\w\.-]+@[\w\.-]+', str(col).lower())
    if toEmail:
        toEmail = str(toEmail.group(0)).lower()
        ES = toEmail.split(" ")
        toEmail = ES[len(ES)-1]
    else:
        toEmail = ""
    
    #get TO number --> TO+' '+ 8 digiit number  --> find regexpression
    colR = col.lower().split('-----original message-----')
    sign = ""
    if(len(colR)>1):
        #print(col.replace('\n', ''))
        col = colR[0]
        response = 1
        response_txt = colR[1]
        #get sender email --> to:...@ --> find regexpression -->  use set & dict combination
        #print(col)
    else:
        response = 0
        response_txt = ""
    c = wordpunct_tokenize(col)
    c = [i for i in c if i.lower() not in list(stopGen)]
    col = " ".join(c)
    colS = col.lower().split('regards')
    if(len(colS) > 1):
        sign = col.lower().split('regards')[1]
        col = str(col.lower().split('regards')[0])
        #print("BEST REGARDS", col)
    else:
        colS = col.lower().split('mit freundlichen grüßen')
        if(len(colS)>1):
            sign = col.lower().split('mit freundlichen grüßen')[1]
            col = col.lower().split('mit freundlichen grüßen')[0]
        else:
            colS = col.lower().split('mit freundlichen grüssen')
            if(len(colS)>1):
                sign = col.lower().split('mit freundlichen grüssen')[1]
                col = col.lower().split('mit freundlichen grüßen')[0]
            else:
                colS = col.lower().split('cordialement')
                if(len(colS)>1):
                    sign = col.lower().split('cordialement')[1]
                    col = col.lower().split('cordialement')[0]
    #col = colS
    #subj = str(col).split('-----')[0]
    #if subj: get to from subject
        
    m = re.search("to "+r"\b[\d]{8}\b", str(col).lower())
    if(m is None):
        m = re.search("to"+r"\b[\d]{8}\b", str(col).lower())
    #print(col, m)
    if m:
        #print('TO',response, m.group(0))
        m = str(m.group(0)).lower().replace('to', '').replace(' ', '')
    c2 = str(col).split('----')
    #print('C0', c)2[0])
    #print('C2', type(c2), len(c2), c2)
    if(type(c2) is list and len(c2) > 1 and c2[1] is not ''):
        pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body, lang = cleanL(c2[0], c2[1])
    elif(type(c2) is list and len(c2) > 1 and c2[1] is ''):
        pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body, lang = cleanL("", c2[0])
    elif(type(c2) is list and len(c2) == 1):
        pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body, lang = cleanL("", c2[0])

    elif(type(c2) is str):
        pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body, lang = cleanL("", c2)
    return pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body, lang, response, response_txt, sign, m, fromEmail,ccEmail,toEmail

    print('---------')


# In[189]:


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    # Path of the source and destination file
    raw_file = os.path.join(path,'email_lear_start1499000000000end1502000000000.json')
    src_file = os.path.join(path,'email_src.csv')
    dst_file = os.path.join(path,'email_train_lear_start1499000000000end1502000000000_new.csv')
    
    with open(raw_file, 'r', encoding='UTF8') as f:
        data = json.load(f)
    df = pandas.read_json(raw_file, encoding='utf-8')
    #print(df['description'])
    #df.to_csv(src_file, sep=';', quoting=csv.QUOTE_ALL)
    df2 = json_normalize(data=data, record_path='slas')
    #df3 = pandas.concat([df, dhash_only_columnsf2], axis=1)
    # change file path
    df.to_csv(src_file, sep=';', quoting=csv.QUOTE_ALL, encoding = 'utf-8')
    # fill all the cols here that have to be anonymized

    mail_body_column = {'description': {}}
    hash_only_columns = {'assignedGroup': {},
                         'assigneeUsername': {},
                         'projectKey': {},
                         'reporterUsername': {},
                         'responsibleCompanyKey': {},
                         'responsibleCompanyKeyManual': {},
                         'responsibleParty': {}}
    # Insert salt value here, please keep salt secure and confidential

    salt = "insert_salt_her23451e"
    with open(src_file, 'r', encoding='utf-8') as csvfile:
        # The csv file gets assigned to reader
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        # Open a new csv file in which we will write the update content
        with open(dst_file, 'w', encoding = "utf-8") as destination:
            # reset reader to first line
            csvfile.seek(0)
            # Define output columns
            reader.fieldnames = "", "actionCodes", "assignedGroup","assigneeUsername","countComments",                                 "countIncomingMails","countOutgoingMails","created","creationType","description",                                 "issueId","issueKey","issueTypeCode","issueTypeId","issueTypeName",                                 "lastStatusTransistion","loadId","loggedWork","priorityName","projectKey",                                 "reporterUsername","responsibleCompanyKey","responsibleCompanyKeyManual",                                 "responsibleParty","slas","statusName","transportOrderId","updated",                                "slaGoal","slaMet","slaName","slaStatus","slaTimeLeft", "txt_subj",                                 "txt_main", "language", "pos_main", "pos_subj", "neg_subj", "neu_subj", "posi_subj", "compound_subj",                                 "neg_body", "neu_body", "posi_body", "compound_body", "ID", "lang", "response", "response_txt", "sign", "to", "fromEmail", "ccEmail", "toEmail"  
        
            writer = csv.DictWriter(destination, delimiter=';', fieldnames=reader.fieldnames, lineterminator='\n', quotechar='"', quoting=csv.QUOTE_ALL)
            # Write fieldnames of reader to header of new csv
            writer.writeheader()
            print("Successfully finished row 1 (header)")
            # Skip header line and read second line of source csv
            next(reader)
            # Index points to the current row of the iterator
            index=2
            # Loop over every row from line 2 to end of file
            for row in reader:
                temp = row
                if(index < 920):
                    print(index)
                    index = index + 1
                    continue
                #print('desc',temp['description'], len(temp['description']))
                if(temp['description'] is ''):
                    continue
               
                #print(temp['description'])
                # Main routine for the pre-processing of the mail body
                # Generates
                try:
                    # Identify language
                    #temp["language"] = 
                    #print(temp)
                    
                    #[detect(i) for i in temp["description"]]
                    # temp stores the text of column description for the current row in a map
                    # Example: {'Created': '30.Mai.2018', 'Is Cleaned': 'X',
                    # 'T': 'O_CR20 - Carrier fails to execute transport',
                    # 'Description': 'TO 10237544 \n----\nHello 4flow, DHL team,\n\ncould you please... \n',
                    # 'Transport start [Date]': '', 'Transport end [Date]': '', 'Transport Order ID': '10237544',
                    # 'Transportation mode': ''}
                    # temp[col] is the mapped value of a specific column (e.g. a string)
                    #print("Clean",temp['description'])                   

                    pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body, compound_body, lang, response, response_txt, sign, to, fromEmail, ccEmail, toEmail = clean(temp["description"])
                    #print("Clean",clean(temp["description"]))
                    temp['to'] = to
                    temp['lang'] = lang
                    temp["posi_subj"] = posi_subj
                    temp["neu_subj"] = neu_subj
                    temp["neg_subj"] = neg_subj
                    temp["compound_subj"] = compound_subj
                    temp["posi_body"] = posi_body
                    temp["neu_body"] = neu_body
                    temp["neg_body"] = neg_body
                    temp["compound_body"] = compound_body
                    temp["response"] = response
                    
                    #temp["response_txt"] = response_txt
                except:
                    # Error handling
                    print("Temp Error in row " + str(index))
                    continue
                try:
                    #print(temp['language']," ".join(list2_body))
                    # Hash subject POS
                    pos_subj_hash = ""
                    for i in pos_subj:
                        pos_subj_hash+= hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["pos_subj"] = hashing_trick(pos_subj_hash, 1000000, hash_function='md5')
                    # Hash Mail subject
                    txt_subj_hash = ""
                    #temp["txt_subj"] = list2_subj
                    if(type(list2_subj) is str):
                        list2_subj = wordpunct_tokenize(list2_subj)
                    for i in  list2_subj:
                        txt_subj_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                        #print(i)
                    temp["txt_subj"] = hashing_trick(txt_subj_hash, 1000000, hash_function='md5')
                    #temp["txt_subj"] = list2_subj
                    # Hash main POS
                    pos_main_hash = ""
                    for i in pos_body:
                        pos_main_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["pos_main"] = hashing_trick(pos_main_hash, 1000000, hash_function='md5')
                    # Hash main POS
                    txt_main_hash = ""
                    #temp["txt_main"] = list2_body
                    if(type(list2_body) is str):
                        list2_body = wordpunct_tokenize(list2_body)
                    for i in list2_body:
                        txt_main_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["txt_main"] = hashing_trick(txt_main_hash, 1000000, hash_function='md5')
                    #temp["txt_main"] = list2_body
                    txt_resp_hash = ""
                    
                    if(type(response_txt) is str):
                        response_txt = wordpunct_tokenize(response_txt)
                    for i in response_txt:
                        if i not in stopGen:
                            txt_resp_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["response_txt"] = hashing_trick(txt_resp_hash, 1000000, hash_function='md5')
                    #temp["response_txt"] = response_txt
                    sign_hash = ""
                    if(type(sign) is str):
                        sign = wordpunct_tokenize(sign)
                    for i in sign:
                        sign_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["sign"] = hashing_trick(sign_hash, 1000000, hash_function='md5')
                    temp['fromEmail'] = hashing_trick(hashlib.sha512((salt + fromEmail).encode('utf-8')).hexdigest() + " ",1000000, hash_function='md5')
                    temp['ccEmail'] = hashing_trick(hashlib.sha512((salt + ccEmail).encode('utf-8')).hexdigest() + " ",1000000, hash_function='md5')
                    temp['toEmail'] = hashing_trick(hashlib.sha512((salt + toEmail).encode('utf-8')).hexdigest() + " ",1000000, hash_function='md5')
                    #temp["sign"] = sign
                    #temp["pos_subj"] = [' '.join(map(str, hashing_trick(i, 1000000, hash_function='md5'))) for i in pos_subj]
                    #temp['txt_subj'] = [' '.join(map(str, hashing_trick(i, 1000000, hash_function='md5'))) for i in list2_subj]
                    #temp["pos_main"] = [' '.join(map(str, hashing_trick(i, 1000000, hash_function='md5'))) for i in pos_body]
                    #temp['txt_main'] = [' '.join(map(str, hashing_trick(i, 1000000, hash_function='md5'))) for i in list2_body]
                except:
                    # Error handling
                    print("Error in row " + str(index))
                    raise
                # For every row, iterate over columns defined in hash_only_columns
                # Columns not assigned in hash_only_columns will simply be transported
                # from source to destination w/o changing anything
                for col in hash_only_columns:
                    try:
                        col_hash = ""
                        for i in temp[col].split():
                            col_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                        temp[col] = hashing_trick(col_hash, 1000000, hash_function='md5')
                        #temp[col] = ' '.join(map(str, hashing_trick(temp[col], 1000000, hash_function='md5')))
                    except:
                        # Error handling
                        print("Error in row " + str(index), temp)
                        raise
                # remove the column description, this column has been transformed into
                # "sub_subj", "pol_subj", "pos_subj", "txt_subj", "sub_main", "pol_main","pos_main", "txt_main"
                
                # Assign a incremental ID to the email to later return the class and the ID
                temp['ID'] = str(index-1)
                temp['actionCodes'] = None
                temp['assigneeUsername'] = None
                temp['countComments'] = None
                temp['countIncomingMails'] = None
                temp['description'] = None
                temp['countOutgoingMails'] = None
                temp['lastStatusTransistion'] = None
                temp['loggedWork'] = None
                temp['priorityName'] = None
                temp['reporterUsername'] = None
                temp['responsibleCompanyKey'] = None
                temp['responsibleCompanyKeyManual'] = None
                temp['responsibleParty'] = None
                temp['statusName'] = None
                temp['transportOrderId'] = None                
                temp['updated'] = None
                temp['slaGoal'] = None
                temp['slaMet'] = None
                temp['slaName'] = None
                temp['slas'] = None
                temp['slaStatus'] = None
                temp['slaTimeLeft'] = None

                #print("Successfully finished row " + str(index))
                #print('break')
                index += 1
                "", "actionCodes", "assignedGroup","assigneeUsername","countComments",                 "countIncomingMails","countOutgoingMails","created","creationType","description",                 "issueId","issueKey","issueTypeCode","issueTypeId","issueTypeName",                 "lastStatusTransistion","loadId","loggedWork","priorityName","projectKey",                 "reporterUsername","responsibleCompanyKey","responsibleCompanyKeyManual",                 "responsibleParty","slas","statusName","transportOrderId","updated",                "slaGoal","slaMet","slaName","slaStatus","slaTimeLeft", "txt_subj",                 "txt_main", "language", "pos_main", "pos_subj", "neg_subj", "neu_subj", "posi_subj", "compound_subj",                 "neg_body", "neu_body", "posi_body", "compound_body", "ID", "lang", "response", "response_txt", "sign", "to"  

                # Row is written including the changes made
                #print(temp)
                writer.writerow(temp)

