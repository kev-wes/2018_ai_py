# Needed project interpreters:
# keras
# tensorflow
from keras.preprocessing.text import hashing_trick
import csv, os
from langdetect import detect
# Needed project interpreters:
# keras
# tensorflow
from keras.preprocessing.text import hashing_trick
import csv, os
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentAn2 = SentimentIntensityAnalyzer()
import pandas, json
from pandas.io.json import json_normalize
import hashlib



def token(column):
    list_of_words = [i.lower() for i in wordpunct_tokenize(column) if i.lower() not in stop]
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
    lemmatizer = WordNetLemmatizer()
    list2 = [lemmatizer.lemmatize(i) for i in c if i.lower() not in stop2 or i not in stop2]
    sentPol = sentAn2.polarity_scores(col)
    #sent = sentiment(list2)

    # print("s2", sentiment(col))
    P = nltk.pos_tag(list2)
    POS = [i[1] for i in P]
    #print(col, POS, sent, list2)
    return col, POS, list2, sentPol['neg'], sentPol['neu'], sentPol['pos'], sentPol['compound']
    # print(list2)
    # print([wordpunct_tokenize(col.lower())])


stop = set(stopwords.words('english'))
# print(len(stop))
strsplit = [i.split('\t')[0] for i in string.punctuation]
stop.remove('not')
stop.remove('same')
stop.update(['Mit', 'freundlichen', 'Grüßen', 'Grüssen'])
stop.update(
    ['.,', '.?', 'œ', 'donâ', '¤', 'Â', 'â', '€', '.', ',', '"', "'", '°', '?', '!', '^', '“', '°', '–', '\n', '\r',
     ':', ';', '`', '´', '(', ')', '[', ']', '{', '}', '-', '--', '----', '/', "'\'",
     '...'] + strsplit)  # remove it if you need punctuation
stop2 = [i.lower() for i in stop]
# print(stop2)
listR = []


def clean(col):
    c2 = col.split('----')
    if (len(c2) > 1):
        col_subj, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj = cleanPart(c2[0])
        col_body, pos_body, list2_body,neg_body, neu_body, posi_body,compound_body = cleanPart(c2[1])
        return pos_body, list2_body, pos_subj, list2_subj,neg_subj, neu_subj, posi_subj,compound_subj, neg_body, neu_body, posi_body,compound_body
    else:
        col_body, pos_body, list2_body,neg_body, neu_body, posi_body,compound_body = cleanPart(c2[0])
        col_subj, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj,compound_subj = "", "", "", 0,0,0,0
        return pos_body, list2_body, pos_subj, list2_subj, neg_body, neu_body, posi_body,compound_body,neg_subj, neu_subj, posi_subj,compound_subj
    print('---------')


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    # Path of the source and destination file
    raw_file = os.path.join(path,'email_raw.json')
    src_file = os.path.join(path,'email_src.csv')
    dst_file = os.path.join(path,'email_train.csv')

    with open(raw_file, 'r', encoding='UTF8') as f:
        data = json.load(f)
    df = pandas.read_json(raw_file, encoding='UTF8')
    #df.to_csv(src_file, sep=';', quoting=csv.QUOTE_ALL)
    df2 = json_normalize(data=data, record_path='slas')
    df3 = pandas.concat([df, df2], axis=1)
    # change file path
    df3.to_csv(src_file, sep=';', quoting=csv.QUOTE_ALL)
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
    salt = "insert_salt_here"

    with open(src_file, 'r', encoding='utf8') as csvfile:
        # The csv file gets assigned to reader
        reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        # Open a new csv file in which we will write the update content
        with open(dst_file, 'w') as destination:
            # reset reader to first line
            csvfile.seek(0)
            # Define output columns
            reader.fieldnames = "", "actionCodes", "assignedGroup","assigneeUsername","countComments", \
                                "countIncomingMails","countOutgoingMails","created","creationType","description", \
                                "issueId","issueKey","issueTypeCode","issueTypeId","issueTypeName", \
                                "lastStatusTransistion","loadId","loggedWork","priorityName","projectKey", \
                                "reporterUsername","responsibleCompanyKey","responsibleCompanyKeyManual", \
                                "responsibleParty","slas","statusName","transportOrderId","updated",\
                                "slaGoal","slaMet","slaName","slaStatus","slaTimeLeft", "txt_subj", \
                                "txt_main", "pos_main", "pos_subj", "neg_subj", "neu_subj", "posi_subj", "compound_subj", \
                                "neg_body", "neu_body", "posi_body", "compound_body", "ID", "language"
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
                # Main routine for the pre-processing of the mail body
                # Generates
                try:
                    # Identify language
                    temp["language"] = detect(temp["description"])
                    # temp stores the text of column description for the current row in a map
                    # Example: {'Created': '30.Mai.2018', 'Is Cleaned': 'X',
                    # 'T': 'O_CR20 - Carrier fails to execute transport',
                    # 'Description': 'TO 10237544 \n----\nHello 4flow, DHL team,\n\ncould you please... \n',
                    # 'Transport start [Date]': '', 'Transport end [Date]': '', 'Transport Order ID': '10237544',
                    # 'Transportation mode': ''}
                    # temp[col] is the mapped value of a specific column (e.g. a string)
                    pos_body, list2_body, pos_subj, list2_subj, neg_subj, neu_subj, posi_subj, compound_subj, neg_body, neu_body, posi_body, compound_body = clean(temp["description"])
                    temp["posi_subj"] = posi_subj
                    temp["neu_subj"] = neu_subj
                    temp["neg_subj"] = neg_subj
                    temp["compound_subj"] = compound_subj
                    temp["posi_body"] = posi_body
                    temp["neu_body"] = neu_body
                    temp["neg_body"] = neg_body
                    temp["compound_body"] = compound_body
                    # Hash subject POS
                    pos_subj_hash = ""
                    for i in pos_subj:
                        pos_subj_hash+= hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["pos_subj"] = hashing_trick(pos_subj_hash, 1000000, hash_function='md5')
                    # Hash Mail subject
                    txt_subj_hash = ""
                    for i in list2_subj:
                        txt_subj_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["txt_subj"] = hashing_trick(txt_subj_hash, 1000000, hash_function='md5')
                    # Hash main POS
                    pos_main_hash = ""
                    for i in pos_body:
                        pos_main_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["pos_main"] = hashing_trick(pos_main_hash, 1000000, hash_function='md5')
                    # Hash main POS
                    txt_main_hash = ""
                    for i in list2_body:
                        txt_main_hash += hashlib.sha512((salt + str(i)).encode('utf-8')).hexdigest() + " "
                    temp["txt_main"] = hashing_trick(txt_main_hash, 1000000, hash_function='md5')
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
                        print("Error in row " + str(index))
                        raise
                # remove the column description, this column has been transformed into
                # "sub_subj", "pol_subj", "pos_subj", "txt_subj", "sub_main", "pol_main","pos_main", "txt_main"
                temp['description'] = ""
                # Assign a incremental ID to the email to later return the class and the ID
                temp['ID'] = str(index-1)
                print("Successfully finished row " + str(index))
                index += 1

                # Row is written including the changes made
                writer.writerow(temp)


