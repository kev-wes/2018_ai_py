# Needed project interpreters:
# sklearn
import pickle
from sklearn.svm import SVC
from api import tfidf_extractor, read_data

if __name__ == "__main__":
        # Read features and labels of csv file
        # text_mail is an array with all email bodies
        # label_mail is an array with all the labels of the training mails
        text_mail, label_mail, id_mail = read_data('pre_processing\email_train.csv', True)
        # Export:
        # text: Text bodies of mail, each mail body stored in one array index
        # ngram_range: range of ngrams which are constructed
        # (1, 1) => only unigrams
        # (1, 2) => unigrams and bigrams
        # (2, 2) => only bigrams etc.
        # Return
        # tfidf_vectorizer: vectorizer
        # tfidf_train_features: term-document matrix with following structure:
        # ([index_of_mail], [index_of_hash_in_vocab])   [tf_idf_weight]
        # e.g. (0, 50)	0.46831693439753186
        tfidf_vectorizer, tfidf_features = tfidf_extractor(text_mail, (1, 1))
        # Save the vectorizer
        filename = 'tfidf_vectorizer.sav'
        pickle.dump(tfidf_vectorizer, open(filename, 'wb'))
        # The features are saved
        filename = 'tfidf_train_features.sav'
        pickle.dump(tfidf_features, open(filename, 'wb'))
        ## Print the tf-idf-weighted document-term matrix
        # print(tfidf_train_features)
        ## Print the hashes and their mapped vocab index
        # print(tfidf_vectorizer.vocabulary_)
        # The model is parameterized as a support vector machine
        classifier = SVC(kernel='linear', C=1.0, probability=True, random_state=0)
        # Build model
        classifier.fit(tfidf_features, label_mail)
        # Save the model to disk
        filename = 'svm_email.sav'
        pickle.dump(classifier, open(filename, 'wb'))