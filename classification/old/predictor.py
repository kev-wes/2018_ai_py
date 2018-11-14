# Needed project interpreters:
# sklearn
import pickle
from api import read_data
import numpy as np

if __name__ == "__main__":
        # Read features and labels of csv file
        # text_mail is an array with all email bodies
        # label_mail is empty, because train_mode == False
        text_mail, label_mail, id_mail = read_data('pre_processing\email_test.csv', False)
        # Read the saved files created by trainer.py
        # Load SVM
        filename = 'svm_email.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        # Load vectorizer
        filename = 'tfidf_vectorizer.sav'  # Load the tfidf word matrix which the model trained
        tfidf_vectorizer = pickle.load(open(filename, 'rb'))
        # Vectorize features
        # tfidf_train_features is a word matrix, with all the words of the mails
        tfidf_test_features = tfidf_vectorizer.transform(text_mail)
        # Run predictor
        predictions = loaded_model.predict(tfidf_test_features)
        # Calculate the probabilities of the prediction
        output_proba = loaded_model.predict_proba(tfidf_test_features)

        for id, pred, proba in zip(id_mail, predictions, output_proba):
                ind = np.argpartition(proba, -1)[-1:]  # Take the better probability
                print("Row "+ str(id) + ": The exception \'"+ pred + "\' has been predicted with an accuracy of " + str(proba[ind]))