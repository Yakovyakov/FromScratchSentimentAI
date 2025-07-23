import numpy as np
import csv
import os
from utils.preprocessing import clean_text
from utils.tfidf_vectorizer import TFIDFVectorizer
from models.logistic_regression import LogisticRegression
from models.neural_network import NeuralNetwork

def save_model(model, filename):
    model_dir = os.path.join(os.path.dirname(__file__), '../models/trained')
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, filename), model)

def train_and_save():
    
    
    data_path = os.path.join(os.path.dirname(__file__), '../data/raw/reviews.csv')
    data_array = []
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            if len(row) == 2:
                transformed_row = [f'"{row[0]}"', row[1]]
                data_array.append(transformed_row)
    data_array = data_array[1:]
    data = np.array(data_array, dtype='object')


    texts = [clean_text(text) for text in data[:, 0]]
    labels = data[:, 1].astype(int)


    # TF-IDF
    tfidf = TFIDFVectorizer()
    tfidf.fit(texts)
    X = tfidf.transform(texts)
    
    # Guardar vocabulario
    vocab_dir = os.path.join(os.path.dirname(__file__), '../models/vocabulary')
    os.makedirs(vocab_dir, exist_ok=True)
    np.save(os.path.join(vocab_dir, 'vocab.npy'), np.array(tfidf.vocab))
    np.save(os.path.join(vocab_dir, 'idf.npy'), np.array(list(tfidf.idf.values())))

    
    # Train models and save
    lr_model = LogisticRegression(lr=0.1, epochs=1000)
    lr_model.fit(X, labels)
    save_model({'weights': lr_model.weights, 'bias': lr_model.bias}, 'logistic_regression.npy')

    nn_model = NeuralNetwork(input_size=X.shape[1])
    nn_model.train(X, labels, epochs=1000, learning_rate=0.01)
    save_model({'W1': nn_model.W1, 'b1': nn_model.b1, 'W2': nn_model.W2, 'b2': nn_model.b2}, 'neural_network.npy')
    
if __name__ == "__main__":
    train_and_save()
    print("✅ Training completed and models saved")
