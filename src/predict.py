import numpy as np
import os
from utils.preprocessing import clean_text
from models.logistic_regression import LogisticRegression
from models.neural_network import NeuralNetwork

def load_model(filename):
    model_path = os.path.join(os.path.dirname(__file__), f'../models/trained/{filename}')
    return np.load(model_path, allow_pickle=True).item()

def load_vocabulary():
    vocab_path = os.path.join(os.path.dirname(__file__), '../models/vocabulary/vocab.npy')
    idf_path = os.path.join(os.path.dirname(__file__), '../models/vocabulary/idf.npy')
    return np.load(vocab_path), np.load(idf_path)

def predict(texts):
    # Load vocabulary and models
    vocab, idf = load_vocabulary()
    lr_data = load_model('logistic_regression.npy')
    nn_data = load_model('neural_network.npy')
    
    # manual TF-IDF 
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        words = clean_text(text).split()
        for word in words:
            if word in vocab:
                idx = np.where(vocab == word)[0][0]
                tf = words.count(word) / len(words)
                X[i, idx] = tf * idf[idx]
    
    # 3. Configurar modelos
    lr_model = LogisticRegression()
    lr_model.weights = lr_data['weights']
    lr_model.bias = lr_data['bias']
    
    nn_model = NeuralNetwork(input_size=len(vocab))
    nn_model.W1 = nn_data['W1']
    nn_model.b1 = nn_data['b1']
    nn_model.W2 = nn_data['W2']
    nn_model.b2 = nn_data['b2']
    
    # Predict
    lr_preds = lr_model.predict(X)
    nn_preds = nn_model.predict(X)
    
    return lr_preds, nn_preds

if __name__ == "__main__":
    test_texts = [
        "The movie was a masterpiece with brilliant performances",
        "Terrible plot and awful acting",
        "It was okay, nothing special"
    ]
    
    lr_results, nn_results = predict(test_texts)
    
    for text, lr, nn in zip(test_texts, lr_results, nn_results):
        print(f"\n📄 Text: '{text}'")
        prob_nn = nn[0]
        confidence_nn = prob_nn if prob_nn > 0.5 else 1 - prob_nn

        prob_lr = lr
        confidence_lr= prob_lr if prob_lr > 0.5 else 1 - prob_lr

        print(f"  - Logistic Regression: {'👍 Positive' if prob_lr > 0.5 else '👎 Negative'}")
        print(f"    - Confidence: {confidence_lr*100:.1f}%")
        print(f"  - Red Neuronal: {'👍 Positive' if prob_nn > 0.5 else '👎 Negative'}")
        print(f"    - Confidence: {confidence_nn*100:.1f}%")