<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Movie Review Sentiment Analyzer

🧩 🎭 🎬 🤖

Final project for the Building AI course

## Summary

A sentiment analysis system that classifies movie reviews as positive or negative using TF-IDF with NumPy, logistic regression, and neural networks - all implemented from scratch without ML libraries.

## Background

**Problems solved:**

* Subjective manual review analysis
* Lack of accessible text classification tools
* Over-reliance on complex libraries

**Personal motivation:**
As a cinema enthusiast and AI student, I wanted to create a transparent system that reveals the inner workings of NLP algorithms.

**Importance:**

* Demonstrates text processing fundamentals
* Provides educational code for students
* 100% customizable with no frameworks

## How is it used?

**Workflow:**

1. Training:

    ```bash
    python src/train.py
    ```

2. Prediction:

    ```bash
    python src/predict.py "excellent movie"
    ```

**Use cases:**

* Streaming platforms for review analysis

* Small businesses understanding customer feedback

* Educational tool for AI courses

## Data sources and AI methods

Data source:

* Custom review dataset (50 samples) + optional IMDB Dataset

| Method      | Implememtation | Accuracy |
| ----------- | ----------- | -----------  |
| TF-IDF       | Manual NumPy vectorization kernel | -       |
| Logistic Regression   | From-scratch algorithm | 80%        |
| Neural Network | 2-layer (ReLU + Sigmoid) | 85% |

## Challenges

**Limitations:**

* Only works with known vocabulary

* Lower accuracy than transformer solutions

* Requires retraining for new domains

**Ethical considerations:**

* Potential training data biases

* Not suitable for critical automated decisions

## What next?

**Planned improvements:**

* Add Flask web interface

* Implement Word2Vec manually

* Multi-language support

**Required assistance:**

* UI/UX designers

* Contributors to expand datasets

## Acknowledgments

* Inspired by University of Helsinki's "Building AI" course

* Code structure adapted from Implementing NLP from Scratch

* Sample dataset created by author for educational purposes

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
