# Next-Word Predictor using LSTM

## Overview
This project implements a next-word prediction model using Long Short-Term Memory (LSTM) networks. The dataset used for training is *The Tragedie of Hamlet by William Shakespeare (1599)*. The model takes a sequence of words as input and predicts the most probable next word.

## Dataset
The dataset consists of the text from *Hamlet*, preprocessed to remove unnecessary characters and formatted into sequences for training the LSTM model.

## Technologies Used
- **Python**
- **TensorFlow & Keras**
- **Natural Language Processing (NLP)**
- **LSTM Neural Networks**
- **Streamlit** (for deployment)

## Workflow

* Data Collection: We use the text of Shakespeare's Hamlet as our dataset. This rich, complex text provides a good challenge for our model.
* Data Preprocessing: The text data is tokenized, converted into sequences, and padded to ensure uniform input lengths. The sequences are then split into training and testing sets.
* Model Building: An LSTM model is constructed with an embedding layer, two LSTM layers, and a dense output layer with a softmax activation function to predict the probability of the next word.
* Model Training: The model is trained using the prepared sequences, with early stopping implemented to prevent overfitting. Early stopping monitors the validation loss and stops training when the loss stops improving.
* Model Evaluation: The model is evaluated using a set of example sentences to test its ability to predict the next word accurately.
* Deployment: A Streamlit web application is developed to allow users to input a sequence of words and get the predicted next word in real-time.

