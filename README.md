# CharNameClassifier

This repository contains a Streamlit web app for name classification using Recurrent Neural Network (RNN). The app predicts the likely origin of a given name based on its spelling.



## Description

The CharNameClassifier app uses a pre-trained RNN to classify names into different categories based on their spelling. It supports multiple categories and provides the top three predictions with their respective scores.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/DecX-x/CharNameClassifier.git
    cd CharNameClassifier
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model:**
    Ensure that the `rnn.pth` file is in the root directory of the project.

## Usage

To start the Streamlit web app, run the following command:
```bash
streamlit run app.py
```

Open your web browser and go to `http://localhost:8501`. Enter a name in the input box and click the "Classify" button to see the classification results.

## Model Details

#### Model Architecture

The model is a Recurrent Neural Network (RNN) with the following architecture:

- **Input Layer**: Converts characters to tensor representations.
- **Hidden Layers**: Capture the sequence information using linear transformations and Tanh activation functions.
- **Output Layer**: Uses a softmax function to predict the category.

#### Input and Output Details

- **Input**: A tensor representing a sequence of characters, where each character is converted to a one-hot encoded vector of size `n_letters`.
- **Hidden State**: A tensor of size `hidden_size` that maintains the state across the sequence.
- **Output**: A tensor of size `output_size`, representing the log probabilities of each category.

#### Training Details

The model was trained on a dataset of names categorized by their origin. The training process involved:

- **Loss Function**: Negative Log-Likelihood Loss.
- **Optimizer**: Stochastic Gradient Descent (SGD).
- **Training Data**: Names from different categories, converted to ASCII and one-hot encoded.

#### Usage

To use the model for inference:

1. **Load the Pre-trained Model**: Load the model parameters from a file.
2. **Prepare the Input**: Convert the input name to a tensor.
3. **Make Predictions**: Pass the input tensor through the model to get the category probabilities.

The model can classify names into categories such as "Scottish," "Polish," "Russian," etc., and provides the top N categories along with their respective scores.
