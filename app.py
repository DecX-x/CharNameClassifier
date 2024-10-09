import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
import string
import os
import glob

def findfiles(path): return glob.glob(path)
print(findfiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

category_lines = {}
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findfiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

# Define the RNN model class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Load the pre-trained model
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_hidden = 128
n_categories = len(all_categories)

rnn = RNN(n_letters, n_hidden, n_categories)
rnn.load_state_dict(torch.load('rnn.pth'))

# Helper functions
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def topCategoriesFromOutput(output, n=3):
    top_n, top_i = output.topk(n)
    categories_and_scores = [(all_categories[top_i[0][i].item()], top_n[0][i].item()) for i in range(n)]
    return categories_and_scores

def predict(input_line, n=3):
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))
        categories_and_scores = topCategoriesFromOutput(output, n)
        return categories_and_scores

# Create a Streamlit web app interface
st.title('Name Classification App')
name_input = st.text_input('Enter a name:')
if st.button('Classify'):
    categories_and_scores = predict(name_input)
    st.write(f'The name "{name_input}" is classified as:')
    for category, score in categories_and_scores:
        st.write(f'({score:.2f}) {category}')