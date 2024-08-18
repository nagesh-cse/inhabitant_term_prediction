from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.utils import shuffle
import streamlit as st


# Initialize Porter stemmer
stemmer = PorterStemmer()

df = pd.read_csv('./demonym.csv')
trainSize = int(0.8*df.shape[0])
valSize = int(0.1*df.shape[0])
df = shuffle(df, random_state=20)

# Spliting test train data of ration 3:7
trainD = df[:trainSize]
valD = df[trainSize:valSize+trainSize]
testD = df[valSize+trainSize:]

# Generating corpus of inhabitant terms
corpus = ' '.join([i for i in trainD['Demonym']]).lower()

# byte pair algorithm
bpe_codes_file = "bpe_codes.txt"
num_operations = 10000
corpus_words = corpus.strip().split(" ")

# Training the BPE and generate bpe_codes for rules
with open(bpe_codes_file, 'w', encoding='utf-8') as outfile:
    learn_bpe(corpus_words, outfile, num_operations)

# Load BPE codes
bpe = BPE(open(bpe_codes_file, encoding='utf-8'))


# Read and process the BPE codes as rules from the file
bpe_rules = {}
with open("bpe_codes.txt", 'r', encoding='utf-8') as infile:
    for line in infile:
        line = line.strip()
        if line:
            components = line.split(' ')
            merged_unit = components[0]
            subword_units = components[1]
            # Ensuring only the most repeated pairs remain in rules
            # Also included only those rules with ending symbol
            if '</w>' in subword_units and merged_unit not in bpe_rules:
                bpe_rules[merged_unit] = subword_units.replace('</w>', '')


def find_inhabitant_term(city_name, bpe_rules):
    city_name = stemmer.stem(city_name)
    tokenized_units = bpe.segment(city_name).split()
    tokenized_units = [unit.replace('@@', '') for unit in tokenized_units]

    for i in range(0, len(tokenized_units)):
        # Finding if the maximum part of city name (by joined token from start to end) in the rules
        # Removing each from token at every iteration
        partial_units = tokenized_units[i:]
        partial_word = ''.join(partial_units)
        if partial_word in bpe_rules:
            return city_name + bpe_rules[partial_word]

    for i in range(0, len(tokenized_units[-1])):
        # Incase any token of name does not have rule then searching character wise
        # Removing each from character of last token at each iteration
        partial_word = tokenized_units[-1][i:]
        if partial_word in bpe_rules:
            return city_name + bpe_rules[partial_word]

    # Incase no rule for name,, then default option
    return city_name + 'er'

# Score(accuracy) of model out of 100
# df is dataframe['City', 'Demonym']
def score(df):
    df_copy = df.copy()

    df_copy['City'] = df_copy['City'].apply(lambda x: find_inhabitant_term(x, bpe_rules))
    df_copy['Demonym'] = df_copy['Demonym'].apply(lambda x: x.lower())

    accuracy = df_copy[df_copy.City == df_copy.Demonym].shape[0]/df_copy.shape[0]
    print("Accuracy - " + str("%.2f"%(accuracy*100)))

# Testing the model on various dataset - Test set, train set, whole data
print("On Test set")
score(testD)
print("\nOn Validation set")
score(valD)
print("\nOn Training set")
score(trainD)
print("\nOn whole data")
score(df)

print("\n")

# To check for some name 
city_name = st.text_input("Enter the city name")

if city_name:
    st.write("Output: " + find_inhabitant_term(city_name,bpe_rules).title())
