from nltk.stem import PorterStemmer
import pandas as pd
import streamlit as st


# Initialize Porter stemmer
stemmer = PorterStemmer()

inflectional_rules = {
    'a': 'an',
    'i': 'ian',
    'u': 'an',
    'e': 'an',
    'o': 'ian',
}

def find_inhabitant_term(city_name):
    # Apply Porter stemming to city name
    base_form = stemmer.stem(city_name)

    if base_form[-1] in inflectional_rules:
        inhabitant_term = base_form[:-1] + inflectional_rules[base_form[-1]]
    else:
        inhabitant_term = base_form + 'er'

    return inhabitant_term


def score(df):
    df_copy = df.copy()

    df_copy['City'] = df_copy['City'].apply(lambda x: find_inhabitant_term(x))
    df_copy['Demonym'] = df_copy['Demonym'].apply(lambda x: x.lower())

    accuracy = df_copy[df_copy.City == df_copy.Demonym].shape[0]/df_copy.shape[0]
    print("Accuracy - " + str("%.2f" % (accuracy*100)))


df = pd.read_csv('./demonym.csv')
score(df)


# To check for some name 


city_name = st.text_input("Enter the city name")
if city_name:
    st.write("Output: " + find_inhabitant_term(city_name).title())