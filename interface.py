import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import streamlit as st
from os import listdir, path

IMAGES_FOLDER = 'my project image'
all_images = listdir(IMAGES_FOLDER)

def show_images(keyword):
    found_images = [image for image in all_images if image.lower().startswith(keyword)]
    for image in found_images:
        st.image(path.join(IMAGES_FOLDER, image), width=300)
        st.write(image.split('.')[0])

st.title('Fedpoffa (Mini Campus) Chatbot App')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model("chatbot_model.keras")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_TRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_TRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


import random


def get_response(intents_list, intents_json):
    if not intents_list:
        print("No intent detected. Fallback triggered.")
        return "Sorry, I didn't understand that. Could you rephrase?"

    tag = intents_list[0]['intent']
    print(f"Detected intent: {tag}")

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            answer = intent['responses']
            print(answer)
            response = random.choice(answer)

            print(f"Selected response: {response}")
            return response

    print("No matching tag found. Fallback triggered.")
    return "I'm not sure how to respond to that."


feedback = st.text
user_message = st.text_area('User: ')
button = st.button('Send')
placeholder = st.empty()
response = ''


default_value = ''

if 'display' not in st.session_state:
    st.session_state.display = default_value

if button:
    if user_message == '' or user_message is None:
        feedback('Please enter the message first')
    else:
        predictions = predict_class(user_message.lower())
        response = get_response(predictions, intents)
        st.session_state.display += f'User: {user_message}\n'
        st.session_state.display += f'Bot: {response}\n'
        st.session_state.display += f"{'-'*50}\n"

display = placeholder.text_area('', height=15, key='display')
if 'computer science' in response.lower():
    show_images("comsci")
elif 'adesoye' in response.lower() or 'hadesoye' in response.lower():
    show_images('aah')
elif 'library' in response.lower():
    show_images('library')
elif 'stella' in response.lower():
    show_images('stella')
elif 'olatinwo' in response.lower() or 'olawoyin' in response.lower():
    show_images('olat')
elif 'boys' in response.lower():
    show_images('boys')
elif 'security' in response.lower(): #its not show image
    show_images('security')
elif 'statistic' in response.lower():
    show_images('statistic')
elif 'medical' in response.lower():
    show_images('medical')
elif 'admin' in response.lower():
    show_images('admin') #its drop location for another place
elif 'ice office' in response.lower():
    show_images('ice office')
elif 'ice venue' in response.lower():
    show_images('ice venue')
elif 'moremi' in response.lower():#Im looking for moremi class?
    show_images('moremi')
elif 'engineering' in response.lower():
    show_images('engine')
elif 'hnd venue' in response.lower():
    show_images('hnd venue')
elif 'mosque' in response.lower():
    show_images('mosque')
elif 'bio data' in response.lower():
    show_images('bio data')
elif 'biochemistry' in response.lower():
    show_images('biochemistry')
elif 'biology' in response.lower():
    show_images('biology')
elif 'stadium' in response.lower():
    show_images('stadium')
elif 'clearance' in response.lower():
    show_images('clearance')
elif 'girls' in response.lower():
    show_images('girls')
elif 'market' in response.lower():
    show_images('market')
elif 'aluta' in response.lower():
    show_images('aluta')
elif 'market' in response.lower():
    show_images('market')
elif 'ptf' in response.lower():
    show_images('ptf')
elif 'love garden' in response.lower():
    show_images('love garden')
elif 'ijmb' in response.lower():
    show_images('ijmb')
elif 'physic' in response.lower():
    show_images('physic')
elif 'osun' in response.lower():
    show_images('osun')
elif 'purchase' in response.lower():
    show_images('purchase')
elif 'bakre' in response.lower():
    show_images('bakre')
elif 'bank' in response.lower():
    show_images('bank')

# while True:
#     # message = input('user: ')
#     user_message = st.text_area('User:', )
#     ints = predict_class(message)
#     res = get_response(ints, intents)
#     print(res)