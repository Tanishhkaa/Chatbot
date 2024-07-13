import nltk
import warnings
import random
import string
from flask import Flask, render_template, request

# creates a web application instance
app = Flask(__name__)

f = open("D:\\gehu.txt", 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')
#Provides wordnet data in multiple languages
nltk.download('omw-1.4')

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# Checking for greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# the words need to be encoded as integers or floating point values
from sklearn.feature_extraction.text import TfidfVectorizer

# find the similarity between words entered by the user and the words in the corpus.
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # Cosine similarity is a measure of similarity between two non-zero vectors.
    # Cosine Similarity (d1, d2) =  Dot product(d1, d2) / ||d1|| * ||d2||
    vals = cosine_similarity(tfidf[-1], tfidf)

   # It returns an array of indices
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()

    flat.sort()
    req_tfidf = flat[-2]

    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

def simple_chatbot(user_message):
    user_response = user_message.lower()

    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            return "You are welcome.."
        else:
            if greeting(user_response) is not None:
                return greeting(user_response)
            else:
                return response(user_response)
    else:
        return "Bye! take care.."

user_messages = []
bot_messages = []
#Flask route for the home page
@app.route('/', methods=['GET', 'POST'])

def home():

    global user_messages,bot_messages
    #If HTTP request is post
    if request.method == 'POST':
        user_message = request.form['message']
        user_messages.append(user_message)

        bot_message = simple_chatbot(user_message)
        bot_messages.append(bot_message)

    return render_template('index.html', user_messages=user_messages, bot_messages=bot_messages)
app.run(host='0.0.0.0', port=5000, debug=True)
# whether the script is run directly
if __name__ == '__main__':
    app.run()
import webbrowser
flask_url = 'http://localhost:5000'
webbrowser.open(flask_url)




