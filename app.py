# Importing required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem.porter import PorterStemmer
import pickle
from flask import Flask, request, jsonify, render_template

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# 1. Data Loading and Preprocessing
df = pd.read_csv("spam.csv", encoding="ISO-8859-1")

# Dropping unnecessary columns and renaming the necessary ones
df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target (ham -> 0, spam -> 1)
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Remove duplicates
df.drop_duplicates(keep="first", inplace=True)

# 2. Text Preprocessing Function
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuations
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Perform stemming
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Apply transformation to the text data
df['transformed_text'] = df['text'].apply(transform_text)

# 3. Feature Extraction using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()

# Target variable
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 4. Model Building
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

# 6. Save the Model and Vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))

# 7. Flask Application
app = Flask(__name__)

# Load the saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the email text from the form
        email_text = request.form['email_text']

        # Preprocess and transform the email text using the vectorizer
        transformed_text = vectorizer.transform([email_text])

        # Make the prediction
        prediction = model.predict(transformed_text)[0]

        # 1 means spam, 0 means ham (not spam)
        result = 'Spam' if prediction == 1 else 'Not Spam'

        # Return the result to the HTML page
        return render_template('index.html', prediction=result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
