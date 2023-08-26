from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('personality.csv')

# Preprocessing: Extract features from resumes
def extract_resume_features(resume_text):
    # Assuming you've loaded NLTK and stopwords
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    # Preprocess text: Remove special characters, lowercase, tokenize, remove stopwords, and apply stemming
    clean_text = re.sub(r'\W+', ' ', resume_text).lower()
    words = clean_text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Combine personality responses and resume features
def combine_user_data(user_responses, user_resume_features):
    # Combine the user's responses and the preprocessed resume text
    user_combined = user_responses + [user_resume_features]
    return user_combined

# Preprocess the dataset and split into features (X) and labels (y)
X = dataset.drop(['Personality'], axis=1)  # Remove 'Personality' column from features
y = dataset['Personality']  # 'Personality' column as target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline for personality prediction
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the personality prediction model
pipeline.fit(X_train, y_train)  # Using all available features

# Save the trained model
joblib.dump(pipeline, 'personality_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_responses = [int(request.form['openness']), int(request.form['neuroticism']),
                      int(request.form['conscientiousness']), int(request.form['agreeableness']),
                      int(request.form['extraversion'])]
    user_resume = request.files['resume'].read().decode('utf-8')
    
    user_resume_features = extract_resume_features(user_resume)
    user_combined = combine_user_data(user_responses, user_resume_features)
    
    user_pred = pipeline.predict([user_combined])
    user_personality = user_pred[0]
    
    return render_template('result.html', personality=user_personality)

if __name__ == '__main__':
    pipeline = joblib.load('personality_model.pkl')  # Load the trained model
    app.run(debug=True)
