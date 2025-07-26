import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('news.csv', names=['sentiment', 'text'])
df = df[1:]




df.dropna(inplace=True)

X= df['text']
Y= df['sentiment']

model = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

X_train , X_test, Y_train, Y_test= train_test_split(X , Y , test_size=0.2)
model.fit(X_train , Y_train)

joblib.dump(model, 'model.pkl')
print("Model fully trained and saved under model.pkl")