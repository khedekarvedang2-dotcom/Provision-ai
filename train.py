from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

texts = [

"i am happy",
"today is awesome",
"life is good",
"this is amazing",

"i am sad",
"bad day",
"i feel terrible",
"i hate this",

"artificial intelligence",
"machine learning",
"python coding",
"technology is cool",

"bmw m5",
"cars are cool",
"i love driving",
"supra is fast"

]

labels = [

"positive",
"positive",
"positive",
"positive",

"negative",
"negative",
"negative",
"negative",

"tech",
"tech",
"tech",
"tech",

"cars",
"cars",
"cars",
"cars"

]

model = Pipeline([
("tfidf", TfidfVectorizer()),
("clf", MultinomialNB())
])

model.fit(texts, labels)

joblib.dump(model,"model.pkl")

print("Training Complete")