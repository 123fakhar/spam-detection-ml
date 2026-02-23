
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv", encoding= "latin-1")
df = df[["v1","v2"]]
df.columns = ["Label","message"]
df["Label"] = df["Label"].map({"ham" : 0 , "spam": 1})
vectorizer = TfidfVectorizer(stop_words = "english")
X = vectorizer.fit_transform(df["message"])
Y = df["Label"]

X_train , X_test , Y_train , Y_test = train_test_split(X , Y ,test_size= 0.2, random_state= 42)
model = LogisticRegression()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(Y_test, y_pred))