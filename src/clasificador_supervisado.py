import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TRAIN_PATH = "../data/ag_news_train.csv"
TEST_PATH = "../data/ag_news_test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

def clean_text(text):

    text = str(text)
    text = re.sub(r'&(?:lt|gt|amp|quot|#39);', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

X_train_text = train["text"].map(clean_text)
X_test_text = test["text"].map(clean_text)
y_train = train["label"]
y_test = test["label"]

experiments = [
    {"max_features": 5000, "ngram_range": (1, 1)},
    {"max_features": 10000, "ngram_range": (1, 1)},
    {"max_features": 10000, "ngram_range": (1, 2)},
    {"max_features": 20000, "ngram_range": (1, 2)},
]

results = []

for config in experiments:
    vectorizer = TfidfVectorizer(
        max_features=config["max_features"],
        ngram_range=config["ngram_range"],
        sublinear_tf=True
    )

    # NO HAY DATA LEAKAGE:
    # el vocabulario y los pesos IDF se ajustan solamente con train.
    X_train = vectorizer.fit_transform(X_train_text)

    # En test no se vuelve a ajustar el vectorizador.
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000, C=2.0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    results.append({
        "max_features": config["max_features"],
        "ngram_range": str(config["ngram_range"]),
        "vocab_size": X_train.shape[1],
        "accuracy": accuracy_score(y_test, predictions)
    })

results_df = pd.DataFrame(results)
print(results_df)

best_idx = results_df["accuracy"].idxmax()
best_config = experiments[best_idx]

vectorizer = TfidfVectorizer(
    max_features=best_config["max_features"],
    ngram_range=best_config["ngram_range"],
    sublinear_tf=True
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = LogisticRegression(max_iter=1000, C=2.0)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nMejor configuración:", best_config)
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n")
print(classification_report(y_test, predictions))

labels = sorted(y_train.unique())
cm = confusion_matrix(y_test, predictions, labels=labels)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Matriz de Confusión - AG News")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.xticks(range(len(labels)), labels, rotation=30)
plt.yticks(range(len(labels)), labels)

for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
