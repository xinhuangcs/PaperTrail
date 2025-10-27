from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from my_tfidf import MyTfidfVectorizer



# Define a sample corpus of documents.

corpus = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document',
]




# Initialize and run the scikit-learn vectorizer.

sklearn_vectorizer = TfidfVectorizer()
sklearn_tfidf_matrix = sklearn_vectorizer.fit_transform(corpus)

# Initialize and run our custom vectorizer.

my_vectorizer = MyTfidfVectorizer()
my_vectorizer.fit(corpus)
my_tfidf_matrix = my_vectorizer.transform(corpus)


print("--- Validation Results ---")

# a. Compare vocabularies
try:
    assert my_vectorizer.vocabulary_ == sklearn_vectorizer.vocabulary_
    print("✅ Vocabulary Test: PASSED")
except AssertionError:
    print("❌ Vocabulary Test: FAILED")
    print("My vocab:", my_vectorizer.vocabulary_)
    print("Sklearn vocab:", sklearn_vectorizer.vocabulary_)

# b. Compare IDF vectors
try:
    # Use np.allclose for robust floating-point comparison.
    assert np.allclose(my_vectorizer.idf_, sklearn_vectorizer.idf_)
    print("✅ IDF Vector Test: PASSED")
except AssertionError:
    print("❌ IDF Vector Test: FAILED")
    print("My IDF:", my_vectorizer.idf_)
    print("Sklearn IDF:", sklearn_vectorizer.idf_)

# c. Compare final TF-IDF matrices
try:
    # Convert sparse matrices to dense arrays for comparison.
    assert np.allclose(my_tfidf_matrix.toarray(), sklearn_tfidf_matrix.toarray())
    print("✅ TF-IDF Matrix Test: PASSED")
except AssertionError:
    print("❌ TF-IDF Matrix Test: FAILED")
    print("My Matrix:\n", my_tfidf_matrix.toarray())
    print("Sklearn Matrix:\n", sklearn_tfidf_matrix.toarray())

print("\n--- Detailed Outputs ---")
print("\nVocabulary (sorted by term):")
print(sorted(my_vectorizer.vocabulary_.items(), key=lambda item: item[0]))
print("\nIDF Vector:")
print(list(zip(my_vectorizer.vocabulary_.keys(), my_vectorizer.idf_)))
print("\nFinal TF-IDF Matrix (dense representation):")
print(my_tfidf_matrix.toarray())