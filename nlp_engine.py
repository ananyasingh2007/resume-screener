from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import string

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

def rank_resumes(jd_text, resumes):
    jd_clean = preprocess(jd_text)
    resume_names = list(resumes.keys())
    resume_texts = [preprocess(resumes[name]) for name in resume_names]

    all_texts = [jd_clean] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors)[0]
    results = [(resume_names[i], round(float(scores[i]) * 100, 2))
               for i in range(len(resume_names))]
    results.sort(key=lambda x: x[1], reverse=True)
    return results