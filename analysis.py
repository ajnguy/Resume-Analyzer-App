from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import fitz 
import spacy
import numpy as np
from transformers import pipeline


def convert_pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text.strip()

class Analyze:
    def __init__(self, resume_pdf_path, job_description):
        self.resume_text = convert_pdf_to_text(resume_pdf_path)
        self.job_description_text = job_description
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load("en_core_web_sm")
        self.missing_keywords = None
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def calculate_cosine_similarity(self):
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([self.resume_text, self.job_description_text])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return similarity_score
    
    def calculate_semantic_similarity(self):
        resume_embedding = self.model.encode(self.resume_text, convert_to_tensor=True)
        job_embedding = self.model.encode(self.job_description_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
        return similarity_score
    
    def extract_keywords_ner(self, text):
        doc = self.nlp(text)
        keywords_with_scores = []

        # Assign importance scores to entity types
        entity_weights = {
            'LANGUAGE': 1.0,     # Programming languages are highly relevant
            'PRODUCT': 0.9,      # Software/technologies are also very important
            'ORG': 0.8,          # Companies/organizations are useful
        }

        for ent in doc.ents:
            if ent.label_ in entity_weights:
                score = entity_weights[ent.label_]
                keywords_with_scores.append((ent.text.lower(), score))

        # Sort keywords by importance score (descending order)
        ranked_keywords = sorted(keywords_with_scores, key=lambda x: x[1], reverse=True)

        return np.array([kw[0] for kw in ranked_keywords])  # Return only the words

    def extract_keywords_tfidf(self, text):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

        # Extract top TF-IDF keywords
        top_keywords = feature_names[np.argsort(tfidf_matrix.sum(axis=0)).flatten()[::-1]]

        return top_keywords  

    def calculate_boosted_semantic_similarity(self):
        # Extract keywords from both texts (NER + TF-IDF)
        resume_keywords_ner = self.extract_keywords_ner(self.resume_text)
        job_keywords_ner = self.extract_keywords_ner(self.job_description_text)

        resume_keywords_tfidf = self.extract_keywords_tfidf(self.resume_text)
        job_keywords_tfidf = self.extract_keywords_tfidf(self.job_description_text)

        # Combine NER and TF-IDF extracted keywords using NumPy union
        resume_keywords = np.union1d(resume_keywords_ner, resume_keywords_tfidf)
        job_keywords = np.union1d(job_keywords_ner, job_keywords_tfidf)
        
        # Setting missing keywords in resume
        missing_keywords = np.setdiff1d(job_keywords, resume_keywords)
        self.missing_keywords = missing_keywords
        
        print(resume_keywords_ner)
        print(job_keywords_ner)
        
        # Calculate semantic similarity using SentenceTransformer
        resume_embedding = self.model.encode(self.resume_text, convert_to_tensor=True)
        job_embedding = self.model.encode(self.job_description_text, convert_to_tensor=True)
        base_similarity = util.pytorch_cos_sim(resume_embedding, job_embedding).item()

        # Adjust the similarity score based on the overlap of important keywords
        keyword_overlap = len(np.intersect1d(resume_keywords, job_keywords))  # NumPy intersection
        boosted_similarity = base_similarity + (0.1 * keyword_overlap)  # Boost the score based on overlap

        return boosted_similarity
    
    def fit_resume(self):
        """
        Uses zero-shot classification to assess the resume's relevance to the job description.
        Returns the predicted category with the highest confidence score.
        """
        labels = ["Highly Relevant", "Moderately Relevant", "Not Relevant"]
        result = self.classifier(self.resume_text, candidate_labels=labels, 
                             hypothesis_template=f"This resume is relevant to the job description: {self.job_description_text} in the following way: {{}}")


        # Get the highest scoring label
        best_fit = result['labels'][0]
        confidence = result['scores'][0]

        return best_fit, confidence
    