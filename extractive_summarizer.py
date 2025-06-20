import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import Counter
import re
import pickle
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ExtractiveSummarizer:
    """
    Advanced Extractive Text Summarizer using multiple techniques:
    1. TF-IDF based scoring
    2. TextRank algorithm
    3. Position-based scoring
    4. Length-based filtering
    """
    
    def __init__(self, method='hybrid', min_sentence_length=10, max_sentence_length=200):
        """
        Initialize the ExtractiveSummarizer
        
        Args:
            method (str): Summarization method ('tfidf', 'textrank', 'hybrid')
            min_sentence_length (int): Minimum sentence length to consider
            max_sentence_length (int): Maximum sentence length to consider
        """
        self.method = method
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True
        )
        
    def preprocess_text(self, text):
        """
        Preprocess the input text
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespaces and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s\.\!\?]', '', text)
        
        return text
    
    def extract_sentences(self, text):
        """
        Extract and filter sentences from text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of filtered sentences
        """
        sentences = sent_tokenize(text)
        
        # Filter sentences based on length
        filtered_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) >= self.min_sentence_length and 
                len(sentence) <= self.max_sentence_length and
                len(word_tokenize(sentence)) >= 5):
                filtered_sentences.append(sentence)
        
        return filtered_sentences
    
    def calculate_tfidf_scores(self, sentences):
        """
        Calculate TF-IDF scores for sentences
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            np.array: TF-IDF scores for each sentence
        """
        if len(sentences) == 0:
            return np.array([])
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores as sum of TF-IDF values
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        return sentence_scores
    
    def calculate_textrank_scores(self, sentences):
        """
        Calculate TextRank scores for sentences
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            np.array: TextRank scores for each sentence
        """
        if len(sentences) <= 1:
            return np.ones(len(sentences))
        
        # Create TF-IDF matrix for similarity calculation
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Create graph and apply PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph, max_iter=100, tol=1e-4)
        
        # Convert to numpy array
        textrank_scores = np.array([scores[i] for i in range(len(sentences))])
        
        return textrank_scores
    
    def calculate_position_scores(self, sentences, total_sentences):
        """
        Calculate position-based scores (higher scores for sentences at beginning and end)
        
        Args:
            sentences (list): List of sentences
            total_sentences (int): Total number of sentences in document
            
        Returns:
            np.array: Position-based scores
        """
        position_scores = np.zeros(len(sentences))
        
        for i in range(len(sentences)):
            # Higher scores for beginning and end
            if i < total_sentences * 0.3:  # First 30%
                position_scores[i] = 0.8
            elif i > total_sentences * 0.7:  # Last 30%
                position_scores[i] = 0.6
            else:  # Middle
                position_scores[i] = 0.4
                
        return position_scores
    
    def summarize(self, text, summary_ratio=0.3, max_sentences=None):
        """
        Generate extractive summary
        
        Args:
            text (str): Input text to summarize
            summary_ratio (float): Ratio of sentences to include in summary
            max_sentences (int): Maximum number of sentences in summary
            
        Returns:
            str: Generated summary
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Extract sentences
        sentences = self.extract_sentences(text)
        
        if len(sentences) == 0:
            return "Unable to generate summary: No valid sentences found."
        
        if len(sentences) <= 2:
            return ' '.join(sentences)
        
        # Calculate number of sentences for summary
        if max_sentences is None:
            num_sentences = max(1, int(len(sentences) * summary_ratio))
        else:
            num_sentences = min(max_sentences, int(len(sentences) * summary_ratio))
            num_sentences = max(1, num_sentences)
        
        # Calculate scores based on method
        if self.method == 'tfidf':
            scores = self.calculate_tfidf_scores(sentences)
        elif self.method == 'textrank':
            scores = self.calculate_textrank_scores(sentences)
        elif self.method == 'hybrid':
            tfidf_scores = self.calculate_tfidf_scores(sentences)
            textrank_scores = self.calculate_textrank_scores(sentences)
            position_scores = self.calculate_position_scores(sentences, len(sentences))
            
            # Normalize scores
            if np.max(tfidf_scores) > 0:
                tfidf_scores = tfidf_scores / np.max(tfidf_scores)
            if np.max(textrank_scores) > 0:
                textrank_scores = textrank_scores / np.max(textrank_scores)
            
            # Combine scores with weights
            scores = (0.4 * tfidf_scores + 
                     0.4 * textrank_scores + 
                     0.2 * position_scores)
        else:
            raise ValueError("Method must be 'tfidf', 'textrank', or 'hybrid'")
        
        # Select top sentences
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Maintain original order
        
        # Generate summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def batch_summarize(self, texts, summary_ratio=0.3, max_sentences=None):
        """
        Summarize multiple texts
        
        Args:
            texts (list): List of texts to summarize
            summary_ratio (float): Ratio of sentences to include in summary
            max_sentences (int): Maximum number of sentences in summary
            
        Returns:
            list: List of generated summaries
        """
        summaries = []
        for text in texts:
            summary = self.summarize(text, summary_ratio, max_sentences)
            summaries.append(summary)
        
        return summaries
    
    def get_sentence_scores(self, text):
        """
        Get scores for all sentences in the text
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: (sentences, scores)
        """
        text = self.preprocess_text(text)
        sentences = self.extract_sentences(text)
        
        if len(sentences) == 0:
            return [], []
        
        if self.method == 'tfidf':
            scores = self.calculate_tfidf_scores(sentences)
        elif self.method == 'textrank':
            scores = self.calculate_textrank_scores(sentences)
        elif self.method == 'hybrid':
            tfidf_scores = self.calculate_tfidf_scores(sentences)
            textrank_scores = self.calculate_textrank_scores(sentences)
            position_scores = self.calculate_position_scores(sentences, len(sentences))
            
            # Normalize scores
            if np.max(tfidf_scores) > 0:
                tfidf_scores = tfidf_scores / np.max(tfidf_scores)
            if np.max(textrank_scores) > 0:
                textrank_scores = textrank_scores / np.max(textrank_scores)
            
            scores = (0.4 * tfidf_scores + 
                     0.4 * textrank_scores + 
                     0.2 * position_scores)
        
        return sentences, scores
    
    def save_model(self, filepath):
        """
        Save the model parameters
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'method': self.method,
            'min_sentence_length': self.min_sentence_length,
            'max_sentence_length': self.max_sentence_length,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Load model parameters
        
        Args:
            filepath (str): Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.method = model_data['method']
        self.min_sentence_length = model_data['min_sentence_length']
        self.max_sentence_length = model_data['max_sentence_length']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']