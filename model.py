import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score

# Download NLTK resources - updated to include all necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Add this line to download punkt_tab resource

class ResumeJobMatcher:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.tfidf_vectorizer = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def load_data(self, data_path=None):
        """Load the resume-job pairs data"""
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            raise ValueError("Data path not specified")
        
        print(f"Loading data from {self.data_path}")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} resume-job pairs")
        return self.data
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Join tokens back into text
        return ' '.join(tokens)
    
    def calculate_jaccard_similarity(self, text1, text2):
        """Calculate Jaccard similarity between two texts"""
        # Create sets of words
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        # Calculate Jaccard similarity
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ngram_overlap(self, text1, text2, n=2):
        """Calculate n-gram overlap between two texts"""
        # Generate n-grams
        tokens1 = text1.split()
        tokens2 = text2.split()
        
        if len(tokens1) < n or len(tokens2) < n:
            return 0.0
        
        ngrams1 = set(zip(*[tokens1[i:] for i in range(n)]))
        ngrams2 = set(zip(*[tokens2[i:] for i in range(n)]))
        
        # Calculate overlap
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_role_similarity(self, resume_job_history, job_title):
        """Calculate similarity between resume job history and job title"""
        if not resume_job_history or not job_title:
            return 0.0
        
        # Convert job title to lowercase
        job_title = job_title.lower()
        
        # Check if any job history title contains the job title
        for job in resume_job_history:
            if job.lower() in job_title or job_title in job.lower():
                return 1.0
        
        # Calculate max similarity using TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer()
        try:
            # Combine job history into a single string
            job_history_text = ' '.join(resume_job_history)
            
            # Create TF-IDF vectors
            tfidf_matrix = vectorizer.fit_transform([job_history_text, job_title])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def create_features(self, calculate_match_label=True):
        """Create features from the resume-job pairs"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("Creating features...")
        features = []
        labels = []
        
        for pair in tqdm(self.data):
            # Text preprocessing
            resume_text = self.preprocess_text(pair.get('resume_text', ''))
            job_description = self.preprocess_text(pair.get('job_description', ''))
            
            # Skills features
            resume_skills = pair.get('resume_skills', [])
            required_skills = pair.get('job_required_skills', [])
            preferred_skills = pair.get('job_preferred_skills', [])
            
            # Convert skills to lowercase for case-insensitive matching
            resume_skills_lower = [skill.lower() for skill in resume_skills]
            required_skills_lower = [skill.lower() for skill in required_skills]
            preferred_skills_lower = [skill.lower() for skill in preferred_skills]
            
            # Calculate required skills match
            required_skills_matched = sum(1 for skill in required_skills_lower if skill in resume_skills_lower)
            required_skills_match_ratio = required_skills_matched / len(required_skills_lower) if required_skills_lower else 0.0
            
            # Calculate preferred skills match
            preferred_skills_matched = sum(1 for skill in preferred_skills_lower if skill in resume_skills_lower)
            preferred_skills_match_ratio = preferred_skills_matched / len(preferred_skills_lower) if preferred_skills_lower else 0.0
            
            # Calculate total skills match
            all_job_skills = required_skills_lower + preferred_skills_lower
            all_skills_matched = sum(1 for skill in all_job_skills if skill in resume_skills_lower)
            total_skills_match_ratio = all_skills_matched / len(all_job_skills) if all_job_skills else 0.0
            
            # Experience features
            resume_experience = pair.get('resume_experience_years', 0)
            job_min_experience = pair.get('job_min_experience', 0)
            
            experience_ratio = resume_experience / job_min_experience if job_min_experience > 0 else (1.0 if resume_experience > 0 else 0.0)
            experience_difference = resume_experience - job_min_experience
            experience_match = 1 if resume_experience >= job_min_experience else 0
            
            # Education features
            education_levels = {
                '': 0,
                'High School': 1,
                'Associate\'s': 2,
                'Bachelor\'s': 3,
                'Master\'s': 4,
                'PhD': 5
            }
            
            resume_education = education_levels.get(pair.get('resume_education_level', ''), 0)
            job_education = education_levels.get(pair.get('job_required_education', ''), 0)
            education_match = 1 if resume_education >= job_education else 0
            education_difference = resume_education - job_education
            
            # Text similarity features
            # TF-IDF and cosine similarity will be calculated later in vectorize_text
            
            # Calculate Jaccard similarity
            jaccard_similarity = self.calculate_jaccard_similarity(resume_text, job_description)
            
            # Calculate n-gram overlap
            bigram_overlap = self.calculate_ngram_overlap(resume_text, job_description, n=2)
            trigram_overlap = self.calculate_ngram_overlap(resume_text, job_description, n=3)
            
            # Role similarity features
            resume_job_history = pair.get('job_history', [])
            job_title = pair.get('job_title', '')
            role_similarity = self.calculate_role_similarity(resume_job_history, job_title)
            
            # Certification match
            resume_certifications = pair.get('certifications', [])
            # Note: This is a placeholder. In a real scenario, you would extract certification requirements from the job description
            certification_match = 1 if resume_certifications else 0
            
            # Combine features
            feature = {
                'resume_text': resume_text,
                'job_description': job_description,
                'required_skills_match_ratio': required_skills_match_ratio,
                'preferred_skills_match_ratio': preferred_skills_match_ratio,
                'total_skills_match_ratio': total_skills_match_ratio,
                'experience_ratio': experience_ratio,
                'experience_difference': experience_difference,
                'experience_match': experience_match,
                'education_match': education_match,
                'education_difference': education_difference,
                'jaccard_similarity': jaccard_similarity,
                'bigram_overlap': bigram_overlap,
                'trigram_overlap': trigram_overlap,
                'role_similarity': role_similarity,
                'certification_match': certification_match,
                'resume_experience_years': resume_experience,
                'job_min_experience': job_min_experience,
                'resume_education_level': resume_education,
                'job_education_level': job_education
            }
            
            features.append(feature)
            
            # Calculate match label (score from 0 to 100)
            if calculate_match_label:
                # Weighted scoring formula
                # 50% skills, 20% experience, 15% education, 10% text similarity, 5% role similarity
                skills_score = total_skills_match_ratio * 50
                experience_score = (experience_match * 10) + (min(experience_ratio, 2.0) / 2.0 * 10)
                education_score = education_match * 15
                text_similarity_score = (jaccard_similarity * 5) + (bigram_overlap * 3) + (trigram_overlap * 2)
                role_score = role_similarity * 5
                
                match_score = skills_score + experience_score + education_score + text_similarity_score + role_score
                labels.append(match_score)
        
        # Convert to DataFrame
        self.features_df = pd.DataFrame(features)
        
        if calculate_match_label:
            self.labels = np.array(labels)
            print(f"Created features with {len(features)} samples")
            print(f"Score distribution: Min={min(self.labels):.2f}, Max={max(self.labels):.2f}, Mean={np.mean(self.labels):.2f}")
        else:
            self.labels = None
            print(f"Created features with {len(features)} samples (no labels)")
        
        return self.features_df, self.labels
    
    def vectorize_text(self, train_df=None, test_df=None):
        """Vectorize text features using TF-IDF and add cosine similarity"""
        if train_df is None:
            train_df = self.features_df
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        
        # Fit vectorizer on both resume and job description texts
        resume_texts_train = train_df['resume_text'].tolist()
        job_texts_train = train_df['job_description'].tolist()
        all_texts_train = resume_texts_train + job_texts_train
        
        self.tfidf_vectorizer.fit(all_texts_train)
        
        # Transform resume and job texts separately
        resume_vectors_train = self.tfidf_vectorizer.transform(resume_texts_train)
        job_vectors_train = self.tfidf_vectorizer.transform(job_texts_train)
        
        # Calculate cosine similarity between resume and job vectors
        cosine_similarities_train = []
        for i in range(len(resume_texts_train)):
            similarity = cosine_similarity(resume_vectors_train[i:i+1], job_vectors_train[i:i+1])[0][0]
            cosine_similarities_train.append(similarity)
        
        # Create feature DataFrame without text columns
        train_features = train_df.drop(['resume_text', 'job_description'], axis=1).copy()
        train_features['cosine_similarity'] = cosine_similarities_train
        
        # Get feature names
        self.feature_names = train_features.columns.tolist()
        
        # Scale numerical features
        numerical_features = train_features.select_dtypes(include=['float64', 'int64']).columns
        train_features[numerical_features] = self.scaler.fit_transform(train_features[numerical_features])
        
        if test_df is not None:
            # Transform test data
            resume_texts_test = test_df['resume_text'].tolist()
            job_texts_test = test_df['job_description'].tolist()
            
            resume_vectors_test = self.tfidf_vectorizer.transform(resume_texts_test)
            job_vectors_test = self.tfidf_vectorizer.transform(job_texts_test)
            
            # Calculate cosine similarity for test data
            cosine_similarities_test = []
            for i in range(len(resume_texts_test)):
                similarity = cosine_similarity(resume_vectors_test[i:i+1], job_vectors_test[i:i+1])[0][0]
                cosine_similarities_test.append(similarity)
            
            # Create test feature DataFrame
            test_features = test_df.drop(['resume_text', 'job_description'], axis=1).copy()
            test_features['cosine_similarity'] = cosine_similarities_test
            
            # Scale numerical features
            test_features[numerical_features] = self.scaler.transform(test_features[numerical_features])
            
            return train_features, test_features
        
        return train_features
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        if self.features_df is None or self.labels is None:
            raise ValueError("Features and labels not created. Call create_features() first.")
        
        # Split the data
        train_df, test_df, self.y_train, self.y_test = train_test_split(
            self.features_df, self.labels, test_size=test_size, random_state=random_state
        )
        
        # Vectorize text features
        self.X_train, self.X_test = self.vectorize_text(train_df, test_df)
        
        print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type='random_forest'):
        """Train the model"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call split_data() first.")
        
        print(f"Training {model_type} model...")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=150,         # Increased for larger dataset
                max_depth=15,             # Increased to capture more complex patterns
                min_samples_split=15,     # Reduced to allow more splits
                min_samples_leaf=5,       # Reduced to allow smaller leaf nodes
                max_features=0.7,         # Increased feature consideration
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=150,         # Increased for larger dataset
                max_depth=6,              # Increased depth
                learning_rate=0.08,       # Increased learning rate
                min_samples_split=15,     # Reduced to allow more splits
                min_samples_leaf=5,       # Reduced to allow smaller leaf nodes
                subsample=0.85,           # Slightly increased subsample ratio
                max_features=0.7,         # Increased feature consideration
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        
        # Get feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        print("Model training completed")
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model on the test set"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not prepared. Call split_data() first.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 'r--')
        plt.xlabel('Actual Match Score')
        plt.ylabel('Predicted Match Score')
        plt.title('Actual vs Predicted Match Scores')
        plt.tight_layout()
        plt.show()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if not hasattr(self, 'feature_importance'):
            raise ValueError("Feature importance not available. Train a model first.")
        
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path):
        """Save the trained model and vectorizer"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump({
            'model': self.model,
            'vectorizer': self.tfidf_vectorizer,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model and vectorizer"""
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found: {model_path}")
        
        # Load model and vectorizer
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.tfidf_vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        
        print(f"Model loaded from {model_path}")
        return self.model
    
    def predict_match(self, resume_text, job_description, resume_skills=None, job_required_skills=None, 
                     job_preferred_skills=None, resume_experience=0, job_min_experience=0, 
                     resume_education="", job_education="", resume_job_history=None, job_title="",
                     resume_certifications=None):
        """Predict match score between a resume and job description"""
        if self.model is None or self.tfidf_vectorizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess text
        resume_text_clean = self.preprocess_text(resume_text)
        job_description_clean = self.preprocess_text(job_description)
        
        # Initialize empty lists if None
        resume_skills = resume_skills or []
        job_required_skills = job_required_skills or []
        job_preferred_skills = job_preferred_skills or []
        resume_job_history = resume_job_history or []
        resume_certifications = resume_certifications or []
        
        # Convert skills to lowercase for case-insensitive matching
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        required_skills_lower = [skill.lower() for skill in job_required_skills]
        preferred_skills_lower = [skill.lower() for skill in job_preferred_skills]
        
        # Calculate required skills match
        required_skills_matched = sum(1 for skill in required_skills_lower if skill in resume_skills_lower)
        required_skills_match_ratio = required_skills_matched / len(required_skills_lower) if required_skills_lower else 0.0
        
        # Calculate preferred skills match
        preferred_skills_matched = sum(1 for skill in preferred_skills_lower if skill in resume_skills_lower)
        preferred_skills_match_ratio = preferred_skills_matched / len(preferred_skills_lower) if preferred_skills_lower else 0.0
        
        # Calculate total skills match
        all_job_skills = required_skills_lower + preferred_skills_lower
        all_skills_matched = sum(1 for skill in all_job_skills if skill in resume_skills_lower)
        total_skills_match_ratio = all_skills_matched / len(all_job_skills) if all_job_skills else 0.0
        
        # Experience features
        experience_ratio = resume_experience / job_min_experience if job_min_experience > 0 else (1.0 if resume_experience > 0 else 0.0)
        experience_difference = resume_experience - job_min_experience
        experience_match = 1 if resume_experience >= job_min_experience else 0
        
        # Education features
        education_levels = {
            '': 0,
            'High School': 1,
            'Associate\'s': 2,
            'Bachelor\'s': 3,
            'Master\'s': 4,
            'PhD': 5
        }
        
        resume_education_level = education_levels.get(resume_education, 0)
        job_education_level = education_levels.get(job_education, 0)
        education_match = 1 if resume_education_level >= job_education_level else 0
        education_difference = resume_education_level - job_education_level
        
        # Calculate text similarity features
        # TF-IDF vectors
        resume_vector = self.tfidf_vectorizer.transform([resume_text_clean])
        job_vector = self.tfidf_vectorizer.transform([job_description_clean])
        
        # Cosine similarity
        cosine_similarity_score = cosine_similarity(resume_vector, job_vector)[0][0]
        
        # Jaccard similarity
        jaccard_similarity = self.calculate_jaccard_similarity(resume_text_clean, job_description_clean)
        
        # N-gram overlap
        bigram_overlap = self.calculate_ngram_overlap(resume_text_clean, job_description_clean, n=2)
        trigram_overlap = self.calculate_ngram_overlap(resume_text_clean, job_description_clean, n=3)
        
        # Role similarity
        role_similarity = self.calculate_role_similarity(resume_job_history, job_title)
        
        # Certification match
        certification_match = 1 if resume_certifications else 0
        
        # Create feature dictionary
        features = {
            'required_skills_match_ratio': required_skills_match_ratio,
            'preferred_skills_match_ratio': preferred_skills_match_ratio,
            'total_skills_match_ratio': total_skills_match_ratio,
            'experience_ratio': experience_ratio,
            'experience_difference': experience_difference,
            'experience_match': experience_match,
            'education_match': education_match,
            'education_difference': education_difference,
            'jaccard_similarity': jaccard_similarity,
            'bigram_overlap': bigram_overlap,
            'trigram_overlap': trigram_overlap,
            'role_similarity': role_similarity,
            'certification_match': certification_match,
            'resume_experience_years': resume_experience,
            'job_min_experience': job_min_experience,
            'resume_education_level': resume_education_level,
            'job_education_level': job_education_level,
            'cosine_similarity': cosine_similarity_score
        }
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features])
        
        # Scale numerical features
        numerical_features = features_df.select_dtypes(include=['float64', 'int64']).columns
        features_df[numerical_features] = self.scaler.transform(features_df[numerical_features])
        
        # Make prediction
        match_score = self.model.predict(features_df)[0]
        
        # Calculate match components for explanation
        skills_score = total_skills_match_ratio * 50
        experience_score = (experience_match * 10) + (min(experience_ratio, 2.0) / 2.0 * 10)
        education_score = education_match * 15
        text_similarity_score = (jaccard_similarity * 5) + (bigram_overlap * 3) + (trigram_overlap * 2)
        role_score = role_similarity * 5
        
        return {
            'match_score': match_score,
            'skill_match': {
                'required_skills_match_ratio': required_skills_match_ratio,
                'preferred_skills_match_ratio': preferred_skills_match_ratio,
                'total_skills_match_ratio': total_skills_match_ratio,
                'skills_score': skills_score
            },
            'experience_match': {
                'experience_ratio': experience_ratio,
                'experience_difference': experience_difference,
                'meets_minimum': bool(experience_match),
                'experience_score': experience_score
            },
            'education_match': {
                'resume_education': resume_education,
                'job_education': job_education,
                'meets_requirement': bool(education_match),
                'education_score': education_score
            },
            'text_similarity': {
                'cosine_similarity': cosine_similarity_score,
                'jaccard_similarity': jaccard_similarity,
                'bigram_overlap': bigram_overlap,
                'trigram_overlap': trigram_overlap,
                'text_similarity_score': text_similarity_score
            },
            'role_similarity': {
                'similarity': role_similarity,
                'role_score': role_score
            }
        }


if __name__ == "__main__":
    # Example usage
    matcher = ResumeJobMatcher()
    
    # Get current directory for relative paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load data using relative path
    data_path = os.path.join(current_dir, "resume_jd_pairs.json")
    matcher.load_data(data_path)
    
    # Create features
    matcher.create_features()
    
    # Split data
    matcher.split_data(test_size=0.2)
    
    # Train model
    matcher.train_model(model_type='random_forest')
    
    # Evaluate model
    evaluation = matcher.evaluate_model()
    
    # Plot feature importance
    matcher.plot_feature_importance(top_n=15)
    
    # Save model using relative path
    model_path = os.path.join(current_dir, "resume_matcher_model.joblib")
    matcher.save_model(model_path)
    
    print("Model training and evaluation completed.")