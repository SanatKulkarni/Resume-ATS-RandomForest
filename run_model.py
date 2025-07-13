from model import ResumeJobMatcher
import os.path

# Initialize the matcher
matcher = ResumeJobMatcher()

# Path to saved model
model_path = r"./resume_matcher_model.joblib"

# Check if a trained model already exists
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    matcher.load_model(model_path)
else:
    print("No pre-trained model found. Training a new model...")
    # Load data
    matcher.load_data(r"./resume_jd_pairs.json")

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

    # Save model
    matcher.save_model(model_path)
    print(f"Model saved to {model_path}")

# Example prediction
sample_resume = "Experienced software developer with 5 years in Python, Java, and SQL. Bachelor's degree in Computer Science."
sample_job = "Looking for a software engineer with Python skills. Minimum 3 years experience required. Bachelor's degree preferred."

prediction = matcher.predict_match(
    resume_text=sample_resume,
    job_description=sample_job,
    resume_skills=["Python", "Java", "SQL"],
    job_required_skills=["Python"],
    job_preferred_skills=["Java"],
    resume_experience=5,
    job_min_experience=3,
    resume_education="Bachelor's",
    job_education="Bachelor's",
    resume_job_history=["Software Developer", "Junior Programmer"],
    job_title="Software Engineer"
)

print("\nSample Prediction:")
print(f"Match Score: {prediction['match_score']:.2f}/100")
print(f"Skills Match: {prediction['skill_match']['total_skills_match_ratio']:.2f}")
print(f"Experience Match: {'Yes' if prediction['experience_match']['meets_minimum'] else 'No'}")
print(f"Education Match: {'Yes' if prediction['education_match']['meets_requirement'] else 'No'}")

# Add a second example with poor match
print("\n--- Second Example (Poor Match) ---")
poor_match_resume = "Recent marketing graduate with internship experience in social media management. Proficient in Adobe Creative Suite and content creation."
poor_match_job = "Senior software engineer position requiring 8+ years of experience in Java, Python, and cloud technologies. Master's degree in Computer Science preferred."

poor_prediction = matcher.predict_match(
    resume_text=poor_match_resume,
    job_description=poor_match_job,
    resume_skills=["Adobe Photoshop", "Social Media", "Content Writing"],
    job_required_skills=["Java", "Python", "AWS", "Kubernetes"],
    job_preferred_skills=["React", "Node.js", "Docker"],
    resume_experience=1,
    job_min_experience=8,
    resume_education="Bachelor's",
    job_education="Master's",
    resume_job_history=["Marketing Intern", "Social Media Assistant"],
    job_title="Senior Software Engineer"
)

print("\nPoor Match Prediction:")
print(f"Match Score: {poor_prediction['match_score']:.2f}/100")
print(f"Skills Match: {poor_prediction['skill_match']['total_skills_match_ratio']:.2f}")
print(f"Experience Match: {'Yes' if poor_prediction['experience_match']['meets_minimum'] else 'No'}")
print(f"Education Match: {'Yes' if poor_prediction['education_match']['meets_requirement'] else 'No'}")