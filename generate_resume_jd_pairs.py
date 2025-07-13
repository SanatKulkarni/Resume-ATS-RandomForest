import os
import json
import random
import pandas as pd
from tqdm import tqdm

def load_json_files(directory):
    """Load all JSON files from a directory into a list of dictionaries."""
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    data.append(json_data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return data

def generate_pairs(resumes, job_descriptions, num_pairs=10000, random_seed=42):
    """Generate pairs of resumes and job descriptions."""
    random.seed(random_seed)
    
    # Calculate total possible pairs
    total_possible_pairs = len(resumes) * len(job_descriptions)
    print(f"Total possible pairs: {total_possible_pairs}")
    
    # Adjust num_pairs if it exceeds total possible pairs
    num_pairs = min(num_pairs, total_possible_pairs)
    print(f"Generating {num_pairs} pairs...")
    
    pairs = []
    
    # Generate random pairs
    for _ in tqdm(range(num_pairs)):
        resume = random.choice(resumes)
        job_desc = random.choice(job_descriptions)
        
        # Create a pair with relevant information
        pair = {
            "resume_id": resume.get("resume_id", ""),
            "resume_text": resume.get("resume_text", ""),
            "resume_skills": resume.get("extracted_skills", []),
            "resume_experience_years": resume.get("experience_years", 0),
            "resume_education_level": resume.get("education_level", ""),
            "job_id": job_desc.get("job_id", ""),
            "job_title": job_desc.get("job_title", ""),
            "job_description": job_desc.get("job_description", ""),
            "job_required_skills": job_desc.get("required_skills", []),
            "job_preferred_skills": job_desc.get("preferred_skills", []),
            "job_min_experience": job_desc.get("min_experience_years", 0),
            "job_required_education": job_desc.get("required_education", "")
        }
        
        # Calculate match features (you can expand this)
        pair["skill_match_ratio"] = calculate_skill_match(
            resume.get("extracted_skills", []),
            job_desc.get("required_skills", []) + job_desc.get("preferred_skills", [])
        )
        
        pair["experience_match"] = 1 if resume.get("experience_years", 0) >= job_desc.get("min_experience_years", 0) else 0
        
        pairs.append(pair)
    
    return pairs

def calculate_skill_match(resume_skills, job_skills):
    """Calculate the ratio of matching skills."""
    if not job_skills:
        return 0.0
    
    # Convert to lowercase for case-insensitive matching
    resume_skills_lower = [skill.lower() for skill in resume_skills]
    job_skills_lower = [skill.lower() for skill in job_skills]
    
    # Count matching skills
    matching_skills = sum(1 for skill in job_skills_lower if skill in resume_skills_lower)
    
    # Calculate match ratio
    return matching_skills / len(job_skills)

def save_pairs(pairs, output_file):
    """Save pairs to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {output_file}")

def main():
    # Paths to your datasets
    resume_dir = r"./syntheticResumeGeneration/dataset/resumeDataset"
    jd_dir = r"./syntheticResumeGeneration/dataset/jobDescDataset"
    output_file = r"./resume_jd_pairs.json"
    
    # Load data
    print("Loading resumes...")
    resumes = load_json_files(resume_dir)
    print(f"Loaded {len(resumes)} resumes")
    
    print("Loading job descriptions...")
    job_descriptions = load_json_files(jd_dir)
    print(f"Loaded {len(job_descriptions)} job descriptions")
    
    # Generate pairs
    pairs = generate_pairs(resumes, job_descriptions, num_pairs=20000)
    
    # Save pairs
    save_pairs(pairs, output_file)
    
    # Create a small sample for inspection
    sample_pairs = random.sample(pairs, min(5, len(pairs)))
    print("\nSample pairs:")
    for pair in sample_pairs:
        print(f"Resume ID: {pair['resume_id']}, Job ID: {pair['job_id']}")
        print(f"Skill Match: {pair['skill_match_ratio']:.2f}, Experience Match: {pair['experience_match']}")
        print("-" * 50)

if __name__ == "__main__":
    main()