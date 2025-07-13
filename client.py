import streamlit as st
import json
from model import ResumeJobMatcher
import os

# Initialize the matcher
@st.cache_resource
def load_matcher():
    matcher = ResumeJobMatcher()
    model_path = r"./resume_matcher_model.joblib"
    if os.path.exists(model_path):
        matcher.load_model(model_path)
    return matcher

def main():
    st.title("Resume-Job Matcher")
    st.write("Enter resume and job details to check the match score")

    # Create tabs
    tab1, tab2 = st.tabs(["Match Analysis", "Model Statistics"])

    with tab1:
        # Resume Details
        st.subheader("Resume Details")
        resume_text = st.text_area("Resume Text", height=150)
        resume_skills = st.text_input("Skills (comma-separated)")
        resume_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=0)
        resume_education = st.selectbox("Education Level", 
            ["", "High School", "Associate's", "Bachelor's", "Master's", "PhD"])
        resume_job_history = st.text_input("Previous Job Titles (comma-separated)")

        # Job Details
        st.subheader("Job Details")
        job_description = st.text_area("Job Description", height=150)
        job_required_skills = st.text_input("Required Skills (comma-separated)")
        job_preferred_skills = st.text_input("Preferred Skills (comma-separated)")
        job_min_experience = st.number_input("Minimum Years Required", min_value=0, max_value=50, value=0)
        job_education = st.selectbox("Required Education", 
            ["", "High School", "Associate's", "Bachelor's", "Master's", "PhD"])
        job_title = st.text_input("Job Title")

        if st.button("Calculate Match"):
            matcher = load_matcher()

            # Process inputs
            resume_skills_list = [s.strip() for s in resume_skills.split(",") if s.strip()]
            job_required_skills_list = [s.strip() for s in job_required_skills.split(",") if s.strip()]
            job_preferred_skills_list = [s.strip() for s in job_preferred_skills.split(",") if s.strip()]
            resume_job_history_list = [s.strip() for s in resume_job_history.split(",") if s.strip()]

            # Get prediction
            prediction = matcher.predict_match(
                resume_text=resume_text,
                job_description=job_description,
                resume_skills=resume_skills_list,
                job_required_skills=job_required_skills_list,
                job_preferred_skills=job_preferred_skills_list,
                resume_experience=resume_experience,
                job_min_experience=job_min_experience,
                resume_education=resume_education,
                job_education=job_education,
                resume_job_history=resume_job_history_list,
                job_title=job_title
            )

            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Match Score", f"{prediction['match_score']:.1f}%")
            
            with col2:
                st.metric("Skills Match", f"{prediction['skill_match']['total_skills_match_ratio']*100:.1f}%")
            
            with col3:
                experience_match = "Yes" if prediction['experience_match']['meets_minimum'] else "No"
                st.metric("Experience Match", experience_match)
            
            with col4:
                education_match = "Yes" if prediction['education_match']['meets_requirement'] else "No"
                st.metric("Education Match", education_match)

            # Detailed Analysis
            st.subheader("Detailed Analysis")
            
            # Skills Analysis
            with st.expander("Skills Analysis"):
                st.write("Required Skills Match:", f"{prediction['skill_match']['required_skills_match_ratio']*100:.1f}%")
                st.write("Preferred Skills Match:", f"{prediction['skill_match']['preferred_skills_match_ratio']*100:.1f}%")
                st.write("Overall Skills Score:", f"{prediction['skill_match']['skills_score']:.1f}/50")

            # Experience Analysis
            with st.expander("Experience Analysis"):
                st.write("Experience Ratio:", f"{prediction['experience_match']['experience_ratio']:.1f}")
                st.write("Experience Difference:", f"{prediction['experience_match']['experience_difference']} years")
                st.write("Experience Score:", f"{prediction['experience_match']['experience_score']:.1f}/20")

            # Text Similarity Analysis
            with st.expander("Text Similarity Analysis"):
                st.write("Cosine Similarity:", f"{prediction['text_similarity']['cosine_similarity']:.3f}")
                st.write("Jaccard Similarity:", f"{prediction['text_similarity']['jaccard_similarity']:.3f}")
                st.write("Text Similarity Score:", f"{prediction['text_similarity']['text_similarity_score']:.1f}/10")

    with tab2:
        st.subheader("Model Information")
        st.write("This model was trained on a dataset of resume-job pairs and uses the following components for scoring:")
        st.write("- Skills Match (50%): Evaluates both required and preferred skills")
        st.write("- Experience Match (20%): Considers years of experience and job requirements")
        st.write("- Education Match (15%): Compares education levels")
        st.write("- Text Similarity (10%): Analyzes resume and job description content")
        st.write("- Role Similarity (5%): Compares job titles and history")

if __name__ == "__main__":
    main()