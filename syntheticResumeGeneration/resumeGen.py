import os
import json
import google.generativeai as genai
from flask import Blueprint, jsonify, request
import time
import random

# Create a directory to store the resume dataset if it doesn't exist
os.makedirs("d:/6TH SEM LABS/ADS/finalProject/resumeDataset", exist_ok=True)

# Initialize the Gemini API
# You'll need to set your API key
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

resume_blueprint = Blueprint('resume', __name__)

@resume_blueprint.route('/generate-resumes', methods=['POST'])
def generate_resumes():
    try:
        # Get the number of resumes to generate from the request, default to 10
        num_resumes = request.json.get('num_resumes', 10)
        
        # Limit the number of resumes per request to avoid timeouts
        if num_resumes > 20:
            num_resumes = 20
        
        # The prompt for generating resumes
        prompt = """
        Resume Dataset Generation Prompt
        Create a synthetic dataset of {num_resumes} tech resumes with the following structure:
        Resume Fields
        Each resume should include:
        resume_id: Unique identifier (string)
        resume_text: Full text of the resume (string)
        extracted_skills: List of 5-15 technical skills from the skill list below
        experience_years: Total years of experience (0-15)
        education_level: One of ["High School", "Associates", "Bachelor's", "Master's", "PhD"]
        job_history: List of 1-4 previous job titles
        certifications: List of 0-3 technical certifications
        Tech Skills (select from these)
        Programming: Python, Java, JavaScript, TypeScript, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin Frontend: React, Angular, Vue.js, HTML5, CSS3, SASS, jQuery, Bootstrap, Tailwind CSS Backend: Node.js, Django, Flask, Spring, Express.js, ASP.NET, Ruby on Rails, Laravel Database: SQL, MySQL, PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB, Elasticsearch Cloud: AWS, Azure, GCP, Docker, Kubernetes, Terraform, CloudFormation, Lambda Data Science: TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, R, Tableau, Power BI DevOps: Git, Jenkins, CircleCI, GitHub Actions, Ansible, Chef, Puppet, Prometheus Mobile: iOS, Android, React Native, Flutter, Xamarin, Ionic
        Job Titles (select from these)
        Software Engineer, Frontend Developer, Backend Developer, Full Stack Developer, Mobile Developer, DevOps Engineer, Site Reliability Engineer, Data Scientist, Machine Learning Engineer, Data Engineer, Cloud Architect, Security Engineer, QA Engineer, Product Manager
        Certifications (select from these)
        AWS Certified Solutions Architect, Google Cloud Professional, Microsoft Azure Administrator, Certified Kubernetes Administrator, Certified Scrum Master, CompTIA Security+, Cisco CCNA, Oracle Certified Professional, Certified Ethical Hacker, Salesforce Certified Developer
        Resume Format
        Generate resume_text in a realistic format like:
        NAME: [Random Name]
        EMAIL: [Random Email]
        PHONE: [Random Phone]

        SUMMARY:
        [1-2 sentences about career background and goals]

        SKILLS:
        [List of skills separated by commas]

        EXPERIENCE:
        [Company 1] - [Job Title] ([Start Year] - [End Year])
        - [Bullet point about responsibility/achievement]
        - [Bullet point about responsibility/achievement]

        [Company 2] - [Job Title] ([Start Year] - [End Year])
        - [Bullet point about responsibility/achievement]
        - [Bullet point about responsibility/achievement]

        EDUCATION:
        [University Name] - [Degree Type] in [Field] ([Year])

        CERTIFICATIONS:
        [List of certifications if any]

        Distribution Guidelines
        Ensure varied skill combinations for different job types
        Create realistic correlations (e.g., Data Scientists should have data skills)
        Vary experience levels from entry-level to senior
        Include both specialized and generalist profiles
        Ensure 20% of profiles have certifications relevant to their field

        Return the data as a JSON array of resume objects.
        """.format(num_resumes=num_resumes)
        
        # Configure the model
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8192,
        }
        
        # Initialize the Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=generation_config
        )
        
        # Generate the resumes
        response = model.generate_content(prompt)
        
        # Parse the response to get the JSON data
        try:
            # Extract the text from the response
            response_text = response.text
            
            # Find the JSON part of the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx:end_idx]
                resumes = json.loads(json_text)
            else:
                # If JSON format not found, try to parse the whole response
                resumes = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return an error
            return jsonify({"error": "Failed to parse the generated resumes as JSON"}), 500
        
        # Save the resumes to files
        timestamp = int(time.time())
        batch_id = f"batch_{timestamp}_{random.randint(1000, 9999)}"
        
        for i, resume in enumerate(resumes):
            file_path = f"d:/6TH SEM LABS/ADS/finalProject/resumeDataset/resume_{batch_id}_{i+1}.json"
            with open(file_path, 'w') as f:
                json.dump(resume, f, indent=2)
        
        return jsonify({
            "message": f"Successfully generated {len(resumes)} resumes",
            "batch_id": batch_id,
            "resumes": resumes
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@resume_blueprint.route('/generate-full-dataset', methods=['POST'])
def generate_full_dataset():
    try:
        # Generate 100 resumes in batches of 10
        total_resumes = []
        batch_size = 10
        num_batches = 10  # 10 batches of 10 = 100 resumes
        
        for i in range(num_batches):
            # The prompt for generating resumes
            prompt = """
            Resume Dataset Generation Prompt
            Create a synthetic dataset of {batch_size} tech resumes with the following structure:
            Resume Fields
            Each resume should include:
            resume_id: Unique identifier (string)
            resume_text: Full text of the resume (string)
            extracted_skills: List of 5-15 technical skills from the skill list below
            experience_years: Total years of experience (0-15)
            education_level: One of ["High School", "Associates", "Bachelor's", "Master's", "PhD"]
            job_history: List of 1-4 previous job titles
            certifications: List of 0-3 technical certifications
            Tech Skills (select from these)
            Programming: Python, Java, JavaScript, TypeScript, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin Frontend: React, Angular, Vue.js, HTML5, CSS3, SASS, jQuery, Bootstrap, Tailwind CSS Backend: Node.js, Django, Flask, Spring, Express.js, ASP.NET, Ruby on Rails, Laravel Database: SQL, MySQL, PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB, Elasticsearch Cloud: AWS, Azure, GCP, Docker, Kubernetes, Terraform, CloudFormation, Lambda Data Science: TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, R, Tableau, Power BI DevOps: Git, Jenkins, CircleCI, GitHub Actions, Ansible, Chef, Puppet, Prometheus Mobile: iOS, Android, React Native, Flutter, Xamarin, Ionic
            Job Titles (select from these)
            Software Engineer, Frontend Developer, Backend Developer, Full Stack Developer, Mobile Developer, DevOps Engineer, Site Reliability Engineer, Data Scientist, Machine Learning Engineer, Data Engineer, Cloud Architect, Security Engineer, QA Engineer, Product Manager
            Certifications (select from these)
            AWS Certified Solutions Architect, Google Cloud Professional, Microsoft Azure Administrator, Certified Kubernetes Administrator, Certified Scrum Master, CompTIA Security+, Cisco CCNA, Oracle Certified Professional, Certified Ethical Hacker, Salesforce Certified Developer
            Resume Format
            Generate resume_text in a realistic format like:
            NAME: [Random Name]
            EMAIL: [Random Email]
            PHONE: [Random Phone]

            SUMMARY:
            [1-2 sentences about career background and goals]

            SKILLS:
            [List of skills separated by commas]

            EXPERIENCE:
            [Company 1] - [Job Title] ([Start Year] - [End Year])
            - [Bullet point about responsibility/achievement]
            - [Bullet point about responsibility/achievement]

            [Company 2] - [Job Title] ([Start Year] - [End Year])
            - [Bullet point about responsibility/achievement]
            - [Bullet point about responsibility/achievement]

            EDUCATION:
            [University Name] - [Degree Type] in [Field] ([Year])

            CERTIFICATIONS:
            [List of certifications if any]

            Distribution Guidelines
            Ensure varied skill combinations for different job types
            Create realistic correlations (e.g., Data Scientists should have data skills)
            Vary experience levels from entry-level to senior
            Include both specialized and generalist profiles
            Ensure 20% of profiles have certifications relevant to their field

            Return the data as a JSON array of resume objects.
            """.format(batch_size=batch_size)
            
            # Configure the model
            generation_config = {
                "temperature": 0.9,
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 8192,
            }
            
            # Initialize the Gemini model
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            # Generate the resumes
            response = model.generate_content(prompt)
            
            # Parse the response to get the JSON data
            try:
                # Extract the text from the response
                response_text = response.text
                
                # Find the JSON part of the response
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_text = response_text[start_idx:end_idx]
                    batch_resumes = json.loads(json_text)
                else:
                    # If JSON format not found, try to parse the whole response
                    batch_resumes = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, continue to the next batch
                continue
            
            # Add batch resumes to total
            total_resumes.extend(batch_resumes)
            
            # Save the resumes to files
            timestamp = int(time.time())
            batch_id = f"batch_{timestamp}_{random.randint(1000, 9999)}"
            
            for j, resume in enumerate(batch_resumes):
                file_path = f"d:/6TH SEM LABS/ADS/finalProject/resumeDataset/resume_{batch_id}_{j+1}.json"
                with open(file_path, 'w') as f:
                    json.dump(resume, f, indent=2)
            
            # Sleep to avoid rate limiting
            time.sleep(2)
        
        # Save the complete dataset
        complete_dataset_path = f"d:/6TH SEM LABS/ADS/finalProject/resumeDataset/complete_dataset.json"
        with open(complete_dataset_path, 'w') as f:
            json.dump(total_resumes, f, indent=2)
        
        return jsonify({
            "message": f"Successfully generated {len(total_resumes)} resumes",
            "total_resumes": len(total_resumes)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# This function can be used to register the blueprint with a Flask app
def register_resume_blueprint(app):
    app.register_blueprint(resume_blueprint, url_prefix='/api/resume')