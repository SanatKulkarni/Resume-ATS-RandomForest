import os
import json
import google.generativeai as genai
from flask import Blueprint, jsonify, request
import time
import random

# Create a directory to store the job description dataset if it doesn't exist
os.makedirs("d:/6TH SEM LABS/ADS/finalProject/jobDescDataset", exist_ok=True)

# Initialize the Gemini API
# You'll need to set your API key
# genai.configure(api_key="YOUR_GEMINI_API_KEY")

job_desc_blueprint = Blueprint('job_desc', __name__)

@job_desc_blueprint.route('/generate-job-descriptions', methods=['POST'])
def generate_job_descriptions():
    try:
        # Get the number of job descriptions to generate from the request, default to 10
        num_jobs = request.json.get('num_jobs', 10)
        
        # Limit the number of job descriptions per request to avoid timeouts
        if num_jobs > 20:
            num_jobs = 20
        
        # The prompt for generating job descriptions
        prompt = """
        # Job Description Dataset Generation Prompt

        Create a synthetic dataset of {num_jobs} tech job descriptions with the following structure:

        ## Job Description Fields
        Each job description should include:
        - `job_id`: Unique identifier (string)
        - `job_title`: Position title (string)
        - `company`: Company name (string)
        - `job_description`: Full text of the job listing (string)
        - `required_skills`: List of 4-8 essential technical skills
        - `preferred_skills`: List of 2-5 nice-to-have skills
        - `min_experience`: Minimum years of experience required (0-10)
        - `education_requirement`: One of ["High School", "Associates", "Bachelor's", "Master's", "PhD"]
        - `job_category`: One of ["Frontend", "Backend", "Full Stack", "Data", "DevOps", "Mobile", "Security", "Management"]
        - `location`: City and remote status (string)
        - `salary_range`: Approximate salary range (string)

        ## Tech Job Titles (use these)
        Software Engineer (Frontend/Backend/Full Stack), DevOps Engineer, Site Reliability Engineer, Data Scientist, Machine Learning Engineer, Data Engineer, Cloud Architect, Security Engineer, QA Engineer, Mobile Developer, UI/UX Developer, Product Manager, Technical Project Manager, Database Administrator, AI Research Engineer

        ## Tech Skills (select from these)
        Programming: Python, Java, JavaScript, TypeScript, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin 
        Frontend: React, Angular, Vue.js, HTML5, CSS3, SASS, jQuery, Bootstrap, Tailwind CSS 
        Backend: Node.js, Django, Flask, Spring, Express.js, ASP.NET, Ruby on Rails, Laravel 
        Database: SQL, MySQL, PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB, Elasticsearch 
        Cloud: AWS, Azure, GCP, Docker, Kubernetes, Terraform, CloudFormation, Lambda 
        Data Science: TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, R, Tableau, Power BI 
        DevOps: Git, Jenkins, CircleCI, GitHub Actions, Ansible, Chef, Puppet, Prometheus 
        Mobile: iOS, Android, React Native, Flutter, Xamarin, Ionic

        ## Company Names (select from these)
        TechCorp, DataSystems, CloudNative, AILabs, WebFront, DevSecOps, FullStack Inc, DataFlow, TechStart, MobilePro, SecurityFirst, CloudScale, AnalyticsHub, FrontierTech, NextGenSystems

        ## Job Description Format
        Generate job_description in a realistic format like:

        ```
        Company: [Company Name]
        Position: [Job Title]
        Location: [City, State or Remote]
        Salary: [Salary Range]

        About Us:
        [2-3 sentences about the company]

        Job Description:
        We are looking for a [Job Title] to join our team. The ideal candidate will have [key skills] and experience with [key technologies].

        Responsibilities:
        - [Responsibility 1]
        - [Responsibility 2]
        - [Responsibility 3]
        - [Responsibility 4]

        Requirements:
        - [Requirement 1]
        - [Requirement 2]
        - [Requirement 3]
        - [Requirement 4]

        Preferred Qualifications:
        - [Preferred Qualification 1]
        - [Preferred Qualification 2]
        - [Preferred Qualification 3]

        Benefits:
        - [Benefit 1]
        - [Benefit 2]
        - [Benefit 3]
        ```

        ## Distribution Guidelines
        - Ensure a balanced mix of job categories and seniority levels
        - Create realistic skill requirements (e.g., Frontend jobs should require HTML/CSS/JavaScript)
        - Vary experience requirements from entry-level (0-1 years) to senior (5+ years)
        - Include both remote and on-site positions
        - Ensure education requirements align with role seniority
        - Create realistic salary ranges based on experience level and role type
        - Include trendy technologies for some positions (e.g., Rust, ML Ops, Edge Computing)
        - Make 20% of jobs have high-demand/difficult-to-fill skill requirements

        Return the data as a JSON array of job description objects.
        """.format(num_jobs=num_jobs)
        
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
        
        # Generate the job descriptions
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
                job_descriptions = json.loads(json_text)
            else:
                # If JSON format not found, try to parse the whole response
                job_descriptions = json.loads(response_text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return an error
            return jsonify({"error": "Failed to parse the generated job descriptions as JSON"}), 500
        
        # Save the job descriptions to files
        timestamp = int(time.time())
        batch_id = f"batch_{timestamp}_{random.randint(1000, 9999)}"
        
        for i, job_desc in enumerate(job_descriptions):
            file_path = f"d:/6TH SEM LABS/ADS/finalProject/jobDescDataset/job_desc_{batch_id}_{i+1}.json"
            with open(file_path, 'w') as f:
                json.dump(job_desc, f, indent=2)
        
        return jsonify({
            "message": f"Successfully generated {len(job_descriptions)} job descriptions",
            "batch_id": batch_id,
            "job_descriptions": job_descriptions
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@job_desc_blueprint.route('/generate-full-dataset', methods=['POST'])
def generate_full_dataset():
    try:
        # Generate 75 job descriptions in batches of 15
        total_job_descriptions = []
        batch_size = 15
        num_batches = 5  # 5 batches of 15 = 75 job descriptions
        
        for i in range(num_batches):
            # The prompt for generating job descriptions
            prompt = """
            # Job Description Dataset Generation Prompt

            Create a synthetic dataset of {batch_size} tech job descriptions with the following structure:

            ## Job Description Fields
            Each job description should include:
            - `job_id`: Unique identifier (string)
            - `job_title`: Position title (string)
            - `company`: Company name (string)
            - `job_description`: Full text of the job listing (string)
            - `required_skills`: List of 4-8 essential technical skills
            - `preferred_skills`: List of 2-5 nice-to-have skills
            - `min_experience`: Minimum years of experience required (0-10)
            - `education_requirement`: One of ["High School", "Associates", "Bachelor's", "Master's", "PhD"]
            - `job_category`: One of ["Frontend", "Backend", "Full Stack", "Data", "DevOps", "Mobile", "Security", "Management"]
            - `location`: City and remote status (string)
            - `salary_range`: Approximate salary range (string)

            ## Tech Job Titles (use these)
            Software Engineer (Frontend/Backend/Full Stack), DevOps Engineer, Site Reliability Engineer, Data Scientist, Machine Learning Engineer, Data Engineer, Cloud Architect, Security Engineer, QA Engineer, Mobile Developer, UI/UX Developer, Product Manager, Technical Project Manager, Database Administrator, AI Research Engineer

            ## Tech Skills (select from these)
            Programming: Python, Java, JavaScript, TypeScript, C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin 
            Frontend: React, Angular, Vue.js, HTML5, CSS3, SASS, jQuery, Bootstrap, Tailwind CSS 
            Backend: Node.js, Django, Flask, Spring, Express.js, ASP.NET, Ruby on Rails, Laravel 
            Database: SQL, MySQL, PostgreSQL, MongoDB, Redis, Cassandra, DynamoDB, Elasticsearch 
            Cloud: AWS, Azure, GCP, Docker, Kubernetes, Terraform, CloudFormation, Lambda 
            Data Science: TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, R, Tableau, Power BI 
            DevOps: Git, Jenkins, CircleCI, GitHub Actions, Ansible, Chef, Puppet, Prometheus 
            Mobile: iOS, Android, React Native, Flutter, Xamarin, Ionic

            ## Company Names (select from these)
            TechCorp, DataSystems, CloudNative, AILabs, WebFront, DevSecOps, FullStack Inc, DataFlow, TechStart, MobilePro, SecurityFirst, CloudScale, AnalyticsHub, FrontierTech, NextGenSystems

            ## Job Description Format
            Generate job_description in a realistic format like:

            ```
            Company: [Company Name]
            Position: [Job Title]
            Location: [City, State or Remote]
            Salary: [Salary Range]

            About Us:
            [2-3 sentences about the company]

            Job Description:
            We are looking for a [Job Title] to join our team. The ideal candidate will have [key skills] and experience with [key technologies].

            Responsibilities:
            - [Responsibility 1]
            - [Responsibility 2]
            - [Responsibility 3]
            - [Responsibility 4]

            Requirements:
            - [Requirement 1]
            - [Requirement 2]
            - [Requirement 3]
            - [Requirement 4]

            Preferred Qualifications:
            - [Preferred Qualification 1]
            - [Preferred Qualification 2]
            - [Preferred Qualification 3]

            Benefits:
            - [Benefit 1]
            - [Benefit 2]
            - [Benefit 3]
            ```

            ## Distribution Guidelines
            - Ensure a balanced mix of job categories and seniority levels
            - Create realistic skill requirements (e.g., Frontend jobs should require HTML/CSS/JavaScript)
            - Vary experience requirements from entry-level (0-1 years) to senior (5+ years)
            - Include both remote and on-site positions
            - Ensure education requirements align with role seniority
            - Create realistic salary ranges based on experience level and role type
            - Include trendy technologies for some positions (e.g., Rust, ML Ops, Edge Computing)
            - Make 20% of jobs have high-demand/difficult-to-fill skill requirements

            Return the data as a JSON array of job description objects.
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
            
            # Generate the job descriptions
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
                    batch_job_descriptions = json.loads(json_text)
                else:
                    # If JSON format not found, try to parse the whole response
                    batch_job_descriptions = json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, continue to the next batch
                continue
            
            # Add batch job descriptions to total
            total_job_descriptions.extend(batch_job_descriptions)
            
            # Save the job descriptions to files
            timestamp = int(time.time())
            batch_id = f"batch_{timestamp}_{random.randint(1000, 9999)}"
            
            for j, job_desc in enumerate(batch_job_descriptions):
                file_path = f"d:/6TH SEM LABS/ADS/finalProject/jobDescDataset/job_desc_{batch_id}_{j+1}.json"
                with open(file_path, 'w') as f:
                    json.dump(job_desc, f, indent=2)
            
            # Sleep to avoid rate limiting
            time.sleep(2)
        
        # Save the complete dataset
        complete_dataset_path = f"d:/6TH SEM LABS/ADS/finalProject/jobDescDataset/complete_dataset.json"
        with open(complete_dataset_path, 'w') as f:
            json.dump(total_job_descriptions, f, indent=2)
        
        return jsonify({
            "message": f"Successfully generated {len(total_job_descriptions)} job descriptions",
            "total_job_descriptions": len(total_job_descriptions)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# This function can be used to register the blueprint with a Flask app
def register_job_desc_blueprint(app):
    app.register_blueprint(job_desc_blueprint, url_prefix='/api/job-desc')