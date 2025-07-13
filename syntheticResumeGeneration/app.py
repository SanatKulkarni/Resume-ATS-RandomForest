from flask import Flask
from resumeGen import register_resume_blueprint
from jobDescGen import register_job_desc_blueprint
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure Gemini API with your API key
genai.configure(api_key="AIzaSyAcYzktRYEKcadLvgYX07ayDg3T4JCGK6w")

# Register both blueprints
register_resume_blueprint(app)
register_job_desc_blueprint(app)  # Make sure this line is present

if __name__ == '__main__':
    app.run(debug=True, port=5000)