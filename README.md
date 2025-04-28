# GenePattern Copilot

This is the Django backend for the GenePattern Copilot application. It provides a RESTful API for interacting with 
various Large Language Models (LLMs) using LangChain and LangGraph.

## Features

* Django 5.2 backend
* Django REST Framework for API endpoints
* CORS support for frontend integration
* Integration with LangChain and LangGraph
* Support for OpenAI, Gemini, Llama and other models
* Database models for tracking conversations, queries, ratings, prompts and LLM steps.
* User response rating (thumbs up/down).

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/genepattern/GPCopilot.git
    cd GPCopilot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    
    *Alternatively, you can use Conda:*

    ```bash
    conda create -n gp_copilot python=3.11
    conda activate gp_copilot
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    Create a `.env` file in the project root directory and add your LLM API keys and other sensitive settings:

    ```dotenv
    # General Django Settings
    SECRET_KEY='your-strong-django-secret-key' # Generate a real one!
    DEBUG=True # Set to False in production
    ALLOWED_HOSTS=localhost,127.0.0.1 # Add your production domain

    # Database (Default: SQLite)
    # DATABASE_URL=sqlite:///db.sqlite3

    # LLM API Keys
    OPENAI_API_KEY='your_openai_api_key'
    GOOGLE_GEMINI_API_KEY='your_google_api_key'
    AWS_ACCESS_KEY_ID='your_aws_access_key'
    AWS_SECRET_ACCESS_KEY='your_aws_secret_key'
    # Add other keys/credentials as needed for any other models

    # CORS Settings (adjust for production)
    CORS_ALLOWED_ORIGINS=http://localhost:3000
    ```

5.  **Apply database migrations:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

6.  **Create a superuser (optional, for admin access):**
    ```bash
    python manage.py createsuperuser
    ```

7.  **Run the development server:**
    ```bash
    python manage.py runserver
    ```

The application will be available at `http://127.0.0.1:8000/`.
