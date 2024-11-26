# Personalized Survey Generator

This script generates a personalized survey based on user responses using OpenAI's GPT models and the Qualtrics API. It can be used to make personalized questionairres for psychological assessment through prompt engineering. 

This specific case was to assess preferences for certain variations of activities that we have prior information of preferences for through the user responses. It extracts activities from the user's input, creates pairs of related activities, formulates survey questions, and constructs a survey in Qualtrics. The survey link is then provided for distribution.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Environment Variables](#environment-variables)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Activity Extraction**: Uses GPT models to extract activities from user responses.
- **Activity Pair Generation**: Creates pairs like stressful vs. relaxing and social vs. solitary versions of activities.
- **Survey Question Creation**: Formulates survey questions based on the generated activity pairs.
- **Qualtrics Integration**: Interacts with the Qualtrics API to create and activate surveys, add questions, and generate distribution links.
- **Automated Workflow**: Provides an end-to-end solution from data extraction to survey generation.

## Prerequisites

- **Python**: Version 3.7 or higher.
- **OpenAI Account**: With API access to GPT models (e.g., `gpt-4o-mini` or `gpt-4o`).
- **Qualtrics Account**: With API access and permissions to create and manage surveys.
- **Python Packages**:
  - `langchain_openai`
  - `requests`
  - `python-dotenv`

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the packages manually:

```bash
pip install langchain_openai requests python-dotenv
```

### 4. Set Up Environment Variables

Create a `.env` file in the project directory with the following content:

```dotenv
OPENAI_API_KEY=your_openai_api_key
QUALTRICS_API_TOKEN=your_qualtrics_api_token
```

Replace `your_openai_api_key` and `your_qualtrics_api_token` with your actual API keys.

## Usage

Run the script using:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with the actual name of the Python script (qualsurv.py or personalized_survey.py).

### Example Usage

The script includes an example in the `__main__` block that demonstrates how to generate a survey using sample user responses.

## Environment Variables

Ensure the following environment variables are set:

- **`OPENAI_API_KEY`**: Your OpenAI API key.
- **`QUALTRICS_API_TOKEN`**: Your Qualtrics API token.

These can be set in the `.env` file or directly in your environment.

## Notes

- **OpenAI Model Access**: Confirm that your OpenAI account has access to the GPT model specified in the script.
- **Qualtrics Data Center**: The `base_url` in the `QualtricsAPI` class is set to `https://yul1.qualtrics.com/API/v3`. Update this if your Qualtrics account is hosted in a different data center.
- **API Permissions**: Your Qualtrics API token must have the necessary permissions to create surveys, add questions, and activate surveys via the API.
- **Dependencies**: All required Python packages must be installed. Use the `pip install` commands provided in the setup.

## Troubleshooting

- **Missing Environment Variables**: If you encounter errors about missing environment variables, ensure that the `.env` file is properly configured and that the variable names are correct.
- **Invalid API Keys**: Double-check that your OpenAI and Qualtrics API keys are valid and have not expired.
- **API Errors**: Review the console output for detailed error messages from API responses. Common issues include incorrect payload formats or insufficient permissions.
- **Survey Activation Issues**: If the survey fails to activate, verify that the `activate_survey` method uses the correct endpoint (`/surveys/{surveyId}`) and payload (`{"isActive": True}`).

## License

This project is licensed under the MIT License.

---

If you encounter any issues, refer to the troubleshooting section or consult the relevant API documentation for OpenAI and Qualtrics.