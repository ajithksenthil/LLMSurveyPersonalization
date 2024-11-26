import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import re
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class QualtricsAPI:
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.base_url = "https://yul1.qualtrics.com/API/v3"
        self.headers = {
            "X-API-TOKEN": api_token,
            "Content-Type": "application/json"
        }

    def create_survey(self, name: str) -> dict:
        url = f"{self.base_url}/survey-definitions"
        payload = {
            "SurveyName": name,
            "Language": "EN",
            "ProjectCategory": "CORE"
        }
        response = requests.post(url, json=payload, headers=self.headers)
        print(f"Create survey response: {response.status_code}, {response.text}")
        return response.json()

    def activate_survey(self, survey_id: str) -> bool:
        """Activate a survey to make it available for responses."""
        url = f"{self.base_url}/surveys/{survey_id}"
        payload = {
            "isActive": True
        }
        try:
            response = requests.put(url, json=payload, headers=self.headers)
            print(f"Activate survey response: {response.status_code}, {response.text}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error activating survey: {str(e)}")
            return False

    def add_question(self, survey_id: str, question_payload: dict) -> dict:
        url = f"{self.base_url}/survey-definitions/{survey_id}/questions"
        response = requests.post(url, json=question_payload, headers=self.headers)
        print(f"Add question response: {response.status_code}, {response.text}")
        return response.json()

    def distribute_survey(self, survey_id: str, distribution_name: str) -> dict:
        # Generate anonymous link directly
        base = self.base_url.replace("/API/v3", "")
        return {
            "result": {
                "id": None,
                "link": f"{base}/jfe/form/{survey_id}"
            }
        }

    def get_distribution_link(self, survey_id: str, distribution_id: str = None) -> str:
        base = self.base_url.replace("/API/v3", "")
        return f"{base}/jfe/form/{survey_id}"

def validate_environment():
    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "QUALTRICS_API_TOKEN": os.getenv("QUALTRICS_API_TOKEN")
    }
    
    missing = [k for k, v in required_vars.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# Initialize the OpenAI LLM via LangChain
validate_environment()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Use "gpt-4" if you have access
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_kwargs={"max_tokens": 1500},
)

# Initialize Qualtrics API
qualtrics = QualtricsAPI(api_token=os.getenv("QUALTRICS_API_TOKEN"))

# Define the prompts
extract_activities_prompt = """
Given the following user responses to various open-ended questions about anxiety, relaxation, loneliness, and methods to reduce loneliness, extract a list of unique, short, and concise activities mentioned by the user. Each activity name should be no longer than 3-5 words.

### User Responses:

{responses}

### Extracted Activities:
- 
"""

generate_stress_relax_pairs_prompt = """
Given the activity "{activity}", provide a **more stressful** version and a **more relaxed** version of that same activity. Ensure both versions are clearly related to the original activity. Each activity name should be short and concise, ideally no longer than 3-5 words.

### Activity Pair:
- Relaxing Activity: {activity}
- Stressful Activity: 
"""

generate_social_solitary_pairs_prompt = """
Given the activity "{activity}", provide a **solitary** version and a **social** version of that same activity. Ensure both versions are clearly related to the original activity. Each activity name should be short and concise, ideally no longer than 3-5 words.

### Activity Pair:
- Solitary Activity: {activity}
- Social Activity: 
"""

create_survey_question_prompt = """
Given two versions of an activity, create a clear and concise survey question asking the respondent to choose between them.

### Activity Pair:
- Option A: {option_a}
- Option B: {option_b}

### Survey Question:
Which of the following would you prefer?
A) {option_a}
B) {option_b}
"""

convert_pair_to_json_prompt = """
You are given an activity pair in text format. Extract the two activities and output them in **valid JSON format** with keys "Option_A" and "Option_B". Ensure the JSON is correctly formatted.

### Activity Pair:

{pair_text}

### Example Input:

- Relaxing Activity: Reading a book
- Stressful Activity: Giving a speech

### Example JSON Output:

{{
    "Option_A": "Reading a book",
    "Option_B": "Giving a speech"
}}

### Your JSON Output:
"""

# Initialize Prompt Templates and Chains
extract_activities_template = PromptTemplate(input_variables=["responses"], template=extract_activities_prompt)
generate_stress_relax_template = PromptTemplate(input_variables=["activity"], template=generate_stress_relax_pairs_prompt)
generate_social_solitary_template = PromptTemplate(input_variables=["activity"], template=generate_social_solitary_pairs_prompt)
create_survey_question_template = PromptTemplate(input_variables=["option_a", "option_b"], template=create_survey_question_prompt)
convert_pair_to_json_template = PromptTemplate(input_variables=["pair_text"], template=convert_pair_to_json_prompt)

extract_activities_chain = LLMChain(llm=llm, prompt=extract_activities_template, output_key="activities")
generate_stress_relax_chain = LLMChain(llm=llm, prompt=generate_stress_relax_template, output_key="stress_relax_pair")
generate_social_solitary_chain = LLMChain(llm=llm, prompt=generate_social_solitary_template, output_key="social_solitary_pair")
create_survey_question_chain = LLMChain(llm=llm, prompt=create_survey_question_template, output_key="survey_question")
convert_pair_to_json_chain = LLMChain(llm=llm, prompt=convert_pair_to_json_template, output_key="json_output")

def ensure_conciseness(activity: str, max_words: int = 5) -> str:
    words = activity.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return activity

def extract_activities(responses: str) -> List[str]:
    output = extract_activities_chain.invoke({"responses": responses})
    activities = output["activities"].split("\n")
    activities = [line.strip("- ").strip() for line in activities if line.strip("- ").strip()]
    seen = set()
    unique_activities = []
    for activity in activities:
        if activity.lower() not in seen:
            seen.add(activity.lower())
            unique_activities.append(activity)
    return unique_activities

def convert_pair_to_json(pair_text: str) -> Dict[str, str]:
    try:
        data = json.loads(pair_text)
        if "Option_A" in data and "Option_B" in data:
            data["Option_A"] = ensure_conciseness(data["Option_A"])
            data["Option_B"] = ensure_conciseness(data["Option_B"])
            return data
    except json.JSONDecodeError:
        pass

    try:
        output = convert_pair_to_json_chain.invoke({"pair_text": pair_text})
        output_text = output.get("json_output", "")

        json_str = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group(0))
            data["Option_A"] = ensure_conciseness(data.get("Option_A", ""))
            data["Option_B"] = ensure_conciseness(data.get("Option_B", ""))
            return data

        lines = output_text.split('\n')
        data = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace('"', '').replace("'", "")
                value = value.strip().strip('"').strip("'")
                if key in ["Option_A", "Option_B"]:
                    data[key] = ensure_conciseness(value)

        if "Option_A" in data and "Option_B" in data:
            return data

        return {
            "Option_A": ensure_conciseness(pair_text.split('\n')[0].strip("- ")),
            "Option_B": ensure_conciseness(pair_text.split('\n')[-1].strip("- "))
        }

    except Exception as e:
        print(f"Error converting pair to JSON: {e}")
        return {"Option_A": "", "Option_B": ""}

def generate_stress_relax_pairs(activity: str) -> Dict[str, str]:
    output = generate_stress_relax_chain.invoke({"activity": activity})
    pair_text = output["stress_relax_pair"]
    return convert_pair_to_json(pair_text)

def generate_social_solitary_pairs(activity: str) -> Dict[str, str]:
    output = generate_social_solitary_chain.invoke({"activity": activity})
    pair_text = output["social_solitary_pair"]
    return convert_pair_to_json(pair_text)

def create_survey_question(option_a: str, option_b: str) -> str:
    output = create_survey_question_chain.invoke({"option_a": option_a, "option_b": option_b})
    return output["survey_question"].strip()

def generate_personalized_survey(responses: str) -> Dict[str, List[str]]:
    try:
        # Extract activities
        activities = extract_activities(responses)
        if not activities:
            raise ValueError("No activities extracted from responses")

        # Initialize lists for storing pairs and questions
        stress_relax_pairs = []
        social_solitary_pairs = []
        survey_questions = []

        # Generate questions for each activity
        for activity in activities:
            try:
                # Generate stress/relax pairs
                stress_relax = generate_stress_relax_pairs(activity)
                if stress_relax.get("Option_A") and stress_relax.get("Option_B"):
                    stress_relax_pairs.append(stress_relax)
                    question = create_survey_question(
                        option_a=stress_relax["Option_A"],
                        option_b=stress_relax["Option_B"]
                    )
                    survey_questions.append(question)

                # Generate social/solitary pairs
                social_solitary = generate_social_solitary_pairs(activity)
                if social_solitary.get("Option_A") and social_solitary.get("Option_B"):
                    social_solitary_pairs.append(social_solitary)
                    question = create_survey_question(
                        option_a=social_solitary["Option_A"],
                        option_b=social_solitary["Option_B"]
                    )
                    survey_questions.append(question)
            except Exception as e:
                print(f"Error processing activity {activity}: {str(e)}")
                continue

        if not survey_questions:
            raise ValueError("No valid survey questions generated")

        # Create survey
        survey_name = "Personalized Activity Preference Survey"

        try:
            survey_response = qualtrics.create_survey(survey_name)
            print(f"Survey creation response: {survey_response}")
            
            # Check for errors in the response
            if "meta" in survey_response and survey_response["meta"].get("httpStatus") != "200 - OK":
                error_msg = survey_response["meta"].get("error", {}).get("errorMessage", "Unknown error")
                raise ValueError(f"Survey creation failed: {error_msg}")
            
            # Validate survey creation response
            result = survey_response.get("result")
            if not result or not result.get("SurveyID"):
                raise ValueError("Failed to create survey - no SurveyID in response")
                
            survey_id = result["SurveyID"]

            # Retrieve default block ID
            survey_details_url = f"{qualtrics.base_url}/survey-definitions/{survey_id}"
            response = requests.get(survey_details_url, headers=qualtrics.headers)
            survey_details = response.json()
            if "result" in survey_details and "Blocks" in survey_details["result"]:
                blocks = survey_details["result"]["Blocks"]
                default_block_id = list(blocks.keys())[0]
            else:
                raise ValueError("Failed to retrieve default block ID")
            block_id = default_block_id
        except Exception as e:
            print(f"Error in survey creation: {str(e)}")
            raise

        # Add questions to survey
        for idx, question_text in enumerate(survey_questions, 1):
            try:
                question_payload = {
                    "QuestionText": question_text,
                    "DataExportTag": f"Q{idx}",
                    "QuestionType": "MC",
                    "Selector": "SAVR",
                    "SubSelector": "TX",
                    "Configuration": {
                        "QuestionDescriptionOption": "UseText"
                    },
                    "Choices": {
                        "1": {"Display": "Option A"},
                        "2": {"Display": "Option B"}
                    },
                    "Validation": {
                        "Settings": {
                            "ForceResponse": "OFF",
                            "Type": "None"
                        }
                    }
                }
                question_response = qualtrics.add_question(survey_id, question_payload)
                print(f"Question added: {question_response}")
                if not question_response.get("result", {}).get("QuestionID"):
                    print(f"Warning: Question may not have been added properly: {question_text}")
            except Exception as e:
                print(f"Error adding question: {str(e)}")
                continue

        # Activate the survey
        activation_success = qualtrics.activate_survey(survey_id)
        if activation_success:
            print(f"Survey {survey_id} activated successfully.")
        else:
            raise ValueError("Failed to activate survey")

        # Get survey link
        distribution_response = qualtrics.distribute_survey(
            survey_id=survey_id,
            distribution_name="Personalized Activity Survey Distribution"
        )
        
        survey_link = distribution_response["result"]["link"]
        print(f"Generated survey link: {survey_link}")
        
        if not survey_link:
            raise ValueError("Failed to generate survey link")
        
        return {
            "Stressful_vs_Relaxing_Pairs": stress_relax_pairs,
            "Solitary_vs_Social_Pairs": social_solitary_pairs,
            "Survey_Questions": survey_questions,
            "Survey_Link": survey_link,
            "Survey_ID": survey_id
        }

    except Exception as e:
        print(f"Error generating survey: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    example_responses = """
    anxiety 
    When do you feel anxious? provide 5 situations
    - Public speaking
    - Meeting new people
    - Deadlines at work
    - Taking exams
    - Financial uncertainty
    
    [Rest of your example responses...]
    """
    
    try:
        survey = generate_personalized_survey(example_responses)
        print("\n### Personalized Survey ###\n")
        for idx, question in enumerate(survey["Survey_Questions"], 1):
            print(f"{idx}. {question}\n")
        print(f"Survey Link: {survey['Survey_Link']}\n")
    except Exception as e:
        print(f"Failed to generate survey: {str(e)}")
