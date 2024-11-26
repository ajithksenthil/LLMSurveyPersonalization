
# personalized_survey.py
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI # use the package you want
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, SecretStr
import json
import re


# Specify the desired number of survey questions here
NUM_QUESTIONS = 10  # Change this value to set the number of questions


# Ensure you have set your OpenAI API key as an environment variable
# You can set it in your environment or directly assign it here
# Example:
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
OPENAI_API_KEY = "your-api-key-here"
# Initialize the OpenAI LLM via LangChain, change this based on the model you are using
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Corrected model name, choose the model you want
    temperature=0.3,
    max_tokens=1500,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_API_KEY,  # Use environment variable or assign directly
    # base_url="...",
    # organization="...",
    # other params...
)

# Define the prompts
extract_activities_prompt = """
You are a helpful assistant that extracts activities from user responses.

Given the following user responses to various open-ended questions about anxiety, relaxation, loneliness, and methods to reduce loneliness, extract a list of UNIQUE, short, and concise activities that based on the user's responses the user would likely enjoy. They NEED to be similar activities to those mentioned. Each activity name should be no longer than 3-5 words.

### User Responses:

{responses}

### Extracted Activities:
- 
"""

generate_stress_relax_pairs_prompt = """
You are a creative assistant that generates two concise variants of the **SAME** activity: one more stressful and one more relaxed. Ensure that both variants are clearly related to the original activity and do not include numbering or bullet points.

Given the activity "{activity}", provide:
- **Option_A**: A more stressful version of the activity.
- **Option_B**: A more relaxed version of the activity.

Each activity name should be short and concise, ideally no longer than 3-5 words.

### Activity Pair:
- Option_A: 
- Option_B: 
"""

generate_social_solitary_pairs_prompt = """
You are a creative assistant that generates two concise variants of the **SAME** activity: one solitary and one social. Ensure that both variants are clearly related to the original activity and do not include numbering or bullet points.

Given the activity "{activity}", provide:
- **Option_A**: A solitary version of the activity.
- **Option_B**: A social version of the activity.

Each activity name should be short and concise, ideally no longer than 3-5 words.

### Activity Pair:
- Option_A: 
- Option_B: 
"""

create_survey_question_prompt = """
You are a survey designer. Given two versions of an activity, create a clear and concise survey question asking the respondent to choose between them.

### Activity Pair:
- Option A: {option_a}
- Option B: {option_b}

### Survey Question:
Which of the following would you prefer?
A) {option_a}
B) {option_b}
"""

convert_pair_to_json_prompt = """
You are an assistant that converts activity pairs into a structured JSON format.

Given the following activity pair:

{pair_text}

### JSON Output:
{
    "Option_A": "",
    "Option_B": ""
}
"""

# Initialize Prompt Templates
extract_activities_template = PromptTemplate(
    input_variables=["responses"],
    template=extract_activities_prompt
)

generate_stress_relax_template = PromptTemplate(
    input_variables=["activity"],
    template=generate_stress_relax_pairs_prompt
)

generate_social_solitary_template = PromptTemplate(
    input_variables=["activity"],
    template=generate_social_solitary_pairs_prompt
)

create_survey_question_template = PromptTemplate(
    input_variables=["option_a", "option_b"],
    template=create_survey_question_prompt
)

convert_pair_to_json_template = PromptTemplate(
    input_variables=["pair_text"],
    template=convert_pair_to_json_prompt
)

# Initialize Chains
extract_activities_chain = LLMChain(
    llm=llm,
    prompt=extract_activities_template,
    output_key="activities"
)

generate_stress_relax_chain = LLMChain(
    llm=llm,
    prompt=generate_stress_relax_template,
    output_key="stress_relax_pair"
)

generate_social_solitary_chain = LLMChain(
    llm=llm,
    prompt=generate_social_solitary_template,
    output_key="social_solitary_pair"
)

create_survey_question_chain = LLMChain(
    llm=llm,
    prompt=create_survey_question_template,
    output_key="survey_question"
)

convert_pair_to_json_chain = LLMChain(
    llm=llm,
    prompt=convert_pair_to_json_template,
    output_key="json_output"
)

def extract_activities(responses: str) -> List[str]:
    """Extracts a list of unique activities from user responses."""
    output = extract_activities_chain.invoke({"responses": responses})
    activities = output["activities"].split("\n")
    activities = [re.sub(r'^\d+\.\s*', '', line.strip("- ").strip()) for line in activities if line.strip("- ").strip()]
    seen = set()
    unique_activities = []
    for activity in activities:
        activity_clean = activity.lower()
        if activity_clean not in seen:
            seen.add(activity_clean)
            unique_activities.append(activity)
    return unique_activities


def ensure_conciseness(activity: str, max_words: int = 10) -> str:
    """Ensures that the activity name is concise, limiting it to a maximum number of words."""
    words = activity.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return activity

def convert_pair_to_json(pair_text: str) -> Dict[str, str]:
    """Converts activity pair text to JSON with concise activity names."""
    print("Debug - Input pair_text:", pair_text)  # Debug print

    try:
        # Attempt to parse as JSON directly
        data = json.loads(pair_text)
        if "Option_A" in data and "Option_B" in data:
            # Ensure conciseness
            data["Option_A"] = ensure_conciseness(data["Option_A"])
            data["Option_B"] = ensure_conciseness(data["Option_B"])
            return data
    except json.JSONDecodeError:
        pass  # Continue if not valid JSON

    try:
        # Use LLM to convert pair_text to JSON
        output = convert_pair_to_json_chain.invoke({"pair_text": pair_text})
        print("Debug - LLM output:", output)  # Debug print

        output_text = output.get("json_output", "")

        # Extract JSON using regex
        json_str = re.search(r'\{.*\}', output_text, re.DOTALL)
        if json_str:
            data = json.loads(json_str.group(0))
            # Ensure conciseness
            data["Option_A"] = ensure_conciseness(data.get("Option_A", ""))
            data["Option_B"] = ensure_conciseness(data.get("Option_B", ""))
            return data

        # Fallback: Manually extract options
        option_a_match = re.search(r'Option_A:\s*(.*)', pair_text)
        option_b_match = re.search(r'Option_B:\s*(.*)', pair_text)
        if option_a_match and option_b_match:
            option_a = ensure_conciseness(option_a_match.group(1).strip())
            option_b = ensure_conciseness(option_b_match.group(1).strip())
            return {"Option_A": option_a, "Option_B": option_b}

        # If extraction fails, return empty options
        print("Failed to extract Option_A and Option_B from pair_text.")
        return {"Option_A": "", "Option_B": ""}

    except Exception as e:
        print("Error converting pair to JSON:", e)
        print("Problematic pair_text:", pair_text)
        return {"Option_A": "", "Option_B": ""}

def generate_stress_relax_pairs(activity: str) -> Dict[str, str]:
    """Generates stressful and relaxing versions of an activity and converts to JSON."""
    output = generate_stress_relax_chain.invoke({"activity": activity})
    pair_text = output["stress_relax_pair"]
    # print("Debug - generate_stress_relax_pairs output:", pair_text)  # Debug print
    pair_json = convert_pair_to_json(pair_text)
    return pair_json

def generate_social_solitary_pairs(activity: str) -> Dict[str, str]:
    """Generates solitary and social versions of an activity and converts to JSON."""
    output = generate_social_solitary_chain.invoke({"activity": activity})
    pair_text = output["social_solitary_pair"]
    # print("Debug - generate_social_solitary_pairs output:", pair_text)  # Debug print
    pair_json = convert_pair_to_json(pair_text)
    return pair_json

def create_survey_question(option_a: str, option_b: str) -> str:
    """Creates a survey question given two activity options."""
    output = create_survey_question_chain.invoke({"option_a": option_a, "option_b": option_b})
    # Clean the output by removing any leading/trailing whitespace
    question = output["survey_question"].strip()
    return question

def generate_personalized_survey(responses: str, num_questions: int = None) -> Dict[str, List[str]]:
    """Generates a personalized survey based on user responses."""
    activities = extract_activities(responses)

    if not activities:
        print("No activities extracted from the responses.")
        return {
            "Stressful_vs_Relaxing_Pairs": [],
            "Solitary_vs_Social_Pairs": [],
            "Survey_Questions": []
        }

    # Determine the number of activities to process based on num_questions
    # Each activity generates 2 survey questions (Stressful vs Relaxing and Solitary vs Social)
    if num_questions:
        max_activities = num_questions // 2
        activities = activities[:max_activities]
        print(f"Generating survey for the first {max_activities} activities based on the specified number of questions.")
    else:
        print(f"Generating survey for all {len(activities)} extracted activities.")

    stress_relax_pairs = []
    social_solitary_pairs = []
    survey_questions = []


    for activity in activities:
        # Generate Stressful vs Relaxing pairs
        stress_relax = generate_stress_relax_pairs(activity)
        if "Option_A" in stress_relax and "Option_B" in stress_relax:
            stress_relax_pairs.append(stress_relax)
            question = create_survey_question(
                option_a=stress_relax["Option_A"],
                option_b=stress_relax["Option_B"]
            )
            survey_questions.append(question)

        # Generate Solitary vs Social pairs
        social_solitary = generate_social_solitary_pairs(activity)
        if "Option_A" in social_solitary and "Option_B" in social_solitary:
            social_solitary_pairs.append(social_solitary)
            question = create_survey_question(
                option_a=social_solitary["Option_A"],
                option_b=social_solitary["Option_B"]
            )
            survey_questions.append(question)

    # If num_questions is specified and there are more survey questions than needed, trim the list
    if num_questions and len(survey_questions) > num_questions:
        survey_questions = survey_questions[:num_questions]
        print(f"Trimmed the survey questions to the specified number of {num_questions}.")

    return {
        "Stressful_vs_Relaxing_Pairs": stress_relax_pairs,
        "Solitary_vs_Social_Pairs": social_solitary_pairs,
        "Survey_Questions": survey_questions
    }


# Example Usage
if __name__ == "__main__":
    # Sample open-ended responses
    user_responses = """
        anxiety 
        When do you feel anxious? provide 5 situations
        - Public speaking
        - Meeting new people
        - Deadlines at work
        - Taking exams
        - Financial uncertainty

        In what situations do you find it hard to stay calm? List 5 examples
        - During conflicts
        - When multitasking
        - In crowded places
        - While driving in heavy traffic
        - When facing unexpected changes

        relax 
        What are 5 activities that make you feel at ease or comfortable?
        - Reading a novel
        - Listening to music
        - Gardening
        - Painting
        - Taking a walk

        What hobbies or leisure activities help you relax? List 5
        - Yoga
        - Cooking
        - Knitting
        - Watching movies
        - Meditation

        loneliness
        What are 5 times when being alone feels negative?
        - After a breakup
        - During holidays
        - When feeling unwell
        - When facing challenges
        - When missing friends and family

        What are 5 times when you wish you had more company or social interaction?
        - While working on a project
        - During meals
        - While traveling
        - When feeling stressed
        - On weekends

        reduce loneliness 
        When you want to be more social, what are 5 things you do?
        - Attend social gatherings
        - Join clubs or groups
        - Volunteer in community events
        - Reach out to friends and family
        - Participate in online forums

        What do you do to reduce loneliness? provide 5 ways
        - Engage in hobbies
        - Practice mindfulness and meditation
        - Exercise regularly
        - Take up new skills or courses
        - Seek professional counseling
        """

    survey = generate_personalized_survey(user_responses, num_questions=NUM_QUESTIONS)

    print("### Personalized Survey ###\n")
    for idx, question in enumerate(survey["Survey_Questions"], 1):
        print(f"{idx}. {question}\n")



"""
TODO: 

we need these other combinations

4 pairs of R,R
10 pairs of R,A
4 pairs of A,A

4 pairs of G,G
10 pairs of G,S
4 pairs of S,S

"""