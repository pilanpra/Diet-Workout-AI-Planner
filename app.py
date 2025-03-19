import json
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from nltk.translate.bleu_score import sentence_bleu
from jiwer import wer

# Load environment variables
load_dotenv()
openai_api_key = os.environ["OPENAI_API_KEY"]

# --- Dataset Loading ---
def load_parquet_data(dataset_folder):
    """
    Load and combine Parquet files from the dataset folder.
    """
    data = {}
    for file in os.listdir(dataset_folder):
        if file.endswith(".parquet"):
            category = file.replace(".parquet", "")
            data[category] = pd.read_parquet(os.path.join(dataset_folder, file))
    return data

datasets = load_parquet_data("dataset")  # Path to dataset folder

# --- User Preferences ---
def gather_user_preferences():
    st.sidebar.title("User Preferences")
    goal = st.sidebar.selectbox(
        "What's your main fitness goal?",
        ["Weight Loss", "Build Muscle", "Endurance", "General Fitness"],
    )
    experience = st.sidebar.radio(
        "What's your experience level?", ["Beginner", "Intermediate", "Advanced"]
    )
    restrictions = st.sidebar.text_area("Any injuries or limitations?")
    return {"goal": goal, "experience": experience, "restrictions": restrictions}


# --- Conversation Setup ---
def craft_fitness_prompt(query, datasets, history, user_preferences):
    """
    Craft the prompt for the chatbot using user preferences and dataset relevance.
    """
    history_summary = ""
    if history:
        history_summary = "You previously mentioned the following: "
        for past_query, past_response in history:
            history_summary += f"- You asked: {past_query}. I responded: {past_response}\n"

    preferences_str = f"Your goals are {user_preferences['goal']}, your experience level is {user_preferences['experience']}, and you noted the following restrictions: {user_preferences['restrictions']}."

    # Extract relevant content from datasets (e.g., fitness, diet, wellness)
    relevant_content = []
    for category, df in datasets.items():
        if query.lower() in df["content"].str.lower().values:
            relevant_content.extend(df["content"].sample(3).values)

    relevant_exercises = ", ".join(relevant_content) if relevant_content else "No specific data found."

    prompt = f"You are a helpful fitness expert. {history_summary} {preferences_str} Please answer the following question: {query}."

    if relevant_exercises:
        prompt += f"\nHere are some potentially relevant insights from the dataset: {relevant_exercises}"

    return prompt


def initialize_chat():
    """
    Initialize the LangChain Chat setup.
    """
    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="Answer the question as truthfully as possible, even if you don't have all the information to provide a perfect solution. If the answer is not apparent, provide guidance on how the user might rephrase the question or find more information."
    )
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            system_msg_template,
            MessagesPlaceholder(variable_name="history"),
            human_msg_template,
        ]
    )

    chat = ConversationChain(
        memory=ConversationBufferWindowMemory(k=3, return_messages=True),
        prompt=prompt_template,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key),
    )
    return chat


# --- Evaluation Metrics ---
def calculate_bleu_score(reference, hypothesis):
    """
    Calculate the BLEU score between the reference and hypothesis.
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu([reference_tokens], hypothesis_tokens)

# --- Load JSON Data ---
def load_curated_data(folder_path):
    """
    Load and combine JSON files from the 'curated' folder.
    """
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r") as f:
                json_data = json.load(f)
                if "entries" in json_data:
                    data.extend(json_data["entries"])
    return pd.DataFrame(data)

def calculate_wer(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER) between the reference and hypothesis.
    """
    return wer(reference, hypothesis)

def get_reference_from_curated(query, curated_data):
    """
    Retrieve the most relevant chatbot response for the given query.
    """
    match = curated_data[curated_data["user_input"].str.contains(query, case=False)]
    if not match.empty:
        return match.iloc[0]["chatbot_response"]
    return "No reference available."


# --- Main Chat Logic ---
def process_query(query, datasets, user_preferences, chat, history):
    """
    Process the user query, generate a response, and update the conversation history.
    """
    try:
        prompt = craft_fitness_prompt(query, datasets, history, user_preferences)
        response = chat.predict(input=prompt)
        history.append((query, response))
        return response, history
    except Exception as e:
        return f"An error occurred: {e}. Please try rephrasing your question.", history


# --- Streamlit UI ---
st.title("Fitness Knowledge Bot")
curated_data = load_curated_data("./curated")  # Path to curated folder

# Gather preferences
user_preferences = gather_user_preferences()

# Initialize chat state
if "chat" not in st.session_state:
    st.session_state.chat = initialize_chat()
    st.session_state.history = []
    st.session_state.reference = "Expected Reference Response."  # Placeholder for expected responses

# Chat Input
user_input = st.text_input("Ask me about workouts or fitness...", key="input")

# Display Chat History
history_container = st.container()

# Chat Interaction
if user_input:
    with st.spinner("Thinking..."):
        response, st.session_state.history = process_query(
            user_input, datasets, user_preferences, st.session_state.chat, st.session_state.history
        )

        # Add Evaluation Metrics
        reference = st.session_state.reference  # Replace this with a dynamic reference as needed
        reference = get_reference_from_curated(user_input, curated_data)
        bleu = calculate_bleu_score(reference, response)
        wer_score = calculate_wer(reference, response)

        # Store BLEU and WER scores in session state for accuracy tracking
        if "metrics" not in st.session_state:
            st.session_state.metrics = []
        st.session_state.metrics.append({"bleu": bleu, "wer": wer_score})

    # Display history with Streamlit Chat
    with history_container:
        for idx, (user_msg, bot_msg) in enumerate(st.session_state.history):
            message(user_msg, is_user=True, key=f"user_msg_{idx}")
            message(bot_msg, key=f"bot_msg_{idx}")

        # Display BLEU, WER, and Accuracy
        for idx, metric in enumerate(st.session_state.metrics):
            st.write(f"Interaction {idx + 1} Metrics:")
            st.write(f"- **WER Score:** {metric['wer']:.2f}")
