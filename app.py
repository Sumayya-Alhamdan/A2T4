import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from mistralai import Mistral, UserMessage

# ========================
# Set up API key and config
# ========================
# Replace with your own Mistral API key
os.environ["MISTRAL_API_KEY"] = "oexFUNi7HS8gq55pCPijrij9cd5Q2EMn"
api_key = os.getenv("MISTRAL_API_KEY")

# ========================
# (Optional) Scrape UDST Policies
# ========================
# You might want to scrape the UDST policies page at:
# "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures"
# and extract individual policy texts. For this example, we define them manually.
#
def scrape_udst_policies(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Adjust extraction logic as needed
    content = soup.get_text()
    return content

udst_policies_text = scrape_udst_policies("https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures")

# ========================
# Load policies (manually defined for now)
# ========================
@st.cache(allow_output_mutation=True)
def load_policies():
    policies = {
        "Academic Policy": "Detailed content for Academic Policy from UDST goes here...",
        "Student Discipline Policy": "Detailed content for Student Discipline Policy from UDST goes here...",
        "Research Policy": "Detailed content for Research Policy from UDST goes here...",
        "Faculty Policy": "Detailed content for Faculty Policy from UDST goes here...",
        "IT Policy": "Detailed content for IT Policy from UDST goes here...",
        "Human Resources Policy": "Detailed content for Human Resources Policy from UDST goes here...",
        "Financial Policy": "Detailed content for Financial Policy from UDST goes here...",
        "Health and Safety Policy": "Detailed content for Health and Safety Policy from UDST goes here...",
        "Environmental Policy": "Detailed content for Environmental Policy from UDST goes here...",
        "Diversity and Inclusion Policy": "Detailed content for Diversity and Inclusion Policy from UDST goes here..."
    }
    return policies

policies = load_policies()
policy_names = list(policies.keys())

# ========================
# Streamlit User Interface
# ========================
st.title("UDST Policies Chatbot")
st.markdown("Select a UDST policy, enter your query, and get an answer based solely on the selected policy content.")

selected_policy = st.selectbox("Select a Policy", policy_names)
user_query = st.text_input("Enter your Query")

# ========================
# Helper Functions
# ========================
def chunk_text(text, chunk_size=512):
    """Split the text into chunks of specified size."""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_embedding(text_chunks):
    """
    Uses Mistral to generate embeddings for a list of text chunks.
    Returns a list of embedding objects.
    """
    client = Mistral(api_key=api_key)
    response = client.embeddings.create(model="mistral-embed", inputs=text_chunks)
    return response.data  # Each element has an 'embedding' attribute

def generate_answer(prompt, model="mistral-large-latest"):
    """
    Uses Mistral's chat model to generate an answer given a prompt.
    """
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=prompt)]
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content

# ========================
# Process the Query
# ========================
if st.button("Submit"):
    if user_query.strip() == "":
        st.warning("Please enter a query.")
    else:
        # Retrieve the selected policy content
        policy_text = policies[selected_policy]
        
        # Split policy text into chunks
        chunks = chunk_text(policy_text, chunk_size=512)
        
        # Generate embeddings for the chunks
        embeddings_data = get_text_embedding(chunks)
        # Convert list of embeddings into a NumPy array
        embeddings = np.array([emb.embedding for emb in embeddings_data])
        
        # Build FAISS index (L2 distance)
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        
        # Get embedding for the user's query
        query_embedding_data = get_text_embedding([user_query])
        query_embedding = np.array([query_embedding_data[0].embedding])
        
        # Retrieve the top 2 most similar chunks
        k = 2
        distances, indices = index.search(query_embedding, k)
        retrieved_chunks = [chunks[i] for i in indices[0]]
        
        # Build the prompt with retrieved context
        prompt = f"""
Context information:
---------------------
{' '.join(retrieved_chunks)}
---------------------
Based solely on the above context and without any external knowledge, answer the following query:
Query: {user_query}
Answer:
"""
        # Generate the answer using Mistral chat model
        answer = generate_answer(prompt)
        
        # Display the answer
        st.text_area("Answer", value=answer, height=300)
