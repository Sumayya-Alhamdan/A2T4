import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from mistralai import Mistral
import faiss
import time


# Configure Mistral API key
api_key = "bR3uQ4MimCQdHqqPJpIIdCF4wKG51fL2"
os.environ["MISTRAL_API_KEY"] = api_key

# Retrieve and extract policy data from the given URL
def fetch_policy_data(url):
    response = requests.get(url)
    html_doc = response.text
    soup = BeautifulSoup(html_doc, "html.parser")
    tag = soup.find("div")
    text = tag.text.strip()
    return text

# Splits text into smaller chunks of a specified size
def chunk_text(text, chunk_size=512):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Generate embeddings for a list of text chunks
def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=api_key)
    time.sleep(1)  
    embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=list_txt_chunks)
    return embeddings_batch_response.data

# Create and initialize a FAISS index with the given embeddings
def create_faiss_index(embeddings):
    embedding_vectors = np.array([embedding.embedding for embedding in embeddings])
    d = embedding_vectors.shape[1]
    index = faiss.IndexFlatL2(d)
    faiss_index = faiss.IndexIDMap(index)
    faiss_index.add_with_ids(embedding_vectors, np.array(range(len(embedding_vectors))))
    return faiss_index

# Perform a similarity search on the FAISS index to find the most relevant chunks
def search_relevant_chunks(faiss_index, query_embedding, k=2):
    D, I = faiss_index.search(query_embedding, k)
    return I

# Generate an answer using the Mistral model based on the provided context
def mistral_answer(query, context):
    prompt = f"""
    Context information is below.
    ---------------------
    {context}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    Answer:
    """
    client = Mistral(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
    return chat_response.choices[0].message.content

# Streamlit UI 
def streamlit_app():
    st.title('UDST Policies')
    st.markdown(
        "<h3 style='color: #aea0fd;'>Ask a question about a policy</h3>",
        unsafe_allow_html=True
    )

    # List of university policies with descriptive titles
    policies = [
        "Registration Policy",
        "Examination Policy",
        "Graduation Policy",
        "Admissions Policy",
        "Transfer Policy",
        "International Student Policy",
        "Academic Standing Policy",
        "Intellectual Property Policy",
        "Credit Hour Policy",
        "Student Conduct Policy",
        "Student Appeals Policy",
        "Academic Appraisal Policy",
        "Student Engagement Policy",
        "Academic Credentials Policy",
        "Graduate Admissions Policy",
        "Student Attendance Policy",
        "Final Grade Policy",
        "Academic Freedom Policy",
        "Academic Qualifications Policy",
        "Joint Appointment Policy",
        "Program Accreditation Policy",
        "Academic Schedule Policy",
        "Graduate Academic Standing Policy",
        "Academic Members’ Retention Policy",
        "Academic Professional Development Policy",
        "Academic Annual Leave Policy",
        "Graduate Final Grade Policy",
        "Student Counselling Services Policy",
        "Scholarship and Financial Assistance Policy",
        "Use of Library Space Policy",
        "Sport and Wellness Facilities Policy"
    ]
    
    # Corresponding URLs for each policy document
    policy_urls = [
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/intellectual-property-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/credit-hour-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-appraisal-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-credentials-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-freedom-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-qualifications-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/joint-appointment-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/program-accreditation-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-members%E2%80%99-retention-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-professional-development",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-annual-leave-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
        "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and"
    ]


    # Create a dropdown menu for selecting a policy
    selected_policy = st.selectbox("Select a Policy", policies)
    
    # Retrieve the corresponding URL for the selected policy
    selected_policy_url = policy_urls[policies.index(selected_policy)]
    
    # Fetch the policy content from the webpage
    policy_text = fetch_policy_data(selected_policy_url)
    
    # Split the policy text into smaller chunks
    chunks = chunk_text(policy_text)
    
    # Generate embeddings for text chunks and build a FAISS index
    embeddings = get_text_embedding(chunks)
    faiss_index = create_faiss_index(embeddings)
    
    # Input box for user query
    query = st.text_input("Enter your Query:")
    
    if query:
        # Generate an embedding for the user query and search for relevant text chunks
        query_embedding = np.array([get_text_embedding([query])[0].embedding])
        I = search_relevant_chunks(faiss_index, query_embedding, k=2)
        retrieved_chunks = [chunks[i] for i in I.tolist()[0]]
        context = " ".join(retrieved_chunks)
        
        # Generate a response based on the retrieved context
        answer = mistral_answer(query, context)
        
        # Display the generated response
        st.text_area("Answer:", answer, height=200)

# Launch the Streamlit app
if __name__ == '__main__':
    streamlit_app()
