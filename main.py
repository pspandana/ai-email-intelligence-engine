# main.py

from dotenv import load_dotenv
load_dotenv() # This line reads the .env file

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# --- 1. LOAD OUR MODELS ON STARTUP ---

# Load the Triage Classifier model
triage_classifier = pipeline(
    "text-classification",
    model="./models/triage_classifier"
)

# Load the Vector Database
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./db", embedding_function=embeddings)

# Configure a local LLM to run on your CPU
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    config={'temperature': 0.7, 'max_new_tokens': 256}
)

# Create a robust prompt template
prompt_template = PromptTemplate(
    template="""
You are an expert AI assistant. Your task is to analyze an email and provide a helpful response.
Use the CONTEXT section for fact-checking and to find relevant information, but the main goal is to respond to the EMAIL.

---
[START OF CONTEXT]
{context}
[END OF CONTEXT]
---
[START OF EMAIL]
{email}
[END OF EMAIL]
---

Based on the EMAIL and referencing the CONTEXT, please perform the following actions:
1.  **Summary:** Provide a single, concise sentence summarizing the core request of the email.
2.  **Next Steps:** Suggest 3 short, actionable next steps for the recipient in a numbered list.
3.  **Draft Reply:** Write a polite, professional reply to the email.

YOUR RESPONSE:
""",
    input_variables=["context", "email"]
)

# --- 2. SETUP THE FASTAPI APP ---

app = FastAPI()

class Email(BaseModel):
    body: str

# --- 3. DEFINE THE API ENDPOINT ---

@app.post("/process-email")
def process_email(email: Email):
    # Step 1: Use our fast, local triage classifier
    triage_result = triage_classifier(email.body)[0]
    label_parts = triage_result['label'].split('_')
    intent = label_parts[0]
    urgency = label_parts[1]

    # Step 2: Use RAG to get relevant context
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(email.body)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Step 3: Use the powerful LLM for summary and reply generation
    formatted_prompt = prompt_template.format(context=context, email=email.body)
    llm_response = llm.invoke(formatted_prompt)

    # Step 4: Combine all the results into a dictionary
    response_data = {
        "triage_classification": {
            "intent": intent,
            "urgency": urgency,
            "confidence": triage_result['score']
        },
        "retrieved_context": context,
        "llm_analysis": llm_response
    }

    # --- THIS IS THE ADDED DEBUGGING STEP ---
    print("--- API Response ---")
    print(response_data)
    print("--------------------")
    # ----------------------------------------

    # Return the final results
    return response_data
