from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
#importing
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np
from groq import Groq
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import json
from typing import List  # Make sure to import List from typing

#loading keys and env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify the domains you want to allow
    allow_credentials=True,
    allow_methods=["*"],  # or specify the methods you want to allow
    allow_headers=["*"],  # or specify the headers you want to allow
)

#initializing groq client to send api request
client = Groq(api_key=groq_api_key)

# instructions to final model (llama 3.1 8b)
primer = f"""You are a personal assistant. Answer any questions I have about the Youtube Video provided.
Translate in specific language if user asks you to
"""


#defining hugging face embedding variable
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#initializing vector database "Chroma db"
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=hf_embeddings,
)


# function to create vector embeddings of the user query 

def get_embedding(text, model="sentence-transformers/all-MiniLM-L6-v2"):
    response= HuggingFaceEmbeddings(model_name=model)
    embedded_response = hf_embeddings.embed_query(text)
    return embedded_response


#making api
# Define the QueryRequest model to accept a list of messages
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]



@app.post('/api/ask/')


#function for getting response from the llm
async def get_response(request:QueryRequest):
  print('api started running')
  try:
    all_messages= request.messages
 #calling the function for user query vector embeddings
    raw_query_embedding=get_embedding(all_messages[-1].content)
    
    #after converting query to vector, performing similarity search by vector 
    results = vector_store.similarity_search_by_vector(
        embedding=raw_query_embedding, k=1
    )
    
    #providing the context to llm by performing similarity vector search
    contexts = [doc.page_content for doc in results]
    
    #maintain user chat history
    conversation_history= "\n\n".join(f"{msg.role}: {msg.content}" for msg in all_messages)
    msg_string=json.dumps(conversation_history)
    #combining context with user query to allow model to perform RAG
    augmented_query = (
            "<CONTEXT>\n" +
            "\n\n-------\n\n".join(contexts) +
            "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" +
            conversation_history
        )
    
    # makng call to llm to get the answer
    response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
            {"role": "system", "content": primer},
            {"role": "user", "content": augmented_query},
            ],
            max_tokens=1000,
            temperature=1.2)
    
    #returning the response of llm
    return {'assistantMessage':response.choices[0].message.content}
  except Exception as e:
      raise HTTPException(status_code=500,detail= str(e))




