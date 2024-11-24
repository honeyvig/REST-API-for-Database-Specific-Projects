# REST-API-for-Database-Specific-Projects
To create a REST API for a database-specific project using FastAPI and PostgreSQL, the API would need to support functionalities like session management, reasoning from prompts, building questions from chat history, and integrating with a database for persistent data storage. You would also want to incorporate a RAG (Retrieval Augmented Generation) system, where information is fetched from the database and then used for generating meaningful responses using AI models.

Below is a detailed Python code for setting up the REST API, integrating FastAPI with PostgreSQL, session management, and a basic framework for building questions from past prompts.
1. Install Required Dependencies

Before proceeding, ensure you have the following dependencies installed:

pip i

nstall fastapi uvicorn psycopg2 sqlalchemy openai

    fastapi: For building the API.
    uvicorn: For running the ASGI server.
    psycopg2: PostgreSQL adapter for Python.
    sqlalchemy: For managing database models and ORM.
    openai: If you are integrating with OpenAI API for question generation or reasoning.

2. Define Database Models Using SQLAlchemy

Define the models for storing chat sessions and their corresponding prompts in PostgreSQL.
models.py

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql://user:password@localhost/db_name"  # Replace with your PostgreSQL URL

Base = declarative_base()

# Create a database session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Chat Session Model
class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    prompt = Column(Text)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the database tables
Base.metadata.create_all(bind=engine)

This code defines a ChatSession model to store the user ID, prompt, response, and timestamp of each session.
3. FastAPI App Setup for Chat Session Management

You can now create the FastAPI app that handles API requests, manages chat sessions, and integrates with a RAG system for reasoning.
main.py

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import SessionLocal, ChatSession
import openai
import uuid

# Initialize FastAPI app
app = FastAPI()

# OpenAI API Key for reasoning (optional)
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Pydantic models for request validation
class ChatRequest(BaseModel):
    user_id: int
    prompt: str

class ChatResponse(BaseModel):
    user_id: int
    prompt: str
    response: str

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API Endpoint for generating a response from the system
@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    # Get previous session history for the user
    previous_chats = db.query(ChatSession).filter(ChatSession.user_id == request.user_id).order_by(ChatSession.timestamp.desc()).limit(5).all()
    previous_prompts = [chat.prompt for chat in previous_chats]
    
    # Combine previous prompts for reasoning (RAG - Retrieval Augmented Generation)
    prompt_with_history = "\n".join(previous_prompts) + "\n" + request.prompt
    
    # Generate a response using OpenAI API or any other reasoning model
    try:
        # Here, we use OpenAI's GPT to generate a response based on the prompt
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt_with_history,
            max_tokens=150
        )
        ai_response = response.choices[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Save chat session to the database
    chat_session = ChatSession(
        user_id=request.user_id,
        prompt=request.prompt,
        response=ai_response
    )
    db.add(chat_session)
    db.commit()

    # Return the response
    return {"user_id": request.user_id, "prompt": request.prompt, "response": ai_response}

# API Endpoint to fetch chat history for a user
@app.get("/chats/{user_id}")
def get_chat_history(user_id: int, db: Session = Depends(get_db)):
    chats = db.query(ChatSession).filter(ChatSession.user_id == user_id).order_by(ChatSession.timestamp.desc()).all()
    
    if not chats:
        raise HTTPException(status_code=404, detail="No chats found for this user")
    
    return {"user_id": user_id, "chats": [{"prompt": chat.prompt, "response": chat.response, "timestamp": chat.timestamp} for chat in chats]}

# API Endpoint to delete a chat session
@app.delete("/chats/{session_id}")
def delete_chat_session(session_id: int, db: Session = Depends(get_db)):
    chat = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    db.delete(chat)
    db.commit()
    return {"detail": "Chat session deleted successfully"}

Explanation of API Endpoints:

    /chat/:
        POST request to generate a response based on the prompt, utilizing the user's previous chat history.
        The RAG system is implemented here, where the last 5 chat prompts are retrieved and concatenated to build a context for generating a response.
        The response is generated using OpenAI or another reasoning model (e.g., GPT, BERT).

    /chats/{user_id}:
        GET request to fetch the chat history of a specific user.

    /chats/{session_id}:
        DELETE request to remove a chat session from the database.

4. Running the FastAPI App

To run the FastAPI app, use the following command:

uvicorn main:app --reload

    This will start the server locally, and you can test the API by making requests to http://localhost:8000.

5. Integrating with PostgreSQL

    The app uses SQLAlchemy to interact with the PostgreSQL database.
    Chat sessions are stored in a chat_sessions table.
    PostgreSQL connection string: "postgresql://user:password@localhost/db_name".
        Make sure to replace it with your actual database credentials.

6. Testing the API
Create Chat Session (POST /chat/)

Example request body:

{
  "user_id": 1,
  "prompt": "What is the capital of France?"
}

Example response:

{
  "user_id": 1,
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris."
}

Get Chat History (GET /chats/1)

{
  "user_id": 1,
  "chats": [
    {
      "prompt": "What is the capital of France?",
      "response": "The capital of France is Paris.",
      "timestamp": "2024-11-24T13:45:00"
    }
  ]
}

Delete Chat Session (DELETE /chats/{session_id})

Example response:

{
  "detail": "Chat session deleted successfully"
}

Conclusion

This REST API provides a backend service for managing user chat sessions, using a RAG (Retrieval Augmented Generation) system to generate AI-driven responses based on past interactions, and integrates with PostgreSQL for session persistence. It also supports FastAPI's rapid development and scalability features, making it suitable for building conversational AI and session management systems
