from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import crag_model

app = FastAPI()

# Enable CORS for all origins; adjust as needed for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body.
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        question = request.question
        if not question:
            raise HTTPException(status_code=400, detail="Please provide a valid question!")
        
        # Moderation check
        if crag_model.moderate_response(question):
            response_text = (
                "warning: So sorry!, but your statement has been flagged as inappropriate. "
                "Please rephrase your input and try again"
            )
        else:
            # Process the question using your agent
            response_text = crag_model.agent(question)
            print(f"Generated response: {response_text}, {type(response_text)}")
        
        return {"answer": response_text}
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
