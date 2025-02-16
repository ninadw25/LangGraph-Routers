from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from router.query_router import setup_graph_workflow
from config import initialize_db, setup_embeddings

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and embeddings
astra_vector_store = initialize_db()
app_workflow = setup_graph_workflow(astra_vector_store)

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def process_query(query: Query) -> Dict[str, Any]:
    try:
        inputs = {"question": query.question}
        final_output = None
        
        # Process through the graph workflow
        for output in app_workflow.stream(inputs):
            final_output = output
            
        if final_output:
            # Extract relevant information from the final output
            documents = final_output.get(list(final_output.keys())[-1], {}).get('documents', [])
            
            # Format the response based on the document type
            if isinstance(documents, list):
                if documents and hasattr(documents[0], 'dict'):
                    response = documents[0].dict().get('metadata', {}).get('description', '')
                else:
                    response = str(documents)
            else:
                response = str(documents)
                
            return {"response": response}
        else:
            raise HTTPException(status_code=500, detail="No response generated")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)