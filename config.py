import os
from dotenv import load_dotenv
import cassio
from langchain_community.vectorstores import Cassandra
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def initialize_db():
    # Initialize Astra DB connection
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
    
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)
    
    # Initialize embeddings
    embeddings = setup_embeddings()
    
    # Initialize vector store
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="qa_mini_demo",
        session=None,
        keyspace=None
    )
    
    return astra_vector_store

def setup_embeddings():
    # You can modify this to use different embedding models
    return HuggingFaceEmbeddings()