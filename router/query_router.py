from typing import Literal, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict
from langchain.schema import Document
import os

class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "wiki_search"] = Field(
        description="Route user query to wikipedia or vectorstore."
    )

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def setup_graph_workflow(vector_store):
    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="Gemma2-9b-It"
    )
    structured_llm_router = llm.with_structured_output(RouteQuery)

    # Setup Wikipedia tool
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

    # Setup retriever
    retriever = vector_store.as_retriever()

    # Define node functions
    def retrieve(state):
        question = state["question"]
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    def wiki_search(state):
        question = state["question"]
        docs = wiki.invoke({"query": question})
        wiki_results = Document(page_content=docs)
        return {"documents": wiki_results, "question": question}

    def route_question(state):
        system_prompt = """You are an expert at routing user questions.
        The vectorstore contains documents about agents, prompt engineering, and adversarial attacks.
        Use vectorstore for these topics. Otherwise, use wiki-search."""
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])
        
        question = state["question"]
        source = structured_llm_router.invoke(
            route_prompt.format_messages(question=question)
        )
        
        return "wiki_search" if source.datasource == "wiki_search" else "vectorstore"

    # Setup graph
    workflow = StateGraph(GraphState)
    workflow.add_node("wiki_search", wiki_search)
    workflow.add_node("retrieve", retrieve)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "wiki_search": "wiki_search",
            "vectorstore": "retrieve",
        },
    )
    
    workflow.add_edge("retrieve", END)
    workflow.add_edge("wiki_search", END)

    return workflow.compile()