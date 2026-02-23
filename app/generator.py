import os
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

def get_groq_llm():
    """
    Initializes the ChatGroq client.
    Uses the requested model for high-speed generation.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key":
        raise ValueError("GROQ_API_KEY is not configured.")
        
    return ChatGroq(
        groq_api_key=api_key,
        model_name="moonshotai/kimi-k2-instruct-0905",
        temperature=0.0 # Force determinism to prevent hallucination
    )

def generate_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Synthesizes a final answer using the Groq LLM based precisely on the provided context chunks.
    """
    llm = get_groq_llm()
    
    # Extract the plain text content from the chunks to form the context payload
    context_text = "\n\n".join([f"Review excerpt: {chunk['page_content']}" for chunk in retrieved_chunks])
    
    # Prompt engineering
    prompt_template = """
    You are an AI assistant designed to help users determine the truth about a product based ONLY on real customer reviews.
    
    Context (Filtered Customer Reviews):
    ---
    {context}
    ---
    
    User Question: {question}
    
    Instructions:
    1. Answer the user's question directly based ONLY on the context provided above.
    2. If the context does not contain enough information to answer the question, say "I cannot find enough information in the verified reviews to answer that." 
    3. Do not synthesize outside knowledge or hallucinate features.
    4. Keep your answer concise and helpful. Be sure to highlight both positive and negative aspects if mentioned.
    
    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Construct the pipeline
    chain = prompt | llm
    
    # Execute the chain
    response = chain.invoke({"context": context_text, "question": query})
    
    # Return the string content of the AI message
    return response.content
