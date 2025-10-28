import os
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader
)
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class LoadDocumentInput(BaseModel):
    """Input for loading a document. """
    filepath: str = Field(description="The full path to the document file")

class GetDocumentInput(BaseModel):
    """Input for retrievign document content."""
    document_id: str = Field(description="The document ID (filename)")

load_dotenv('.env.local')

llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    temperature=0
)

class DocumentStore:
    def __init__(self):
        self.documents = {}
        self.metadata = {}

    def add_document(self, document_id: str, content: str, filepath: str):
        self.documents[document_id] = content
        self.metadata[document_id] = {
            "filepath": filepath,
            "upload_time": datetime.now().isoformat(),
            "length": len(content),
            "type": filepath.split('.')[-1]
        }

    def get_document(self, document_id: str) -> Optional[str]:
        return self.documents.get(document_id)

    def get_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.metadata.get(document_id)

    def list_documents(self) -> List[str]:
        return list(self.documents.keys())

    def search_documents(self, keyword: str) -> List[str]:
        matches = []
        for doc_id, content in self.documents.items():
            if keyword.lower() in content.lower():
                matches.append(doc_id)
        return matches
    

doc_store = DocumentStore()

def load_document(filepath: str):
    try:
        if filepath.endswith('.txt'):
            loader = TextLoader(filepath)
        elif filepath.endswith('.pdf'):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(filepath)
        elif filepath.endswith('.csv'):
            loader = CSVLoader(filepath)
        else:
            loader = TextLoader(filepath)

        documents = loader.load()

        text = "\n".join([doc.page_content for doc in documents])

                # Store in memory
        doc_id = os.path.basename(filepath)
        doc_store.add_document(doc_id, text, filepath)
        
        metadata = doc_store.get_metadata(doc_id)
        
        return f"""Document loaded successfully!
            
                ID: {doc_id}
                Type: {metadata['type']}
                Size: {metadata['length']} characters
                Upload Time: {metadata['upload_time']}

                Preview (first 500 chars):
                {text[:500]}..."""
    except Exception as e:
        return f"Error loading document: {str(e)}"


def list_loaded_documents(input: str=""):
    docs = doc_store.list_documents()
    if not docs: 
        return "No documents loaded yet."

    result = "Loaded Documents:\n"

    for doc_id in docs:
        meta = doc_store.get_metadata(doc_id)
        result += f"\n- {doc_id} ({meta['type']}, {meta['length']} chars, loaded: {meta['upload_time']})"
    return result

def search_documents(keyword: str):
    docs = doc_store.search_documents(keyword)
    if not docs:
        return "No documents found."
    
    result = "Search Results:\n"
    for doc_id in docs:
        meta = doc_store.get_metadata(doc_id)
        result += f"\n- {doc_id} ({meta['type']}, {meta['length']} chars, loaded: {meta['upload_time']})"
    return result

def get_document_content(document_id: str) -> str: 
    content = doc_store.get_document(document_id)
    if content is None:
        return f"Document {document_id} not found. Use list_loaded_documents to see all loaded documents."
    return content
    

tools = [
      StructuredTool.from_function(
          name="load_document",
          func=load_document,
          description="Load a document from a filepath. Input should be the full filepath as a string. Returns confirmation with  document ID and preview.",
          args_schema=LoadDocumentInput
      ),
      Tool(
          name="list_documents",
          func=list_loaded_documents,
          description="List all currently loaded documents with their metadata. No input required."
      ),
      StructuredTool.from_function(
          name="get_document_content",
          func=get_document_content,
          description="Get the full content of a loaded document. Input should be the document ID (filename). Use this to read and analyze document contents.",
          args_schema=GetDocumentInput,
      ),
  ]

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that can load documents. You can:
    - Load a document from a filepath.
    - List loaded documents.
    - Retrieve and analyze document content.
    - Answer questions about the documents.
    - Provide summaries, key insights, and extract specific information.

    When a user asks you to analyze a document:
    1. First check if it's already loaded (use list_documents).
    2. If it's not loaded, load it using the filepath provided.
    3. Retrieve the content using get_document_content.
    4. Provide your analysis based on what the user asked for.

    Be thorough and cite specific parts of the document in your analysis. """),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def main():
    print("Document analysis Agent started!")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        try:
            response = agent_executor.invoke({"input": user_input})
            print("\nAssistant: ", response['output'])
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
