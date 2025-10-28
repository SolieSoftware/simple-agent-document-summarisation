import os
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from datetime import datetime
from typing import Dict, List, Optional
import re
from dotenv import load_dotenv

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
                Loaded at: {metadata['loaded_at']}

                Preview (first 500 chars):
                {text[:500]}..."""
    except Exception as e:
        return f"Error loading document: {str(e)}"


def list_loaded_documents():
    docs = doc_store.list_documents()
    if not docs: 
        return "No documents loaded yet."

    result = "Loaded Documents:\n"

    for doc_id in docs:
        meta = doc_store.get_metadata(doc_id)
        result += f"\n- {doc_id} ({meta['type']}, {meta['length']} chars, loaded: {meta['loaded_at']})"
    return result