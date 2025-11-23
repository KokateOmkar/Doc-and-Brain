import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from pydantic import BaseModel, Field

# 1. Setup
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# 2. Define Structured Data (What to extract for Math)
class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Name of the vendor/company")
    total_amount: float = Field(description="Total amount of the invoice")
    date: str = Field(description="Date of the invoice (YYYY-MM-DD)")
    invoice_number: str = Field(description="Invoice number")

def ingest_data():
    print("🔄 Starting Ingestion...")
    
    # A. Load PDF
    pdf_path = "data/invoice.pdf" # Make sure this file exists!
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Please put a PDF in {pdf_path}")
        return

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # B. Extract Structured Data (The "Accountant")
    print("📊 Extracting structured data...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    structured_llm = llm.with_structured_output(InvoiceData)
    
    # We feed the first page content to extract key fields
    # (For complex docs, you might loop through all pages)
    extraction = structured_llm.invoke(docs[0].page_content)
    
    # Save to SQL
    conn = sqlite3.connect("financial_data.db")
    data = {
        "vendor": [extraction.vendor_name],
        "amount": [extraction.total_amount],
        "date": [extraction.date],
        "invoice_id": [extraction.invoice_number]
    }
    df = pd.DataFrame(data)
    df.to_sql("invoices", conn, if_exists="replace", index=False)
    conn.close()
    print("✅ Structured data saved to SQL Database!")

    # C. Store Unstructured Text (The "Librarian")
    print("📚 Storing text in Vector DB...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    Qdrant.from_documents(
        splits,
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        path="db/local_qdrant", # Saves to disk
        collection_name="my_docs"
    )
    print("✅ Text saved to Qdrant Vector DB!")
    print("🚀 Ingestion Complete!")

if __name__ == "__main__":
    ingest_data()