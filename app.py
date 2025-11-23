import patches
import streamlit as st
import os
import pandas as pd
import sqlite3
import re
import shutil
import uuid
from dotenv import load_dotenv

# --- FREE GOOGLE LIBRARIES ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- STANDARD LIBRARIES ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_classic.chains import RetrievalQA
from pydantic import BaseModel, Field

# 1. Setup & Config
st.set_page_config(
    page_title="Doc & Brain | Intelligent Invoice Analyst",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SESSION MANAGEMENT ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Custom CSS for professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp header {
        background-color: #ffffff;
        border-bottom: 1px solid #e0e0e0;
    }
    h1 {
        color: #1a73e8;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #1a73e8;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1557b0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e8f0fe;
        border: 1px solid #d2e3fc;
    }
    .chat-message.bot {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    .chat-message .message {
        flex-grow: 1;
        font-family: 'Segoe UI', sans-serif;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2534/2534866.png", width=80)
    st.title("Doc & Brain")
    st.markdown("---")
    st.markdown("### 📂 Document Center")
    st.info("Upload your invoice PDF to start analyzing.")
    st.markdown("---")
    st.markdown("### 🛠️ System Status")
    if "GOOGLE_API_KEY" in os.environ:
        st.success("API Key Active")
    else:
        st.error("API Key Missing")
    
    st.markdown("---")
    if st.button("🔄 Reset Session"):
        # Cleanup old vector db folder
        old_path = f"db/qdrant_{st.session_state.session_id}"
        if os.path.exists(old_path):
            try:
                shutil.rmtree(old_path)
            except Exception:
                pass
        
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.processed_files = set()
        st.session_state.messages = []
        if "vector_store" in st.session_state:
            del st.session_state.vector_store
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    st.caption("Powered by Gemini 2.0 Flash & Qdrant")

# Main Content
st.title("🧠 Intelligent Invoice Analyst")
st.markdown("#### Hybrid AI Agent for Financial Documents")
st.markdown("Upload an invoice to extract data, perform SQL analysis, and ask semantic questions.")

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("⚠️ GOOGLE_API_KEY missing! Get it for free at aistudio.google.com")
    st.stop()

# 2. Define Data Structure (The "Accountant's" Rules)
class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Name of the vendor/company")
    total_amount: float = Field(description="Total amount of the invoice")
    date: str = Field(description="Date of the invoice (YYYY-MM-DD)")
    invoice_number: str = Field(description="Invoice number")

def clean_filename(filename):
    # Remove extension and keep only alphanumeric characters
    name = os.path.splitext(filename)[0]
    return re.sub(r'[^a-zA-Z0-9]', '_', name).lower()

# 3. The Ingestion Function
def process_document(uploaded_file):
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Add source metadata
    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
    
    # --- SWAP: Use Gemini 2.0 Flash (New & Fast) ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Extraction
    structured_llm = llm.with_structured_output(InvoiceData)
    try:
        extraction = structured_llm.invoke(docs[0].page_content)
    except Exception as e:
        return None, f"❌ Extraction failed for {uploaded_file.name}: {e}"
    
    # Save to SQL (Session-Isolated Table)
    table_name = f"invoices_{st.session_state.session_id}"
    conn = sqlite3.connect("financial_data.db")
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            vendor TEXT, 
            amount REAL, 
            date TEXT, 
            invoice_id TEXT,
            source_file TEXT
        )
    ''')
    
    # Remove old data for this specific file to prevent duplicates
    cursor.execute(f'DELETE FROM "{table_name}" WHERE source_file = ?', (uploaded_file.name,))
    
    # Insert new data
    cursor.execute(f'INSERT INTO "{table_name}" VALUES (?, ?, ?, ?, ?)', 
                   (extraction.vendor_name, extraction.total_amount, extraction.date, extraction.invoice_number, uploaded_file.name))
    
    conn.commit()
    conn.close()
    
    os.remove("temp.pdf")
    return docs, f"✅ Processed {uploaded_file.name}"

# 4. File Uploader & Dashboard
with st.container():
    st.markdown("### multi-file Upload")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

if uploaded_files:
    all_new_docs = []
    status_messages = []
    
    # Initialize processed files tracker
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Process Loop
    with st.status("🚀 Processing Documents...", expanded=True) as status_box:
        for file in uploaded_files:
            if file.name not in st.session_state.processed_files:
                st.write(f"📄 Reading {file.name}...")
                docs, msg = process_document(file)
                if docs:
                    all_new_docs.extend(docs)
                    st.session_state.processed_files.add(file.name)
                    status_messages.append(msg)
                    st.write(f"💾 Saved data for {file.name}")
        
        # Update Vector Store if new docs found
        if all_new_docs:
            st.write("📚 Updating Knowledge Base...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(all_new_docs)
            
            if "vector_store" in st.session_state:
                st.session_state.vector_store.add_documents(splits)
            else:
                # Use Disk Storage for stability (Session-Isolated Path)
                # Unique path per session prevents "Storage folder already accessed" errors
                db_path = f"db/qdrant_{st.session_state.session_id}"
                st.session_state.vector_store = Qdrant.from_documents(
                    splits,
                    embeddings,
                    path=db_path,
                    collection_name="my_docs"
                )
        
        status_box.update(label="✅ All Files Processed!", state="complete", expanded=False)

    # --- DASHBOARD SECTION ---
    st.markdown("---")
    st.markdown("### 📊 Financial Dashboard")
    
    table_name = f"invoices_{st.session_state.session_id}"
    conn = sqlite3.connect("financial_data.db")
    try:
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        if not df.empty:
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Spend", f"${df['amount'].sum():,.2f}")
            with col2:
                st.metric("Total Invoices", len(df))
            with col3:
                st.metric("Unique Vendors", df['vendor'].nunique())
            
            # Charts
            tab1, tab2 = st.tabs(["💰 Spend by Vendor", "📅 Spend over Time"])
            
            with tab1:
                st.bar_chart(df, x="vendor", y="amount", color="#1a73e8")
            
            with tab2:
                # Ensure date is datetime for proper sorting
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.sort_values('date')
                st.line_chart(df, x="date", y="amount", color="#34a853")
                
            # Data Table
            with st.expander("📄 View Raw Data"):
                st.dataframe(df)
        else:
            st.info("No data available yet.")
    except Exception:
        st.info("Upload files to see the dashboard.")
    conn.close()

    # 5. Initialize Agent (Only after data load)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    # Tool 1: SQL Analyst
    table_name = f"invoices_{st.session_state.session_id}"
    db = SQLDatabase.from_uri("sqlite:///financial_data.db", include_tables=[table_name])
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, verbose=True)
    
    # Tool 2: General Chat (Simple Fallback since Qdrant is in-memory inside function)
    # Note: In a 1-file demo, preserving Qdrant memory state across reruns is tricky. 
    # For this "Free" demo, the SQL part is the most impressive 'Hybrid' feature.
    
    # Create a simple function to handle queries
    def handle_query(query: str) -> str:
        # Try SQL agent first for data-related questions
        sql_keywords = ['total', 'sum', 'amount', 'count', 'average', 'date', 'vendor', 'invoice']
        if any(keyword in query.lower() for keyword in sql_keywords):
            try:
                return sql_agent.run(query)
            except Exception as e:
                return f"I had trouble with the database query. Error: {str(e)}"
        else:
            # Use Vector Store for text-based questions
            if "vector_store" in st.session_state:
                retriever = st.session_state.vector_store.as_retriever()
                # Create a simple RAG chain
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
                return qa.run(query)
            else:
                return "I can help you analyze the invoice data in the database. Try asking about totals, amounts, vendors, or dates."