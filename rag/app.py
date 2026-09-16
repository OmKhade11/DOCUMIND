import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv("objectbox/.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found.")
    st.stop()


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 DocuMind RAG")
st.write("Upload a PDF and ask questions about its content.")


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)


# --------------------------------------------------
# Process document
# --------------------------------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    with st.spinner("Processing document..."):

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        documents = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        db = Chroma.from_documents(
            documents,
            embeddings
        )

    st.success(
        f"Document processed successfully! "
        f"Created {len(documents)} chunks."
    )


    # --------------------------------------------------
    # Question
    # --------------------------------------------------

    question = st.text_input(
        "Ask a question about the document:"
    )


    if question:

        with st.spinner("Searching and generating answer..."):

            retrieved_docs = db.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            prompt = ChatPromptTemplate.from_template(
                """
                Answer the question based only on the
                following context.

                Context:
                {context}

                Question:
                {question}

                If the answer cannot be found in the
                context, say that the information is
                not available in the document.

                Answer:
                """
            )

            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                temperature=0
            )

            response = llm.invoke(
                prompt.invoke({
                    "context": context,
                    "question": question
                })
            )


        # --------------------------------------------------
        # Display answer
        # --------------------------------------------------

        st.subheader("Answer")

        st.write(response.content)


        # --------------------------------------------------
        # Display sources
        # --------------------------------------------------

        st.subheader("Retrieved Sources")

        for i, doc in enumerate(retrieved_docs, 1):

            with st.expander(f"Source {i}"):

                st.write(doc.page_content)