# RAG-Based Document QA System

A Retrieval-Augmented Generation (RAG) system for answering user queries over a local document collection with citation support and secure query handling.

## Overview

This project implements a lightweight RAG pipeline that retrieves relevant document chunks and generates grounded answers using a large language model.

Instead of relying solely on the model’s internal knowledge, responses are strictly based on retrieved context to improve accuracy and reduce hallucination.

## Features

- Document loading from local Markdown files
- Text chunking for efficient retrieval
- Embedding-based semantic search
- Top-k similarity retrieval
- Citation-aware responses (source files)
- Prompt injection and sensitive data protection
- Interactive Streamlit interface

## How It Works

1. Documents are loaded from the `docs/` directory  
2. Text is split into smaller chunks  
3. Each chunk is converted into an embedding  
4. User queries are embedded and compared using similarity search  
5. Top-k relevant chunks are retrieved  
6. The LLM generates a response using only retrieved context  
7. The system returns the answer along with source citations  

## Project Structure
