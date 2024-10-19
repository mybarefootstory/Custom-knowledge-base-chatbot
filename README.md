# Chatbot using RAG with Langchain, Pinecone, and Hugging Face

Welcome to the repository for the **RAG-based Chatbot** built using **Langchain**, **Pinecone**, and **Hugging Face**. This chatbot uses a Retrieval-Augmented Generation (RAG) approach, making it more robust and efficient for answering queries by retrieving the most relevant documents and generating responses from them.

## Key Features

- **Retrieval-Augmented Generation (RAG)**: The chatbot follows the RAG architecture. It retrieves documents relevant to a user's query using Pinecone’s vector search and then uses Hugging Face models to generate a response based on the retrieved context.
  
- **Langchain Integration**: Langchain powers the document splitting, similarity search, and embedding steps, ensuring modularity and flexibility.
  
- **Pinecone Vector Store**: Pinecone is used as the vector store for embedding document vectors and performing efficient similarity searches. It supports high-dimensional search space optimization and serverless architecture with fast response times.

- **Hugging Face Embeddings**: The chatbot uses **MiniLM (all-MiniLM-L6-v2)** from Hugging Face to generate document embeddings. This model is lightweight and optimized for embedding creation, ensuring fast and accurate results.

- **PDF Support**: The chatbot can handle PDF documents, allowing the user to upload and split large PDFs into manageable chunks for better search and response capabilities.

## Models Used

1. **Hugging Face Embeddings**: The chatbot leverages the **all-MiniLM-L6-v2** model for document embeddings. This model is part of the **SentenceTransformers** library and is well-suited for semantic search, making it perfect for document retrieval tasks.

2. **Langchain’s RecursiveCharacterTextSplitter**: This is used to split long documents into chunks of manageable size, ensuring that embeddings are efficiently created, and the context is not lost during the process.

3. **Pinecone Index**: Pinecone is used as the vector store, with embeddings stored and searched within it. Pinecone's serverless specification provides a cloud-based, scalable infrastructure.

## How It Works

1. **Document Upload**: Users can upload a PDF, which is processed using the PyMuPDF library (`fitz`). The entire text of the document is extracted and prepared for embedding.

2. **Text Splitting**: The document is split into smaller chunks using the `RecursiveCharacterTextSplitter` from Langchain. This ensures that the document is divided into manageable pieces without breaking important context.

3. **Embedding Creation**: After splitting, each chunk is embedded using the **Hugging Face model** (MiniLM-L6-v2).

4. **Pinecone Indexing**: These embeddings are stored in a Pinecone vector index for efficient retrieval based on semantic similarity.

5. **Similarity Search**: When a query is made, the chatbot searches for the most relevant chunks from the vector index using **cosine similarity** as the metric.

6. **Response Generation**: The chatbot finds the best matching documents using keyword-based filtering, ensuring the most relevant answer is returned based on context.

## Usage

To run the chatbot locally or in a cloud-based environment like Google Colab, follow these steps:

### 1. Install the required libraries:

```bash
!pip install sentence_transformers unstructured detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.6 PyMuPDF langchain-community tensorflow pinecone-client[grpc]
```
### 2. Initialize Pinecone, upload the PDF, and run the code: 
Pinecone will store the document embeddings and handle the similarity search based on queries.

### 3. Use the chatbot by providing a query. For example:
```python
query = "How was the world's economy in 2020?"
similar_docs = vectorstore.similarity_search(query)
```
### 4.The chatbot will return the most relevant document based on the search and generate an insightful response.

## Key Configuration
1. **Pinecone API Key** : The API key for Pinecone must be set in the environment variables.
2. **Query and Response** : The chatbot can be used to ask questions based on the document’s content, especially for complex queries requiring document context.

