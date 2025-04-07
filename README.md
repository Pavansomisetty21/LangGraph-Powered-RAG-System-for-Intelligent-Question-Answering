

# **LangGraph-Powered RAG System for Intelligent Question Answering**

## **Project Overview**

The **LangGraph-Powered Retrieval-Augmented Generation (RAG) System** is a state-of-the-art architecture designed for **Intelligent Question Answering** (QA). This system utilizes a graph-based framework called **LangGraph** to optimize the process of retrieving and generating accurate responses. It combines the power of **document retrieval**, **document grading**, **web search augmentation**, and **Large Language Models (LLMs)** to deliver precise and contextually relevant answers to user queries.

### Key Highlights:
- **Graph-based Workflow:** The system adopts a LangGraph-powered approach, where each node in the graph represents a specific action (like retrieving documents, grading them, or generating answers). This structure helps in managing the workflow efficiently, ensuring that each step is executed based on the preceding actions.
- **Retrieval-Augmented Generation (RAG):** By combining document retrieval from a local vector store (ChromaDB) and web search for real-time information, this system enhances answer accuracy by integrating both stored knowledge and up-to-date content from the web.
- **Document Grading:** Prior to answer generation, the retrieved documents undergo a grading process to assess their relevance. This step ensures that only the most pertinent documents are used to generate answers, leading to better-quality results.
- **LLM Answer Generation:** After the documents are retrieved and graded, the system uses **Llama-3.3-70B** (or another suitable LLM) to generate human-like, coherent, and contextually correct responses.
- **Decision-making Workflow:** If the retrieved documents are deemed insufficient, the system can trigger a **web search** to gather external knowledge, ensuring that the generated answers are comprehensive and up-to-date.

## **System Workflow**

### Step-by-Step Process:
1. **User Query Submission:**  
   The user asks a question, which is the input to the system.
   
2. **Document Retrieval (ChromaDB):**  
   The system first searches the local **ChromaDB** vector store for relevant documents. ChromaDB stores document embeddings, which are used for efficient similarity search, ensuring that the most relevant documents are retrieved based on the user's query.

3. **Document Grading:**  
   After the documents are retrieved, they undergo a **grading process**. This process involves evaluating the relevance and quality of the documents based on the query. Grading helps filter out irrelevant or low-quality documents, ensuring that only the most useful ones are used in the next steps.

4. **Decision Node:**  
   At this point, the system decides whether the retrieved documents are sufficient to generate a quality answer:
   - **Sufficient Documents:** If the graded documents are relevant and comprehensive enough, the system moves to the next step of **Answer Generation**.
   - **Insufficient Documents:** If the retrieved documents are insufficient or lack critical information, the system triggers a **Web Search** to gather additional knowledge from the web.

5. **Web Search (if needed):**  
   If the initial retrieval doesn't provide enough information, the system queries external search engines or APIs to fetch real-time knowledge that can enrich the answer generation process.

6. **Answer Generation (LLM):**  
   Once sufficient documents (either from ChromaDB or from web search) are available, the system uses **Llama-3.3-70B** or another pre-trained LLM to generate a detailed and contextually appropriate answer. The LLM synthesizes the relevant content from the retrieved documents and provides a coherent response to the user's question.

7. **Return the Answer:**  
   The system presents the generated answer to the user. The final answer could be purely based on local document retrieval, or it may include insights from web search if the initial documents were not adequate.

## **Technologies Used**

The **LangGraph-Powered RAG System** incorporates the following technologies:
- **LangGraph:** A graph-based workflow framework used to orchestrate the steps of document retrieval, grading, decision-making, and answer generation.
- **ChromaDB:** A vector database that stores document embeddings and enables efficient similarity search. It is used to retrieve the most relevant documents based on a user's query.
- **Llama-3.3-70B:** A large pre-trained language model used for generating human-like responses. Llama-3.3-70B synthesizes the retrieved content and generates high-quality answers.
- **Web Search API:** Used to enhance the retrieval process by gathering additional knowledge from the web when the local document store does not provide sufficient information.
- **FastAPI / Streamlit:** The backend (FastAPI) and frontend (Streamlit) frameworks used to interact with the system. FastAPI handles API calls and logic, while Streamlit provides a user-friendly interface for input and output.

## **System Architecture**

The system's architecture is structured around a **graph-based workflow**. LangGraph acts as the central control, managing the sequence of tasks and decisions. Here's a breakdown of the architecture:

```plaintext
    [ Start ]  
        ↓  
    [ Retrieve Documents ]  ← Fetch from ChromaDB  
        ↓  
    [ Grade Documents ]  ← Evaluate document relevance  
        ↓  
    [ Decision Node ]  ← Check sufficiency  
      /       \  
[ Web Search ]  [ LLM Generate ]  
      \       /  
      [ End ]  
```

1. **Start:** The process begins with a user query.
2. **Retrieve Documents:** Relevant documents are fetched from **ChromaDB** based on vector embeddings.
3. **Grade Documents:** The system grades the documents based on their relevance to the query.
4. **Decision Node:** If sufficient information is available, the system generates an answer using the LLM; otherwise, it triggers a web search for additional content.
5. **Web Search / LLM Generate:** Depending on the decision, either a web search is conducted, or the LLM generates an answer based on the graded documents.
6. **End:** The final response is returned to the user.


## **Conclusion**

The **LangGraph-Powered RAG System** is an innovative solution for **Intelligent Question Answering**, leveraging both **local document retrieval** and **real-time web search augmentation**. By utilizing **LangGraph's graph-based flow** for decision-making and task orchestration, the system ensures efficient, accurate, and contextually relevant responses.

## output

![image](https://github.com/user-attachments/assets/3048a638-4557-4be2-9517-3a3e59b91a88)


