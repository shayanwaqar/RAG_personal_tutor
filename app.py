import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from config import OPENAI_API_KEY

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Step 1: Document Ingestion
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents

# Step 2: Split Documents for Embedding
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

# Step 3: Generate and Store Embeddings
def create_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Step 4: Set Up Retrieval and Q&A
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    qa_chain = create_retrieval_chain(retriever=retriever, model=llm)
    return qa_chain

# Main Execution
def main():
    # Load and process documents
    folder_path = "study_materials"  # Correct relative path
    raw_documents = load_documents(folder_path)
    processed_documents = split_documents(raw_documents)

    # Create vectorstore
    vectorstore = create_vectorstore(processed_documents)

    # Create QA chain
    qa_chain = create_qa_chain(vectorstore)

    # Interact with the assistant
    print("Study Assistant Ready! Ask your questions:")
    while True:
        query = input("\nYour Question (type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = qa_chain.run(query)
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()
