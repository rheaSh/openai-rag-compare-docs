import os
import shutil
import argparse
from langchain_openai import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

FILE_DB_MAPPING = [("chroma-cmp-1", "data/services-doc1"),
                   ("chroma-cmp-2", "data/services-doc2")]

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Sample value is EMBED_MODEL=text-embedding-3-small
embed_model = os.getenv('EMBED_MODEL')


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    for chroma_path, data_path in FILE_DB_MAPPING:
        # Create (or update) the data store.
        documents = load_documents(data_path)
        chunks = split_documents(documents)
        add_to_chroma(chunks, chroma_path)


def load_documents(datapath: str):
    document_loader = PyPDFDirectoryLoader(datapath)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], chroma_path: str):
    # Load the existing database.
    embedding_function = OpenAIEmbeddings(model=embed_model, openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB {chroma_path}: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"Sample document added to ChromaDB {chroma_path} is: {new_chunks[1]}")
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print(f"✅ No new documents to add to DB {chroma_path}")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    for chroma_path, data_path in FILE_DB_MAPPING:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)


if __name__ == "__main__":
    main()
