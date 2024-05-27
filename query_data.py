import os
import argparse
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores.chroma import Chroma

GAME_CHROMA_PATH = "chroma"
CHROMA_PATH_CMP_1 = "chroma-cmp-1"
CHROMA_PATH_CMP_2 = "chroma-cmp-2"

PROMPT_TEMPLATE_SINGLE_DOC = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
PROMPT_TEMPLATE_CMP = """
Analyze and compare the below two contexts from different documents:

Context 1:
{context_1}

Context 2:
{context_2}
---

Summarize the differences relevant to the question from the above contexts: {question} 
"""

# Get environmental variables
# Current parameters are QUERY_MODEL=gpt-3.5-turbo-0125; EMBED_MODEL=text-embedding-3-small
QUERY_MODEL = os.getenv('QUERY_MODEL')
EMBED_MODEL = os.getenv('EMBED_MODEL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text")
    args = parser.parse_args()
    query_text = args.query_text
    # query_rag(query_text)
    query_two_dbs(query_text)


def query_rag(query_text: str):
    """
    Sample Query: "How many players can play Catan?"
    :param query_text:
    :return:
    """

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=GAME_CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)

    # Format the question asked and the search results into the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_SINGLE_DOC)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Query the OpenAI model to get a response to prompt
    model = ChatOpenAI(model=QUERY_MODEL, openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)

    # Format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response)

    return formatted_response


def query_two_dbs(query_text: str):
    """
    Sample Query: "How do Snowflake and Redshift handle data collaboration?"
    :param query_text:
    :return:
    """

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
    db1 = Chroma(persist_directory=CHROMA_PATH_CMP_1, embedding_function=embedding_function)
    db2 = Chroma(persist_directory=CHROMA_PATH_CMP_2, embedding_function=embedding_function)

    # Search the DB for our query
    results1 = db1.similarity_search_with_score(query_text, k=2)
    results2 = db2.similarity_search_with_score(query_text, k=2)
    results = results1 + results2

    # Format the search results as the contexts needed for the prompt
    context_text_1 = "\n\n---\n\n".join([doc.page_content for doc, _score in results1])
    context_text_2 = "\n\n---\n\n".join([doc.page_content for doc, _score in results2])

    # Format the question asked into the context for the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_CMP)
    prompt = prompt_template.format(context_1=context_text_1, context_2=context_text_2, question=query_text)
    print(f"The prompt to be sent to the LLM is: {prompt}")

    # Query the OpenAI model to get a response to prompt
    model = ChatOpenAI(model=QUERY_MODEL, openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)

    # Format the response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"

    print(formatted_response)

    return formatted_response


if __name__ == "__main__":
    main()
