import os
import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

CHROMA_PATH = "chroma"
CHROMA_PATH_CMP_1 = "chroma-cmp-1"
CHROMA_PATH_CMP_2 = "chroma-cmp-2"

PROMPT_TEMPLATE = """
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


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Sample parameters are query_model=gpt-3.5-turbo-0125;embed_model=text-embedding-3-small
query_model = os.getenv('query_model')
embed_model = os.getenv('embed_model')


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
    # embedding_function =  get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=OpenAIEmbeddings(model=embed_model, openai_api_key=OPENAI_API_KEY))

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=4)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Use the below if we use OpenAI
    model = ChatOpenAI(model=query_model, openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    # print(formatted_response)
    return response_text


def query_two_dbs(query_text: str):
    """
    Sample Query: "How do Snowflake and Redshift handle data collaboration?"
    :param query_text:
    :return:
    """

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings(model=embed_model, openai_api_key=OPENAI_API_KEY)
    db1 = Chroma(persist_directory=CHROMA_PATH_CMP_1, embedding_function=embedding_function)
    db2 = Chroma(persist_directory=CHROMA_PATH_CMP_2, embedding_function=embedding_function)

    # Search the DB for our query
    results1 = db1.similarity_search_with_score(query_text, k=2)
    results2 = db2.similarity_search_with_score(query_text, k=2)
    results = results1 + results2

    context_text_1 = "\n\n---\n\n".join([doc.page_content for doc, _score in results1])
    context_text_2 = "\n\n---\n\n".join([doc.page_content for doc, _score in results2])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_CMP)
    prompt = prompt_template.format(context_1=context_text_1, context_2=context_text_2, question=query_text)
    print(f"The prompt to be sent to the LLM is: {prompt}")

    # Use the below if we use OpenAI
    model = ChatOpenAI(model=query_model, openai_api_key=OPENAI_API_KEY)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
