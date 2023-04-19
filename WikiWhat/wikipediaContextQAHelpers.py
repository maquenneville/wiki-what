# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:04:44 2023

@author: marca
"""

import pandas as pd
import time
from requests.exceptions import RequestException
import wikipedia
from wikipedia.exceptions import PageError
import requests
import re
from typing import Set
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import tiktoken
import openai
from openai.error import RateLimitError, InvalidRequestError, APIError
import os
import sys
import configparser
import pinecone
from pinecone import PineconeProtocolError
import csv


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")



def count_tokens(text):
    tokens = len(encoding.encode(text))
    return tokens

def get_api_keys(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    openai_api_key = config.get("API_KEYS", "OpenAI_API_KEY")
    pinecone_api_key = config.get("API_KEYS", "Pinecone_API_KEY")
    pinecone_env = config.get("API_KEYS", "Pinecone_ENV")
    namespace = config.get("API_KEYS", "Namespace")

    return openai_api_key, pinecone_api_key, pinecone_env, namespace

openai_api_key, pinecone_api_key, pinecone_env, namespace = get_api_keys('config.ini')

openai.api_key = openai_api_key


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_NAMESPACE = namespace
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENV = pinecone_env
PAGES_RECORD = 'wiki_page_record.txt'


page_cache = {}

wikipedia.set_lang("en")
wikipedia.set_user_agent("wikipediaapi (https://github.com/wikipedia-api/wikipedia-api)")



def save_list_to_txt_file(file_name, input_list):
    with open(file_name, 'a') as file:
        for item in input_list:
            file.write(str(item) + ",")

def load_list_from_txt_file(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        if not content:  # Check if the content is empty
            return []
        items = content.split(",")
        # Remove the last element from the list, as it will be empty due to the trailing comma
        items.pop()
        return items


saved_pages = load_list_from_txt_file(PAGES_RECORD)


def get_keywords(title):
    filler_words = set(stopwords.words('english'))
    words = title.lower().split()
    filtered_words = [word for word in words if word not in filler_words]
    return filtered_words



def filter_titles(titles, keywords):
    """
    Get the titles which are related to the given keywords
    """
    titles = [title for title in titles if any(keyword.lower() in title.lower() for keyword in keywords)]
    return titles


def get_wiki_page(title: str):
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page, False
    except wikipedia.DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False), True
    except Exception:
        return None, False



def find_related_pages(title, depth=2):

    initial_page, _ = get_wiki_page(title)
    titles_so_far = [title]
    linked_pages = recursively_find_all_pages(initial_page.links, titles_so_far, depth-1)
    total_pages = [initial_page] + linked_pages

    return total_pages


def recursively_find_all_pages(titles, titles_so_far, depth=2):
    
    global saved_pages

    if depth <= 0:
        return []
    pages = []
    for title in titles:
        if title not in titles_so_far and title not in saved_pages:
            titles_so_far.append(title)
            page, is_disambiguation = get_wiki_page(title)
            if page is None:
                continue
            print(title)
            pages.append(page)
            if not is_disambiguation:
                new_pages = recursively_find_all_pages(page.links, titles_so_far, depth-1)
                pages.extend(new_pages)

    return pages



def reduce_long(long_text: str, long_text_tokens: bool = False, max_len: int = 590) -> str:
    if not long_text_tokens:
        long_text_tokens = count_tokens(long_text)
    if long_text_tokens > max_len:
        sentences = sent_tokenize(long_text.replace("\n", " "))
        ntokens = 0
        for i, sentence in enumerate(sentences):
            ntokens += 1 + count_tokens(sentence)
            if ntokens > max_len:
                return ". ".join(sentences[:i]) + "."

    return long_text

discard_categories = ['See also', 'References', 'External links', 'Further reading', "Footnotes",
    "Bibliography", "Sources", "Citations", "Literature", "Footnotes", "Notes and references",
    "Photo gallery", "Works cited", "Photos", "Gallery", "Notes", "References and sources",
    "References and notes",]

def extract_sections(
    wiki_text: str,
    title: str,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,
) -> str:
    if len(wiki_text) == 0:
        return []

    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont)+4)]

    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []
    for heading, content in zip(headings, contents):
        plain_heading = " ".join(heading.split(" ")[1:-1])
        num_equals = len(heading.split(" ")[0])
        if num_equals <= keep_group_level:
            keep_group_level = max_level

        if num_equals > remove_group_level:
            if num_equals <= keep_group_level:
                continue
        keep_group_level = max_level
        if plain_heading in discard_categories:
            remove_group_level = num_equals
            keep_group_level = max_level
            continue
        nheadings.append(heading.replace("=", "").strip())
        ncontents.append(content)
        remove_group_level = max_level

    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    outputs += [(title, h, c, t) if t < max_len
                else (title, h, reduce_long(c, max_len=max_len), max_len)
                for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)]

    return outputs



def get_embedding(text: str, model: str=EMBEDDING_MODEL):
    while True:
        try:
            result = openai.Embedding.create(
              model=model,
              input=text
            )
            break
        except (APIError, RateLimitError):
            print("OpenAI had an issue, trying again in a few seconds...")
            time.sleep(10)
    return result["data"][0]["embedding"]



def compute_doc_embeddings(df: pd.DataFrame, model: str = EMBEDDING_MODEL):
    embeddings = []
    segment = 0
    for _, row in df.iterrows():
        text = row["title"] + " " + row["heading"] + " " + row["content"]
        embedding = get_embedding(text, model)
        embeddings.append(embedding)
        segment += 1
        print(f"Embedded segment {segment}")

    embedding_columns = {f"embedding{idx}": [embedding[idx] for embedding in embeddings] for idx in range(len(embeddings[0]))}
    embedding_df = pd.DataFrame(embedding_columns)
    df = pd.concat([df, embedding_df], axis=1)

    return df





def load_embeddings(df: pd.DataFrame):
    """
    Read the document embeddings and their keys from a DataFrame.
    
    The DataFrame should have columns "title", "heading", and "embedding".
    """

    return {
        (row['title'], row['heading']): row['embedding']
        for _, row in df.iterrows()
    }




def create_dataframe(pages, output_filename=None):
    res = []
    for page in pages:
        print(page.title)
        res += extract_sections(page.content, page.title)

    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
    df = df[df.tokens > 40]
    df = df.drop_duplicates(["title", "heading"])
    df = df.reset_index().drop("index", axis=1)
    df = compute_doc_embeddings(df)

    if output_filename:
        df.to_csv(output_filename, index=False)

    return df



def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))



def order_document_sections_by_query_similarity(query, df):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    embedding_cols = [col for col in df.columns if col.startswith("embedding")]

    document_similarities = sorted([
        (
            vector_similarity(query_embedding, [row[col] for col in embedding_cols]),
            index
        )
        for index, row in df.iterrows()
    ], reverse=True)

    return document_similarities

### PINECONE FUNCTIONS ###

def store_embeddings_in_pinecone(namespace=PINECONE_NAMESPACE, pinecone_api_key=PINECONE_API_KEY, pinecone_env=PINECONE_ENV, csv_filepath=None, topic_name=None, dataframe=None):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Instantiate Pinecone's Index
    pinecone_index = pinecone.Index(index_name=namespace)
    
    print(len(dataframe.columns))
    
    if dataframe is not None and not dataframe.empty:
        batch_size = 80
        vectors_to_upsert = []
        batch_count = 0

        for index, row in dataframe.iterrows():
            context_chunk = row["content"]
            print(context_chunk)
            vector = [float(row[f"embedding{i}"]) for i in range(1536)]
            
            idx = f"wiki_{index}"
            topic_name = f"wiki_{topic_name}"
            vectors_to_upsert.append({
                "id": idx,
                "values": vector,
                "metadata": {"topic_name": topic_name, "context": context_chunk}
            })

            # Upsert when the batch is full or it's the last row
            if len(vectors_to_upsert) == batch_size or index == len(dataframe) - 1:
                while True:
                    print(f"Attempting upsert for batch {batch_count + 1}...") 
                    try:
                        upsert_response = pinecone_index.upsert(
                            vectors=vectors_to_upsert,
                            namespace=namespace
                        )

                        print(f"Upserted batch {batch_count + 1}: {len(vectors_to_upsert)} vectors")
                        batch_count += 1
                        vectors_to_upsert = []
                        break

                    except pinecone.core.client.exceptions.ApiException:
                        print("Pinecone is a little overwhelmed, trying again in a few seconds...")
                        time.sleep(10)

    elif csv_filepath:
        # Read embeddings and metadata from the CSV file
        with open(csv_filepath, newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip the header row

            batch_size = 80
            vectors_to_upsert = []
            batch_count = 0

            for row in reader:
                row_dict = {col_name: value for col_name, value in zip(header, row)}
                idx, context_chunk = row_dict["index"], row_dict["context"]
                vector = [float(row_dict[f"embedding{i}"]) for i in range(len(row) - 2)]

                
                topic_name = f"wiki_{topic_name}"
                vectors_to_upsert.append({
                    "id": idx,
                    "values": vector,
                    "metadata": {"topic_name": topic_name, "context": context_chunk}
                })

                # Upsert when the batch is full or it's the last row
                if len(vectors_to_upsert) == batch_size or idx == header[-1]:
                    upsert_response = index.upsert(
                        vectors=vectors_to_upsert,
                        namespace=namespace
                    )

                    print(f"Upserted batch {batch_count + 1}: {len(vectors_to_upsert)} rows")
                    batch_count += 1
                    vectors_to_upsert = []
    else:
        print("No CSV filepath to retrieve embeddings")


def fetch_context_from_pinecone(query, topic_name, top_n=5, namespace=PINECONE_NAMESPACE, pinecone_api_key=PINECONE_API_KEY, pinecone_env=PINECONE_ENV):
    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    # Generate the query embedding
    query_embedding = get_embedding(query)

    # Query Pinecone for the most similar embeddings
    pinecone_index = pinecone.Index(index_name=namespace)
    
    while True:
        try:
            query_response = pinecone_index.query(
                namespace=namespace,
                top_k=top_n,
                include_values=False,
                include_metadata=True,
                vector=query_embedding,
                filter= {"topic_name": topic_name}
            )
            break
        
        except PineconeProtocolError:
            print("Pinecone needs a moment....")
            time.sleep(3)
            continue
    print(query_response)
    # Retrieve metadata for the relevant embeddings
    context_chunks = [match['metadata']['context'] for match in query_response['matches']]

    return context_chunks



def check_topic_exists_in_pinecone(topic_name: str, namespace: str=PINECONE_NAMESPACE, pinecone_api_key: str=PINECONE_API_KEY, pinecone_env: str=PINECONE_ENV) -> bool:
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    topic_name = f"wiki_{topic_name}"
    metadata_filter = {"topic_name": topic_name}

    index = pinecone.Index(namespace)
    query_embedding = get_embedding(topic_name)

    query_response = index.query(
        namespace=namespace,
        top_k=1,
        include_values=False,
        include_metadata=False,
        filter=metadata_filter,
        vector=query_embedding
    )



    return len(query_response['matches']) != 0


### END PINECONE

def generate_response(
    messages, temperature=0.5, n=1, max_tokens=4000, frequency_penalty=0
):

    model_engine = "gpt-3.5-turbo"

    # Calculate the number of tokens in the messages
    tokens_used = sum([count_tokens(msg["content"]) for msg in messages])
    tokens_available = 4096 - tokens_used

    # Adjust max_tokens to not exceed the available tokens
    max_tokens = min(max_tokens, (tokens_available - 100))

    # Reduce max_tokens further if the total tokens exceed the model limit
    if tokens_used + max_tokens > 4096:
        max_tokens = 4096 - tokens_used - 10

    if max_tokens < 1:
        max_tokens = 1

    # Generate a response
    max_retries = 10
    retries = 0
    while True:
        if retries < max_retries:
            try:
                completion = openai.ChatCompletion.create(
                    model=model_engine,
                    messages=messages,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                )
                break
            except (RateLimitError, KeyboardInterrupt):
                time.sleep(60)
                retries += 1
                print("Server overloaded, retrying in a minute")
                continue
        else:
            print("Failed to generate prompt after max retries")
            return
    response = completion.choices[0].message.content
    return response



def construct_prompt(
    question: str,
    df: pd.DataFrame,
    separator: str = "\n*",
    max_section_len: int = 1000,  # Leave some space for the header and question
):
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(
        question, df
    )

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        document_section = df.loc[section_index]

        chosen_sections_len += document_section.tokens + len(separator)
        if chosen_sections_len > max_section_len:
            break

        chosen_sections.append(separator + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))

    header = (
        """Answer the question as truthfully as possible using the provided context. If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know."\n\nContext:\n"""
    )

    context = header + "".join(chosen_sections)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": context},
        {"role": "user", "content": f"Q: {question}\nA:"},
    ]

    return messages


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    show_prompt=False
):
    messages = construct_prompt(query, df)

    if show_prompt:
        print(messages)

    response = generate_response(messages, temperature=0.5, n=1, max_tokens=1000, frequency_penalty=0)
    return response.strip(" \n")


