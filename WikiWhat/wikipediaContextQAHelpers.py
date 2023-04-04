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
from openai.error import RateLimitError, InvalidRequestError
import os
import sys
import configparser


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")



def count_tokens(text):
    tokens = len(encoding.encode(text))
    return tokens

def get_openai_api_key():
    if getattr(sys, "frozen", False):
        # If the script is running as a PyInstaller-generated .exe
        exe_dir = os.path.dirname(sys.executable)
    else:
        # If the script is running as a regular Python script
        exe_dir = os.path.dirname(os.path.realpath(__file__))

    config = configparser.ConfigParser()
    config_path = os.path.join(exe_dir, "config.ini")
    config.read(config_path)

    openai_api_key = config.get("OpenAI", "api_key")
    return openai_api_key


# Set up the OpenAI API client
openai.api_key = get_openai_api_key()


CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_API_PARAMS = {
    "engine": "gpt-3.5-turbo",
    "temperature": 0.5,
    "max_tokens": 100,
    "n": 1,
    "stop": None,
}

page_cache = {}

wikipedia.set_lang("en")
wikipedia.set_user_agent("wikipediaapi (https://github.com/wikipedia-api/wikipedia-api)")




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


def get_wiki_page(title):
    """
    Get the Wikipedia page given a title.
    """
    try:
        return wikipedia.page(title, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        return wikipedia.page(e.options[0], auto_suggest=False)
    except wikipedia.exceptions.PageError as e:
        return None


def find_related_pages(title, keywords):
    initial_page = get_wiki_page(title)
    if initial_page is None:
        return []

    titles_so_far = {title}
    all_pages = [initial_page]
    linked_pages = recursively_find_all_pages(initial_page.links, titles_so_far, keywords)

    return all_pages + linked_pages


def recursively_find_all_pages(titles, titles_so_far, keywords):
    all_pages = []

    titles = list(set(titles) - titles_so_far)
    titles = filter_titles(titles, keywords)
    titles_so_far.update(titles)
    for title in titles:
        try:
            page = get_wiki_page(title)
        except PageError:
            continue
        
        if page is None:
            continue
        all_pages.append(page)

        new_pages = recursively_find_all_pages(page.links, titles_so_far, keywords)
        for pg in new_pages:
            if pg.title not in [p.title for p in all_pages]:
                all_pages.append(pg)
        titles_so_far.update(page.links)
    return all_pages


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
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]



def compute_doc_embeddings(df: pd.DataFrame, model: str = EMBEDDING_MODEL):
    embeddings = []
    for _, row in df.iterrows():
        text = row["title"] + " " + row["heading"] + " " + row["content"]
        embedding = get_embedding(text, model)
        embeddings.append(embedding)
        time.sleep(1)

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
    separator: str = "\n",
    max_section_len: int = 2000,  # Leave some space for the header and question
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


