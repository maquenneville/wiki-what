# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:21:38 2023

@author: marca
"""


import time
import wikipedia
import re
from typing import Set
from nltk.tokenize import sent_tokenize
import tiktoken
import os
import sys
import configparser
import threading


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
    namespace = config.get("API_KEYS", "Pinecone_Namespace")
    index = config.get("API_KEYS", "Pinecone_Index")

    return openai_api_key, pinecone_api_key, pinecone_env, namespace, index


openai_api_key, pinecone_api_key, pinecone_env, namespace, index = get_api_keys(
    "config.ini"
)


SMART_CHAT_MODEL = "gpt-4"
FAST_CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
PINECONE_NAMESPACE = namespace
PINECONE_API_KEY = pinecone_api_key
PINECONE_ENV = pinecone_env
PAGES_RECORD = "wiki_page_record.txt"
PINECONE_INDEX = index
OPENAI_API_KEY = openai_api_key


page_cache = {}

wikipedia.set_lang("en")
wikipedia.set_user_agent(
    "wikipediaapi (https://github.com/wikipedia-api/wikipedia-api)"
)


def save_list_to_txt_file(file_name, input_list):
    with open(file_name, "a", encoding="utf-8", errors="ignore") as file:
        for item in input_list:
            file.write(str(item) + ",")


def load_list_from_txt_file(file_name):
    with open(file_name, "r", encoding="utf-8", errors="ignore") as file:
        content = file.read()
        if not content:  # Check if the content is empty
            return []
        items = content.split(",")
        # Remove the last element from the list, as it will be empty due to the trailing comma
        items.pop()
        return items


saved_pages = load_list_from_txt_file(PAGES_RECORD)


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
    linked_pages = recursively_find_all_pages(
        initial_page.links, titles_so_far, depth - 1
    )
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
                new_pages = recursively_find_all_pages(
                    page.links, titles_so_far, depth - 1
                )
                pages.extend(new_pages)

    return pages


# Reduces the length of long text to the maximum token length specified
def reduce_long(
    long_text: str, long_text_tokens: bool = False, max_len: int = 590
) -> str:
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


# List of categories to be discarded when extracting sections from the wiki_text
discard_categories = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
    "ISBN",
]


# Function to extract sections from the wiki_text based on the specified conditions
def extract_sections(
    wiki_text: str,
    title: str,
    max_len: int = 1500,
    discard_categories: Set[str] = discard_categories,
) -> str:
    if len(wiki_text) == 0:
        return []

    # Identify headings in the wiki_text
    headings = re.findall("==+ .* ==+", wiki_text)
    for heading in headings:
        wiki_text = wiki_text.replace(heading, "==+ !! ==+")
    contents = wiki_text.split("==+ !! ==+")
    contents = [c.strip() for c in contents]
    assert len(headings) == len(contents) - 1

    # Process the first content section
    cont = contents.pop(0).strip()
    outputs = [(title, "Summary", cont, count_tokens(cont) + 4)]

    max_level = 100
    keep_group_level = max_level
    remove_group_level = max_level
    nheadings, ncontents = [], []

    # Iterate through the headings and contents, filtering out discard categories
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

    # Calculate the token count for each content section
    ncontent_ntokens = [
        count_tokens(c)
        + 3
        + count_tokens(" ".join(h.split(" ")[1:-1]))
        - (1 if len(c) == 0 else 0)
        for h, c in zip(nheadings, ncontents)
    ]

    # Combine the title, heading, content, and token count for each section, ensuring the content does not exceed the max_len
    outputs += [
        (title, h, c, t)
        if t < max_len
        else (title, h, reduce_long(c, max_len=max_len), max_len)
        for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)
    ]

    return outputs


class Spinner:
    def __init__(self, message="Thinking..."):
        self._message = message
        self._running = False
        self._spinner_thread = None

    def start(self):
        self._running = True
        self._spinner_thread = threading.Thread(target=self._spin)
        self._spinner_thread.start()

    def stop(self):
        self._running = False
        self._spinner_thread.join()

    def _spin(self):
        spinner_chars = "|/-\\"
        index = 0

        while self._running:
            sys.stdout.write(
                f"\r{self._message} {spinner_chars[index % len(spinner_chars)]}"
            )
            sys.stdout.flush()
            time.sleep(0.1)
            index += 1

        # Clear the spinner line
        sys.stdout.write("\r" + " " * (len(self._message) + 2))
        sys.stdout.flush()
