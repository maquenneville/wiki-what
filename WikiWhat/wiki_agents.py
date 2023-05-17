# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:20:40 2023

@author: marca
"""


from wiki_tools import *
from openai_pinecone_tools import *


def construct_simple_prompt(question: str):
    """
    Fetch relevant
    """
    most_relevant_document_sections = fetch_context_from_pinecone(question)

    simple_prompt = """Answer the question as truthfully as possible using the provided context blocks. Make your answers highly informative and detailed.  If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know."\n\nContext:\n"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": simple_prompt},
    ]

    for section in most_relevant_document_sections:
        messages.append({"role": "user", "content": f"Context: {section}"})

    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    return messages


def construct_smart_prompt(question: str):
    """
    Fetch relevant
    """
    most_relevant_document_sections = fetch_context_from_pinecone(question)

    smart_prompt = """Answer the question using the provided context blocks. If the question requires a degree of nuance and subjectivity to answer, do your best to give an informative and nuanced answer.  If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know."\n\nContext:\n"""

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": smart_prompt},
    ]

    for section in most_relevant_document_sections:
        messages.append({"role": "user", "content": f"Context: {section}"})

    messages.append({"role": "user", "content": f"Q: {question}\nA:"})

    return messages


def simple_answer_agent(query: str, model=FAST_CHAT_MODEL, show_prompt=False):
    messages = construct_simple_prompt(query)

    if show_prompt:
        print(messages)

    response = generate_response(
        messages, temperature=0.2, n=1, max_tokens=2000, frequency_penalty=0
    )
    return response.strip(" \n")


def smart_answer_agent(query: str, model=SMART_CHAT_MODEL, show_prompt=False):
    messages = construct_smart_prompt(query)

    if show_prompt:
        print(messages)

    response = generate_response(
        messages,
        model=model,
        temperature=0.4,
        n=1,
        max_tokens=3000,
        frequency_penalty=0,
    )
    return response.strip(" \n")


def broad_decision_agent(query):
    # Generate ChatGPT messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "You are my Braod Knowledge Decision Assistant.  Provided a query and relevant context, your job is to provide an answer to the query.  The answers should be nuanced and well-articulated, using the context and your own extensive knowledge of English Literature.  Assume the one asking the question has a grad-school level understanding of English Literature.",
        },
    ]

    messages.append({"role": "user", "content": f"Query:\n{query}"})

    # Use ChatGPT to generate a Wolfram Alpha natural language query
    answer = generate_response(messages, temperature=0.4)

    return answer
