# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:51:57 2023

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

class WikiGather:
    discard_categories = [
        "See also", "References", "External links", "Further reading", "Footnotes",
        "Bibliography", "Sources", "Citations", "Literature", "Footnotes",
        "Notes and references", "Photo gallery", "Works cited", "Photos",
        "Gallery", "Notes", "References and sources", "References and notes", "ISBN",
    ]

    def __init__(self):
        self.relevant_pages = []
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.wiki_chunks = []
        self.page_record = self._load_list_from_txt_file("wiki_page_record.txt")
        
    def _count_tokens(self, text: str):
        tokens = len(self.encoding.encode(text))
        return tokens
    
    def _save_list_to_txt_file(self, file_name, input_list):
        with open(file_name, "a", encoding="utf-8", errors="ignore") as file:
            for item in input_list:
                file.write(str(item) + ",")


    def _load_list_from_txt_file(self, file_name):
        with open(file_name, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
            if not content:  # Check if the content is empty
                return []
            items = content.split(",")
            # Remove the last element from the list, as it will be empty due to the trailing comma
            items.pop()
            return items

    def get_wiki_page(self, title: str):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return page, False
        except wikipedia.DisambiguationError as e:
            try:
                return wikipedia.page(e.options[0], auto_suggest=False), True
            except wikipedia.DisambiguationError:
                return None, False
        except Exception:
            return None, False

    def find_related_pages(self, title, depth=2):
        initial_page, _ = self.get_wiki_page(title)
        titles_so_far = [title]
        linked_pages = self._recursively_find_all_pages(
            initial_page.links, titles_so_far, depth - 1
        )
        total_pages = [initial_page] + linked_pages

        return total_pages

    def _recursively_find_all_pages(self, titles, titles_so_far, depth=2):
        if depth <= 0:
            return []
        pages = []
        for title in titles:
            if title not in titles_so_far and title not in self.page_record:
                titles_so_far.append(title)
                page, is_disambiguation = self.get_wiki_page(title)
                if page is None:
                    continue
                print(title)
                pages.append(page)
                if not is_disambiguation:
                    new_pages = self._recursively_find_all_pages(
                        page.links, titles_so_far, depth - 1
                    )
                    pages.extend(new_pages)

        return pages

    def _reduce_long(self, long_text: str, long_text_tokens: bool = False, max_len: int = 590) -> str:
        # Assume count_tokens and sent_tokenize are imported or defined elsewhere
        if not long_text_tokens:
            long_text_tokens = self._count_tokens(long_text)
        if long_text_tokens > max_len:
            sentences = sent_tokenize(long_text.replace("\n", " "))
            ntokens = 0
            for i, sentence in enumerate(sentences):
                ntokens += 1 + self._count_tokens(sentence)
                if ntokens > max_len:
                    return ". ".join(sentences[:i]) + "."

        return long_text
    
    
    def extract_sections(
        self,
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
        outputs = [(title, "Summary", cont, self._count_tokens(cont) + 4)]
    
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
            self._count_tokens(c)
            + 3
            + self._count_tokens(" ".join(h.split(" ")[1:-1]))
            - (1 if len(c) == 0 else 0)
            for h, c in zip(nheadings, ncontents)
        ]
    
        # Combine the title, heading, content, and token count for each section, ensuring the content does not exceed the max_len
        outputs += [
            (title, h, c, t)
            if t < max_len
            else (title, h, self._reduce_long(c, max_len=max_len), max_len)
            for h, c, t in zip(nheadings, ncontents, ncontent_ntokens)
        ]
        
        content_outputs = [output[2] for output in outputs]
        return content_outputs
    
        
    def gather(self, title):
        
        self.relevant_pages = self.find_related_pages(title)
        
        for page in self.relevant_pages:
            print(page.title)
            self.wiki_chunks += self.extract_sections(page.content, page.title)
            
        return "Wiki context gathered"
            
    def dump(self):
        
        page_titles = [page.title for page in self.relevant_pages]
        
        wiki_stuff = self.wiki_chunks.copy()
        self.wiki_chunks = []
        
        self._save_list_to_txt_file("wiki_page_record.txt", page_titles)
        self.page_record = self._load_list_from_txt_file("wiki_page_record.txt")
        self.relevant_pages = []
        
        
        return wiki_stuff
            
            
            
            