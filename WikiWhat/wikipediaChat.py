# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:09:37 2023

@author: marca
"""

import pandas as pd
from wikipediaContextQAHelpers import *

def main():
    title = input("Please enter a Wikipedia page title: ")
    keywords = get_keywords(title)
    save_data = input("Do you want the chat data to be saved? (yes/no): ")
    save_data = save_data.strip().lower() == "yes"
    output_path = None
    input_path = None
    
    if save_data:
        output_path = input("Please enter the output path for the data: ")
        print("Gathering the background data for this chat.  For more narrow focus wikipedia pages, this could take a couple minutes.  For wider focus, it could take several hours.")
    else:
        has_data = input("Do you already have chat data saved? (yes/no): ")
        has_data = has_data.strip().lower() == "yes"
        if has_data:
            input_path = input("Please enter the file path for the existing data: ")
    
    if input_path:
        df = pd.read_csv(input_path)
    else:
        pages = find_related_pages(title, keywords)
        df = create_dataframe(pages, output_filename=output_path)
    
    print(f"Ok, I'm ready for your questions about {title}.")
    
    while True:
        question = input("Question: ")
        
        if question.lower() == "exit":
            break
        
        answer = answer_query_with_context(question, df)
        print(f"Answer: {answer}")
        
        follow_up = input('Press enter to ask another question, or enter "exit" to exit: ')
        
        if follow_up.lower() == "exit":
            break
    
    print(f"I hope you learned something about {title}! Goodbye!")

if __name__ == "__main__":
    main()
