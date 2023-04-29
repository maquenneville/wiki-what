# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 01:09:37 2023

@author: marca
"""

from wikipediaContextQAHelpers import *

    
    
def main():
    # Set initial state and select the simple answer agent
    exit_program = False
    answer_agent = simple_answer_agent
    spinner = Spinner()

    # Main loop for interacting with the user
    while not exit_program:
        title = input("Please enter a Wikipedia page title (type 'exit' to quit): ")
        if title.lower() == "exit":
            break

        # Check if the topic's data is already in Pinecone
        has_data = check_topic_exists_in_pinecone(title)
        storage_title = title.replace(" ", "_").lower()

        if has_data:
            print(f"\n\nTopic available")

        if not has_data:
            # If the topic is not in Pinecone, gather data, calculate embeddings and store them in Pinecone
            print("\n\nGathering the background data for this chat, calculating its embeddings, and loading them into Pinecone. For more narrow focus Wikipedia pages, this could take a couple of minutes. For wider focus, it could take a while.")
            pages = find_related_pages(title)
            page_titles = [page.title for page in pages]
            df = create_dataframe(pages)
            store_embeddings_in_pinecone(dataframe=df, topic_name=storage_title)
            save_list_to_txt_file(PAGES_RECORD, page_titles)

        print(f"\n\nOk, I'm ready for your questions about {title}.\n\n")

        # Inner loop for processing user commands and questions
        while True:
            command = input("Enter a question or a command (enter 'help' for additional commands): ").lower()

            # Exit the program
            if command == "exit":
                exit_program = True
                break

            # Switch to another topic
            if command == "switch topic":
                break

            # Switch to the smart answer agent (GPT-4)
            if command == "smart answer":
                answer_agent = smart_answer_agent
                print("Switched to smart answer agent.\n")
                continue

            # Switch to the simple answer agent (GPT-3.5-turbo)
            if command == "simple answer":
                answer_agent = simple_answer_agent
                print("Switched to simple answer agent.\n")
                continue

            # Show the help text for available commands
            if command == "help":
                print("""
                      Commands:

                          switch topic: takes you back to enter a new Wikipedia page
                          
                          smart answer: switch to GPT-4 for question-answering
                                              (warning: more expensive, use only for nuanced questions)
                                              
                          simple answer: switch back to GPT-3.5-turbo for question-answering
                                              (good for most questions)
                                              
                          exit: quit program

                      """)
                continue

            else:
                # Start the spinner to indicate processing
                spinner.start()
                
                # Fetch context from Pinecone and answer the question
                answer = answer_agent(command)
                
                # Stop the spinner and display the answer
                spinner.stop()
                print(f"\n\nAnswer: {answer}\n\n")

        if not exit_program:
            print(f"\n\nI hope you learned something about {title}!\n\n")

    print("Goodbye!")




if __name__ == "__main__":
    main()
