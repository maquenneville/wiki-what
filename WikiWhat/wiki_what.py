# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 23:26:03 2023

@author: marca
"""

from typing import List
from chroma_memory import ChromaMemory
from wiki_gather import WikiGather
from simple_bot import SimpleBot
import threading
import sys
import time

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
        

def main():
    # Set initial state and select the simple answer agent
    exit_program = False
    spinner = Spinner()
    bot = SimpleBot("""Answer the question using the provided context blocks. If the question requires a degree of nuance and subjectivity to answer, do your best to give an informative and nuanced answer.  If the answer is not contained within the text below, attempt to use the context and your knowledge to give an answer.  If the context cannot help you find an answer, say "I don't know."\n\nContext:\n""")
    bot.fast_agent() # Using GPT-3.5-turbo as the default
    wiki_gather = WikiGather()
    chroma_memory = ChromaMemory(collection_name="wiki_collection")

    # Main loop for interacting with the user
    while not exit_program:
        title = input(
            "Please enter a Wikipedia page title (type 'exit' to quit, 'skip' to skip data gathering): "
        )
        if title.lower() == "exit":
            break

        skip_data_gathering = title.lower() == "skip"

        if not skip_data_gathering:
            # Gather data, calculate embeddings, and store them in Chroma
            print(
                "\n\nGathering the background data for this chat, calculating its embeddings, and loading them into Chroma. This could take some time depending on the topic's complexity.\n"
            )
            wiki_gather.gather(title)
            wiki_chunks = wiki_gather.dump()
            chroma_memory.store(wiki_chunks)

            print(f"\n\nOk, I'm ready for your questions about {title}.\n\n")

        # Inner loop for processing user commands and questions
        while True:
            command = input(
                "Enter a question or a command (enter 'help' for additional commands): "
            ).lower()

            # Exit the program
            if command == "exit":
                exit_program = True
                break

            # Switch to another topic
            if command == "switch topic":
                break

            # Switch to the smart answer agent (GPT-4)
            if command == "smart answer":
                bot.smart_agent()
                print("Switched to smart answer agent.\n")
                continue

            # Switch to the simple answer agent (GPT-3.5-turbo)
            if command == "simple answer":
                bot.fast_agent()
                print("Switched to simple answer agent.\n")
                continue

            # Show the help text for available commands
            if command == "help":
                print(
                    """
                      Commands:
                          switch topic: takes you back to enter a new Wikipedia page
                          smart answer: switch to GPT-4 for question-answering
                                        (warning: more expensive, use only for nuanced questions)
                          simple answer: switch back to GPT-3.5-turbo for question-answering
                                        (good for most questions)
                          exit: quit program
                      """
                )
                continue

            else:
                # Fetch context from Chroma and add it to the bot's primer
                context_chunks = chroma_memory.fetch_context(command)

                # Generate the answer using the bot
                spinner.start()
                response = bot.chat(command, context_chunks)
                answer = response['choices'][0]['message']['content']
                spinner.stop()
                print(f"\n\nAnswer: {answer}\n\n")

        if not exit_program:
            print(f"\n\nI hope you learned something about {title}!\n\n")

    print("\nGoodbye!")
    
if __name__ == "__main__":
    main()
