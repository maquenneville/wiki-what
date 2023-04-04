# WikiWhat
a Q&amp;A tool for Wikipedia context-informed GPT chat

This script creates a chatbot that answers questions based on a specific Wikipedia page and its related pages. The chatbot uses context from the specified Wikipedia page and related pages to provide more accurate and detailed answers.

# How to use
1. Make sure you have installed the required dependencies:

pip install pandas wikipedia openai bs4 lxml nltk

2. Run the script:

python wikipediachat.py

3. Follow the prompts to input a Wikipedia page title, and choose whether you want to save or load chat data.

4. Ask questions related to the Wikipedia page title you provided. To exit the chatbot, type "exit".

# Example Usage

Please enter a Wikipedia page title: Albert Einstein
Do you want the chat data to be saved? (yes/no): no
Ok, I'm ready for your questions about Albert Einstein.

Question: What is the theory of relativity?
Answer: The theory of relativity is a scientific theory developed by Albert Einstein that fundamentally changed our understanding of space, time, and gravity. It consists of two parts: the special theory of relativity and the general theory of relativity. The special theory of relativity, published in 1905, deals with objects moving at constant speeds, particularly in the absence of gravity. The general theory of relativity, published in 1915, is a theory of gravitation that describes how massive objects cause curvature in spacetime, which leads to the observed motion of other objects.

Press enter to ask another question, or enter "exit" to exit: exit
I hope you learned something about Albert Einstein! Goodbye!

# Notes

-Running into some occasional page errors, currently working on that
