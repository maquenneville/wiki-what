# WikiWhat
a Q&amp;A tool for Wikipedia-based, Pinecone-powered, context-informed GPT chat

This script creates a chatbot that answers questions based on a specific Wikipedia page and its related pages. The chatbot uses context from the specified Wikipedia page and related pages to provide more accurate and detailed answers.

# How to use

0. Get your OpenAI API key if you don't have one already: https://platform.openai.com/overview
Then, set up your Pinecone account, create an Index with this layout:
- Dimensions: 1536
- Metric: cosine
- pod type: p1
Get your API key, Index environment and index name.

Once you have your OpenAI key and your Pinecone env/key/name, enter then into their respective places in the config.ini file.  Then, open Command Prompt and cd into the cloned directory.

1. Make sure you have installed the required dependencies:
pip install -r requirements.txt

2. Run the script:
python wikipediachat.py

3. Follow the prompts to input a Wikipedia page title. The chatbot will gather data, calculate embeddings, and store them in the Pinecone index if needed.

4. Enter questions related to the Wikipedia page title you provided. You can also use additional commands such as 'switch topic', 'smart answer', 'simple answer', or 'help' for more options. To exit the chatbot, type "exit".

# Example Usage

>Please enter a Wikipedia page title: Albert Einstein
>Gathering the background data for this chat, calculating it's embeddings and loading them into Pinecone. For more narrow focus wikipedia pages, this could take a couple minutes. For wider focus, it could take a while.
>Ok, I'm ready for your questions about Albert Einstein.

>Enter a question or a command (enter 'help' for additional commands): What is the theory of relativity?
>Answer: The theory of relativity is a scientific theory developed by Albert Einstein that fundamentally changed our understanding of space, time, and gravity. It >consists of two parts: the special theory of relativity and the general theory of relativity. The special theory of relativity, published in 1905, deals with objects >moving at constant speeds, particularly in the absence of gravity. The general theory of relativity, published in 1915, is a theory of gravitation that describes how >massive objects cause curvature in spacetime, which leads to the observed motion of other objects.

>Press enter to ask another question, or enter "exit" to exit: exit
>I hope you learned something about Albert Einstein! Goodbye!

# Notes

- As the script relies on first creating a table of related sections, the scope of the initial wiki page will vary the data gathering/processing time greatly.  For example, if you use "Sheaf toss", a very niche folk athletic event, it will take a few seconds.  If you use "Fishing", it could take over an hour.

- The embedding process has been optimized for paid OpenAI accounts.  If you are using a free account it will take significantly more time for the context to embed.

- This chat agent tracks the pages that have been loaded into your index with the wiki_page_record.txt file.  This is to help prevent loading multiple copies of the same page.  As you use the program more and more, it will fill out your index with wikipedia context, and should slowly decrease the loading time of the wikipedia context.

- Most recent update added an option for "smart answer".  This agent is powered by GPT-4 and primed to give more nuanced, detailed and informative answers.  This is best used for complex/difficult questions, as it will be less likely to give up and say "I don't know", and also is much more expensive.


