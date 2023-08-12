# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 01:31:03 2023

@author: marca
"""

from claude_api import Client
import configparser
import os

class ClaudeBot:

    def __init__(self, convo_id=None):
        
        self.cookie = self._get_cookie("config.ini")
        self.claude_api = Client(self.cookie)
        
        if not convo_id:
            self.conversation_id = self.claude_api.create_new_chat()['uuid']
            
        else:
            self.conversation_id = convo_id
            
        
        
    def _get_cookie(self, config_file):
        if not os.path.exists("config.ini"):
            raise FileNotFoundError("The config file was not found.")
        config = configparser.ConfigParser()
        config.read(config_file)

        cookie = config.get("API_KEYS", "Claude_Cookie")
        

        return cookie
    
    def get_chat_history(self):
        
        return self.claude_api.chat_conversation_history(self.conversation_id)

    def chat(self, prompt, context_chunks=None, attachment=None):
        
        if context_chunks:
            
            context_block = ""
            for chunk in context_chunks:
                context_block += f"\nContext: {chunk}\n"
            
            prompt = prompt + context_block
        
        if attachment:
            response = self.claude_api.send_message(prompt, self.conversation_id, attachment=attachment)
        
        else:
            response = self.claude_api.send_message(prompt, self.conversation_id)
            
        return response
