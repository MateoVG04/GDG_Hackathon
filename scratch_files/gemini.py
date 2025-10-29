import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
import tiktoken

class Gemini:
    def __init__(self):
        load_dotenv()
        api_key = "AIzaSyDyCQ4Dld0lCkG-nT6tk9Bz3XdOXQhxFPE"
        self.token_size = 900_000
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            exit()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.encoder = tiktoken.get_encoding("cl100k_base")


    def prompt(self, message):
        return self.model.generate_content(message)

    def is_good_chunk(self,chunk):
        """
        checks if the chunk is not to big
        :return: True or False
        """
        text = json.dumps(chunk)
        tokens = self.encoder.encode(text)
        return (len(tokens) <= self.token_size), len(tokens)

    def calc_token_count(self,chunk):
        text = json.dumps(chunk)
        return len(self.encoder.encode(text))

