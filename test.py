### Configuration
###############################################################################################
# Imports
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import nlpcloud
import requests

# Load environment variables from .env file
load_dotenv()

# Script Configuration







### Define Functions and Classes
###############################################################################################










### Execution
###############################################################################################
client = nlpcloud.Client('paraphrase-multilingual-mpnet-base-v2', os.environ['NLP_CLOUD_TOKEN'])
client.embeddings(['John Does works for Google.', 
                   'Janette Doe works for Microsoft.', 
                   'Janie Does works for NLP Cloud.'])





# Wikipedia URL
url = 'https://en.wikipedia.org/wiki/World_War_II'

# Fetch the webpage
response = requests.get(url)

# Check if request was successful
if response.status_code == 200:
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, 'lxml')

    # Find the main content section (ignores tables, sidebars, etc.)
    content_div = soup.find('div', {'id': 'bodyContent'})

    # Extract all paragraphs
    paragraphs = content_div.find_all('p')

    # Combine and clean the text
    article_text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])


else:
    print(f"❌ Failed to fetch page. Status code: {response.status_code}")
