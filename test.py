### Configuration
###############################################################################################
# Imports
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import nlpcloud
import numpy as np
import re
import requests
import tqdm
import transformers


# Load environment variables from .env file
load_dotenv()

# Script Configuration



### Define Functions and Classes
###############################################################################################

def chunk_text_by_character_limit(text: str, char_limit: int = 640) -> list:
    """
    Splits text into chunks without exceeding the character limit, ensuring words are not split.
    Always keeps full sentences together unless a sentence itself exceeds the limit.

    Parameters
    ----------
    text : str
        The input text to be chunked.
    char_limit : int, optional
        The maximum character length per chunk (default: 640).

    Returns
    -------
    list
        A list of text chunks that do not exceed the character limit.
    """
    if len(text) <= char_limit:
        return [text]  # Return as a single chunk if within the limit

    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split on sentence boundaries
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) > char_limit:
            # If a single sentence is too long, split it at a word boundary
            start = 0
            while start < len(sentence):
                end = min(start + char_limit, len(sentence))

                # Try to break at a space before the limit
                if end < len(sentence) and sentence[end].isalnum():
                    end = sentence.rfind(" ", start, end)
                    if end == -1:  # If no space found, hard cut
                        end = min(start + char_limit, len(sentence))

                forced_chunk = sentence[start:end].strip()
                if forced_chunk:
                    chunks.append(forced_chunk)
                start = end  # Move to the next part of the sentence
        elif len(current_chunk) + len(sentence) + 1 <= char_limit:  # +1 for space
            current_chunk += " " + sentence if current_chunk else sentence  # Append sentence
        else:
            chunks.append(current_chunk)  # Store completed chunk
            current_chunk = sentence  # Start a new chunk

    if current_chunk:
        chunks.append(current_chunk)  # Store final chunk

    return chunks



def batch_list(input_list, batch_size):
    """
    Splits a list into smaller batches of size `batch_size`.

    Parameters
    ----------
    input_list : list
        The list to be batched.
    batch_size : int
        The max number of items per batch.

    Returns
    -------
    list of lists
        The list split into batches.
    """
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.

    Parameters
    ----------
    vec1 : list or np.array
        The first embedding vector.
    vec2 : list or np.array
        The second embedding vector.

    Returns
    -------
    float
        Cosine similarity score between -1 and 1.
    """
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_wiki_article(url: str) -> str:
    """
    Fetch and extract the main content of a Wikipedia article.

    Parameters
    ----------
    url : str
        The Wikipedia article URL.

    Returns
    -------
    str
        The extracted article text.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch page: {response.status_code}")

    soup = BeautifulSoup(response.text, 'lxml')
    content_div = soup.find('div', {'id': 'bodyContent'})

    if not content_div:
        raise Exception("Could not find main content on the page.")
    paragraphs = content_div.find_all('p')
    article_text = '\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    return article_text





def find_relevant_sections_with_scores(similarities, std_factor=1.0, context_factor=0.5, max_chunks_in_section=None):
    """
    Identify relevant sections based on similarity scores and expand context.
    Optionally limits each section to a maximum number of chunks around the peak similarity score.

    Parameters
    ----------
    similarities : list of floats
        The similarity scores for document chunks.
    std_factor : float, optional
        The number of standard deviations above the mean to set the high-relevance threshold (default: 1.0).
    context_factor : float, optional
        The number of standard deviations above the mean for including context chunks (default: 0.5).
    max_chunks_in_section : int, optional
        The maximum number of chunks to retain in each section, centered around the peak similarity (default: None, meaning no limit).

    Returns
    -------
    list of dict
        A list of dictionaries where each dictionary represents a relevant section:
        {
            "indices": list of consecutive indices forming a section,
            "scores": list of similarity scores for the section,
            "mean_score": float (mean similarity score for the section),
            "max_score": float (max similarity score for the section),
            "length": int (number of chunks in the section)
        }
    """
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)

    # Thresholds for relevance & context
    high_relevance_threshold = mean_sim + (std_factor * std_sim)
    context_threshold = mean_sim + (context_factor * std_sim)

    relevant_sections = []
    current_section = []
    current_scores = []

    for i, score in enumerate(similarities):
        if score >= context_threshold:
            current_section.append(i)
            current_scores.append(score)
        else:
            if current_section:  # Store completed section
                relevant_sections.append({
                    "indices": current_section,
                    "scores": current_scores,
                    "mean_score": float(np.mean(current_scores)),  # Convert to Python float
                    "max_score": float(np.max(current_scores)),  # Convert to Python float
                    "length": len(current_section)  # Number of chunks in the section
                })
                current_section = []
                current_scores = []

    if current_section:  # Add last section if it exists
        relevant_sections.append({
            "indices": current_section,
            "scores": current_scores,
            "mean_score": float(np.mean(current_scores)),  # Convert to Python float
            "max_score": float(np.max(current_scores)),  # Convert to Python float
            "length": len(current_section)  # Number of chunks in the section
        })

    # Apply `max_chunks_in_section` limitation
    if max_chunks_in_section:
        for section in relevant_sections:
            if section["length"] > max_chunks_in_section:
                # Find the peak (highest similarity score) in the section
                peak_idx = np.argmax(section["scores"])
                peak_position = section["indices"][peak_idx]

                # Compute start & end for centered window around the peak
                half_window = max_chunks_in_section // 2
                start_idx = max(0, peak_idx - half_window)
                end_idx = min(len(section["indices"]), start_idx + max_chunks_in_section)

                # Update section indices and scores
                section["indices"] = section["indices"][start_idx:end_idx]
                section["scores"] = section["scores"][start_idx:end_idx]
                section["mean_score"] = float(np.mean(section["scores"]))
                section["max_score"] = float(np.max(section["scores"]))
                section["length"] = len(section["indices"])  # Update length

    return relevant_sections






def prioritize_sections(sections, min_sections=2, max_sections=5, relevance_drop=0.85):
    """
    Dynamically prioritize relevant sections based on similarity and length.

    Parameters
    ----------
    sections : list of dict
        A list of section dictionaries, each containing "indices", "scores", "mean_score", "max_score", and "length".
    min_sections : int, optional
        The minimum number of sections to return (default: 2).
    max_sections : int, optional
        The maximum number of sections to return (default: 5).
    relevance_drop : float, optional
        The percentage of the top max_score below which sections will be considered (default: 0.85, meaning 85%).

    Returns
    -------
    list of dict
        The prioritized list of sections.
    """
    if not sections:
        return []  # No sections to return

    # Sort by: max_score (desc), mean_score (desc), length (desc)
    sorted_sections = sorted(
        sections,
        key=lambda sec: (sec["max_score"], sec["mean_score"], sec["length"]),
        reverse=True
    )

    # Get the highest max similarity score
    top_max_score = sorted_sections[0]["max_score"]

    # Dynamically filter sections where max_score is at least `relevance_drop` * top_max_score
    dynamic_threshold = top_max_score * relevance_drop
    filtered_sections = [sec for sec in sorted_sections if sec["max_score"] >= dynamic_threshold]

    # Ensure we always return at least `min_sections` and at most `max_sections`
    final_sections = filtered_sections[:max_sections] if len(filtered_sections) > max_sections else filtered_sections
    final_sections = final_sections if len(final_sections) >= min_sections else sorted_sections[:min_sections]

    return final_sections



def make_rag_question_answer_prompt(query: str, supporting_text: str, max_tokens: int = 8192):
    """
    Constructs a RAG-style prompt for an LLM while ensuring the total character count stays within limits.
    Approximates token count without using a tokenizer (1 token ≈ 4 characters).

    Parameters
    ----------
    query : str
        The question to be answered.
    supporting_text : str
        The supporting information from which the answer should be derived.
    max_tokens : int, optional
        The maximum number of tokens the LLM can handle (default: 8192).

    Returns
    -------
    str
        A formatted prompt that fits within the LLM token limit.
    """

    # Approximate max characters based on token limit (1 token ≈ 4 characters)
    max_chars = max_tokens * 4  

    # Define static parts of the prompt
    prompt_structure = f"""
    You are an AI assistant answering questions based strictly on the provided information. 
    Do NOT use outside knowledge. Answer concisely and factually.

    ### Supporting Information:
    """.strip()

    question_structure = f"""

    ### Question:
    {query}

    ### Answer:
    """.strip()

    # Calculate reserved characters
    reserved_chars = len(prompt_structure) + len(question_structure)

    # Determine available space for supporting text
    available_chars = max_chars - reserved_chars

    # If supporting text fits, return full prompt
    if len(supporting_text) <= available_chars:
        return f"{prompt_structure}\n\n{supporting_text}\n\n{question_structure}"

    # If too long, truncate at a sentence boundary
    truncated_text = supporting_text[:available_chars].rsplit(" ", 1)[0]  # Cut at last full word

    return f"{prompt_structure}\n\n{truncated_text}...\n\n{question_structure}"


### Execution
###############################################################################################


class EmbeddingGenerator:
    def __init__(self,
                 api_token: str = os.environ['NLP_CLOUD_TOKEN'],
                 character_limit: int = 640,
                 model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.api_token = api_token
        self.character_limit = character_limit
        self.model_name = model_name
        self.client = nlpcloud.Client(self.model_name, self.api_token)



    def chunk_text(self, text: str):
        text_chunks = chunk_text_by_character_limit(text = text, char_limit = self.character_limit)
        self.text_chunks = text_chunks
        return text_chunks

    def embed_document_text(self, text: str, batch_size: int = 5):
        """Embed document text in chunks"""
        doc_text = self.chunk_text(text)
        doc_embed = []
        for batch in tqdm.tqdm(batch_list(doc_text, batch_size), desc = 'Embedding document chunks'):
            response = self.client.embeddings(batch)
            doc_embed.extend(response['embeddings'])
        return doc_embed
    
    def embed_query(self, text: str):
        """Embed query. Must be less than or equal to self.character_limit"""
        if len(text) > self.character_limit:
            raise ValueError(f"Query exceeds character limit of {self.character_limit} characters.")
        response = self.client.embeddings([text])
        return response['embeddings'][0]
        



# wiki_url = 'https://en.wikipedia.org/wiki/World_War_II'
#wiki_url = 'https://en.wikipedia.org/wiki/United_States_federal_government_targets_of_Elon_Musk'


# Get wikipedia text
wiki_url = 'https://en.wikipedia.org/wiki/Twitter_use_by_Donald_Trump'
wiki_text = get_wiki_article(wiki_url)


# Test LLM knowledge without context
test_query = """When Donald Trump retweets himself, what phrase does he often use?"""

client = nlpcloud.Client('llama-3-1-405b', os.environ['NLP_CLOUD_TOKEN'], gpu=True)
baseline_answer = client.question(test_query)['answer']
print(baseline_answer)

"""
When Donald Trump retweets himself, he often uses the phrase "Twitter" followed by a punctuation mark,
such as an exclamation point. For instance, he has been known to say "Twitter!" at the end of his retweets.
This practice has been a signature part of his social media style.
"""

# Use RAG
embedding_gen = EmbeddingGenerator()
wiki_embed = embedding_gen.embed_document_text(wiki_text)
query_embed = embedding_gen.embed_query(test_query)
similarities = [float(cosine_similarity(query_embed, w_emb)) for w_emb in wiki_embed]
relevant_sections = find_relevant_sections_with_scores(similarities, std_factor=1.0, context_factor = 0.5, max_chunks_in_section = 3)
prioritized_sections = prioritize_sections(relevant_sections, max_sections = 3)
final_text = " | ".join([" ".join([embedding_gen.text_chunks[i] for i in section["indices"]]) for section in prioritized_sections])
llm_prompt = make_rag_question_answer_prompt(query = test_query, supporting_text = final_text, max_tokens = 128000)
client = nlpcloud.Client('llama-3-1-405b', os.environ['NLP_CLOUD_TOKEN'], gpu=True)
rag_answer = client.question(llm_prompt)['answer']
print(rag_answer)

# Much better
"""
Donald Trump often comments "so true" when he retweets himself.
"""


