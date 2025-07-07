# web_search_tool.py
import os
from tavily import TavilyClient
from typing import List

class WebSearchTool:
    def __init__(self):
        """
        Initializes the WebSearchTool with Tavily API client.
        Ensures TAVILY_API_KEY is set in environment variables.
        """
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set.")
        self.client = TavilyClient(api_key=self.api_key)
        print("WebSearchTool initialized.")

    def search(self, query: str, max_results: int = 3) -> List[str]:
        """
        Performs a web search using Tavily API and returns a list of snippets.
        """
        try:
            # Using Tavily's search method which returns a dictionary
            # We are interested in the 'results' key which contains a list of dictionaries
            # Each result dictionary has 'content' and 'url'
            response = self.client.search(query=query, max_results=max_results, include_answer=False, include_raw_content=False)
            
            snippets = [result['content'] for result in response.get('results', []) if 'content' in result]
            
            return snippets
        except Exception as e:
            print(f"Error during Tavily web search for query '{query}': {e}")
            return []

# Example usage (for testing this file independently)
if __name__ == "__main__":
    # For testing, you might temporarily set the API key like this:
    # os.environ["TAVILY_API_KEY"] = "YOUR_TAVILY_API_KEY_HERE"
    
    # Ensure .env is loaded if running independently
    from dotenv import load_dotenv
    load_dotenv()

    try:
        search_tool = WebSearchTool()
        print("\n--- Searching for 'Eiffel Tower opening hours July 2025' ---")
        results = search_tool.search("Eiffel Tower opening hours July 2025")
        for i, r in enumerate(results):
            print(f"Snippet {i+1}: {r[:200]}...") # Print first 200 chars of snippet
        
        print("\n--- Searching for 'Best family restaurants Paris Le Marais' ---")
        results = search_tool.search("Best family restaurants Paris Le Marais")
        for i, r in enumerate(results):
            print(f"Snippet {i+1}: {r[:200]}...")

    except ValueError as e:
        print(f"Setup Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")

