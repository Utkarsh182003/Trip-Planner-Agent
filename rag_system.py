# rag_system.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict

class RAGSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the RAG system with a Sentence Transformer model.
        Loads a predefined knowledge base for demonstration.
        """
        self.model = SentenceTransformer(model_name)
        self.knowledge_base = self._load_knowledge_base()
        self.documents = [item['text'] for item in self.knowledge_base]
        self.document_embeddings = self.model.encode(self.documents, convert_to_tensor=False)
        print("RAGSystem initialized with knowledge base and embeddings.")

    def _load_knowledge_base(self) -> List[Dict]:
        """
        Loads a small, predefined knowledge base. In a real application, this
        would come from a database, files, or external APIs.
        """
        # This is a small, hardcoded knowledge base for demonstration.
        # In a real app, this would be much larger and dynamic.
        return [
            {"id": "paris_eiffel", "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is a global cultural icon of France and one of the most recognisable structures in the world. It is open daily, usually from 9:00 AM to 12:45 AM. Tickets can be purchased online in advance."},
            {"id": "paris_louvre", "text": "The Louvre Museum in Paris, France, is the world's most-visited museum and a historic monument. It is home to thousands of works of art, including the Mona Lisa. It is typically closed on Tuesdays. Opening hours are usually 9:00 AM to 6:00 PM. Booking tickets online is highly recommended."},
            {"id": "paris_notre_dame", "text": "Notre-Dame de Paris, also known as Notre-Dame Cathedral, is a medieval Catholic cathedral on the Île de la Cité in the 4th arrondissement of Paris. While it is undergoing reconstruction after the 2019 fire, its exterior can still be admired, and the surrounding area is historically significant."},
            {"id": "paris_food_bistro", "text": "Paris is famous for its bistros and cafes. Traditional French cuisine includes dishes like Coq au Vin, Beef Bourguignon, and croissants. Many family-friendly restaurants offer simpler menus and welcoming atmospheres. Look for establishments in areas like Le Marais or Saint-Germain-des-Prés."},
            {"id": "paris_montmartre", "text": "Montmartre is a large hill in Paris's 18th arrondissement. It's famous for the Basilica of the Sacré-Cœur and its artistic history, with many street artists in Place du Tertre. It offers panoramic views of the city and has a bohemian vibe. It's a great place for a leisurely stroll."},
            {"id": "paris_versailles", "text": "The Palace of Versailles is a royal château in Versailles, about 20 kilometers (12 miles) southwest of the center of Paris. It was the principal royal residence of France from 1682 to 1789. It features opulent interiors, vast gardens, and the famous Hall of Mirrors. It's a full-day trip from Paris."},
            {"id": "london_buckingham", "text": "Buckingham Palace is the London residence and administrative headquarters of the monarch of the United Kingdom. It is a major tourist attraction and is open to visitors during certain times of the year, usually in summer. The Changing of the Guard ceremony is a popular event."},
            {"id": "london_british_museum", "text": "The British Museum in London is a public institution dedicated to human history, art and culture. Its permanent collection, numbering some 8 million works, is among the largest and most comprehensive in existence. Admission is free, but special exhibitions may require tickets."},
            {"id": "rome_colosseum", "text": "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy. Built of travertine limestone, tuff, and brick-faced concrete, it was the largest amphitheatre ever built. It is a must-visit historical site. Booking tickets well in advance is essential due to high demand."},
            {"id": "rome_vatican", "text": "Vatican City, the smallest independent state in the world, is home to St. Peter's Basilica, the Vatican Museums, and the Sistine Chapel. It is a major religious and cultural site. Dress code applies (shoulders and knees covered). Expect crowds and long lines."},
            {"id": "rome_food", "text": "Roman cuisine is known for pasta dishes like Cacio e Pepe, Carbonara, and Amatriciana, as well as pizza al taglio and gelato. Trattorias and osterias offer authentic local experiences. Food tours are popular for exploring culinary delights."}
        ]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves the most relevant documents from the knowledge base based on the query.
        """
        if not self.documents:
            return []

        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # Calculate cosine similarity between query and all document embeddings
        # Reshape for sklearn's cosine_similarity: (n_samples, n_features)
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.document_embeddings)
        
        # Get the indices of the top_k most similar documents
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        # Return the text of the top_k documents
        retrieved_texts = [self.documents[i] for i in top_k_indices]
        return retrieved_texts

# Example usage (for testing this file independently)
if __name__ == "__main__":
    rag_system = RAGSystem()
    
    print("\n--- Retrieving for 'Paris history' ---")
    results = rag_system.retrieve("historical sites in Paris")
    for r in results:
        print(f"- {r[:100]}...") # Print first 100 chars
    
    print("\n--- Retrieving for 'London museums' ---")
    results = rag_system.retrieve("best museums in London")
    for r in results:
        print(f"- {r[:100]}...")

    print("\n--- Retrieving for 'Rome food' ---")
    results = rag_system.retrieve("authentic Italian food in Rome")
    for r in results:
        print(f"- {r[:100]}...")


# This code defines a simple RAG system that uses a Sentence Transformer model to encode documents
# and queries, retrieves the most relevant documents based on cosine similarity, and prints the results.