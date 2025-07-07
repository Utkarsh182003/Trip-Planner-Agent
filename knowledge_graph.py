# knowledge_graph.py
from typing import Dict, List, Any, Optional
import json

class KnowledgeGraph:
    def __init__(self):
        """
        Initializes a simple in-memory knowledge graph.
        Entities are nodes, and relationships are properties or connections.
        """
        self.graph: Dict[str, Dict[str, Any]] = self._load_static_graph()
        print("KnowledgeGraph initialized with static data.")

    def _load_static_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads a small, static knowledge graph for demonstration purposes.
        In a real application, this would be populated from structured data sources
        or extracted from text.
        """
        return {
            "Eiffel Tower": {
                "type": "landmark",
                "city": "Paris",
                "country": "France",
                "description": "Iconic wrought-iron lattice tower, global cultural icon.",
                "opening_hours": "Typically 9:00 AM - 12:45 AM daily",
                "tickets": "Recommended to book online in advance",
                "nearby_attractions": ["Champ de Mars", "Seine River"],
                "category": "Attraction"
            },
            "Louvre Museum": {
                "type": "museum",
                "city": "Paris",
                "country": "France",
                "description": "World's most-visited museum, home to Mona Lisa.",
                "opening_hours": "Typically 9:00 AM - 6:00 PM, closed Tuesdays",
                "tickets": "Highly recommended to book online",
                "nearby_attractions": ["Tuileries Garden", "Palais Royal"],
                "category": "Attraction"
            },
            "Notre-Dame Cathedral": {
                "type": "cathedral",
                "city": "Paris",
                "country": "France",
                "description": "Medieval Catholic cathedral, currently under reconstruction.",
                "opening_hours": "Exterior viewable",
                "nearby_attractions": ["Île de la Cité"],
                "category": "Attraction"
            },
            "Montmartre": {
                "type": "neighborhood",
                "city": "Paris",
                "country": "France",
                "description": "Historic artistic neighborhood with Sacré-Cœur Basilica and Place du Tertre.",
                "highlights": ["Sacré-Cœur Basilica", "Place du Tertre (artists)"],
                "category": "District"
            },
            "Palace of Versailles": {
                "type": "royal residence",
                "city": "Versailles",
                "country": "France",
                "description": "Opulent royal château with vast gardens, full-day trip from Paris.",
                "category": "Attraction"
            },
            "Paris": {
                "type": "city",
                "country": "France",
                "description": "Capital city of France, famous for art, fashion, gastronomy, and culture.",
                "known_for": ["Eiffel Tower", "Louvre Museum", "Cuisine", "Fashion"],
                "family_friendly_activities": ["Disneyland Paris", "Jardin du Luxembourg", "Boat tours on Seine"]
            },
            "Rome": {
                "type": "city",
                "country": "Italy",
                "description": "Capital city of Italy, known for ancient history, art, and food.",
                "known_for": ["Colosseum", "Vatican City", "Roman Forum", "Pasta", "Gelato"],
                "family_friendly_activities": ["Gladiator School", "Explora Children's Museum", "Villa Borghese"]
            },
            "Colosseum": {
                "type": "amphitheatre",
                "city": "Rome",
                "country": "Italy",
                "description": "Ancient Roman amphitheatre, largest ever built.",
                "tickets": "Book well in advance",
                "category": "Attraction"
            },
            "Vatican City": {
                "type": "city-state",
                "city": "Rome",
                "country": "Vatican City", # Technically its own state, but associated with Rome
                "description": "Smallest independent state, home to St. Peter's Basilica, Vatican Museums, Sistine Chapel.",
                "dress_code": "Shoulders and knees covered",
                "category": "Attraction"
            },
            "Italian Cuisine": {
                "type": "cuisine",
                "description": "Famous for pasta (Cacio e Pepe, Carbonara, Amatriciana), pizza, and gelato.",
                "dining_options": ["Trattorias", "Osterias", "Pizzerias"],
                "category": "Food"
            },
            "French Cuisine": {
                "type": "cuisine",
                "description": "Known for Coq au Vin, Beef Bourguignon, croissants, pastries.",
                "dining_options": ["Bistros", "Cafes", "Brasseries"],
                "category": "Food"
            }
        }

    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves all information about a specific entity.
        """
        return self.graph.get(entity_name)

    def find_entities_by_attribute(self, attribute: str, value: Any) -> List[str]:
        """
        Finds entities that have a specific attribute with a given value.
        Example: find_entities_by_attribute("city", "Paris")
        """
        found_entities = []
        for entity_name, data in self.graph.items():
            if attribute in data and data[attribute] == value:
                found_entities.append(entity_name)
        return found_entities

    def find_attractions_by_city_and_interest(self, city: str, interests: List[str]) -> List[Dict[str, Any]]:
        """
        Finds attractions in a given city that match user interests.
        Returns a list of dictionaries with attraction details.
        """
        matching_attractions = []
        for entity_name, data in self.graph.items():
            if data.get("type") in ["landmark", "museum", "cathedral", "royal residence", "amphitheatre", "city-state"] and data.get("city") == city:
                # Simple keyword matching for interests for now
                if any(interest.lower() in data.get("description", "").lower() or
                       interest.lower() in data.get("category", "").lower() or
                       (data.get("highlights") and any(interest.lower() in h.lower() for h in data["highlights"]))
                       for interest in interests):
                    matching_attractions.append({"name": entity_name, **data})
                elif not interests: # If no specific interests, include all attractions in the city
                    matching_attractions.append({"name": entity_name, **data})
        return matching_attractions

    def get_cuisine_info(self, cuisine_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves information about a specific cuisine type.
        """
        return self.graph.get(cuisine_type)

# Example usage (for testing this file independently)
if __name__ == "__main__":
    kg = KnowledgeGraph()

    print("\n--- Info about Eiffel Tower ---")
    eiffel_info = kg.get_entity_info("Eiffel Tower")
    if eiffel_info:
        print(json.dumps(eiffel_info, indent=2))

    print("\n--- Entities in Paris ---")
    paris_entities = kg.find_entities_by_attribute("city", "Paris")
    print(paris_entities)

    print("\n--- Attractions in Paris interested in History ---")
    history_attractions = kg.find_attractions_by_city_and_interest("Paris", ["History"])
    for attr in history_attractions:
        print(f"- {attr['name']}: {attr.get('description', '')}")

    print("\n--- Cuisine Info for French Cuisine ---")
    french_cuisine = kg.get_cuisine_info("French Cuisine")
    if french_cuisine:
        print(json.dumps(french_cuisine, indent=2))
