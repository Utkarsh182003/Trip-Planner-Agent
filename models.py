# models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Pydantic model for user input
class UserInput(BaseModel):
    """
    Represents the initial input provided by the user for travel planning.
    """
    destination: str = Field(..., description="The desired travel destination (e.g., 'Paris, France').")
    start_date: str = Field(..., description="The start date of the trip (e.g., '2025-09-01').")
    end_date: str = Field(..., description="The end date of the trip (e.g., '2025-09-05').")
    travelers: int = Field(1, ge=1, description="Number of travelers.")
    interests: List[str] = Field(default_factory=list, description="List of user interests (e.g., 'history', 'food', 'nature').")
    budget: str = Field(..., description="Budget range (e.g., 'mid-range', 'luxury', 'budget-friendly').")
    preferences: Optional[str] = Field(None, description="Any specific preferences or requirements.")

# Pydantic model for a single itinerary item
class ItineraryItem(BaseModel):
    """
    Represents a single activity or event in the travel itinerary.
    """
    time: str = Field(..., description="Suggested time for the activity (e.g., '9:00 AM', 'Lunch time').")
    name: str = Field(..., description="Name of the activity or place (e.g., 'Eiffel Tower', 'Louvre Museum').")
    description: str = Field(..., description="Brief description of the activity.")
    location: Optional[str] = Field(None, description="Specific location or address.")
    category: Optional[str] = Field(None, description="Category of the activity (e.g., 'Attraction', 'Restaurant', 'Shopping').")
    estimated_duration: Optional[str] = Field(None, description="Estimated duration of the activity (e.g., '2-3 hours').")

# Pydantic model for a daily itinerary
class DailyItinerary(BaseModel):
    """
    Represents the plan for a single day of the trip.
    """
    date: str = Field(..., description="The date for this day's itinerary (e.g., 'YYYY-MM-DD').")
    activities: List[ItineraryItem] = Field(default_factory=list, description="List of activities for the day.")

# NEW: Pydantic model to wrap the list of daily itineraries
class ItineraryResponse(BaseModel):
    """
    Wrapper model for the LLM's detailed itinerary output, which is a list of DailyItinerary.
    """
    itinerary: List[DailyItinerary] = Field(..., description="A list of daily itinerary plans.")

# Pydantic model for the complete travel plan
class TravelPlan(BaseModel):
    """
    Represents the complete generated travel plan, including high-level outline and detailed itinerary.
    """
    high_level_outline: str = Field(..., description="A brief overview of the trip plan.")
    detailed_itinerary: List[DailyItinerary] = Field(default_factory=list, description="Detailed day-by-day itinerary.")

# Pydantic model for the overall agent state (will evolve in later phases)
class AgentState(BaseModel):
    """
    Represents the current state of the multi-agent system,
    including user input, generated plan, and any retrieved data.
    This will be the shared state between agents.
    """
    user_input: UserInput
    travel_plan: Optional[TravelPlan] = None
    retrieved_documents: List[str] = Field(default_factory=list, description="List of relevant document snippets from RAG.")
    knowledge_graph_data: Dict[str, Any] = Field(default_factory=dict, description="Structured data retrieved from the Knowledge Graph.")
    web_search_results: List[str] = Field(default_factory=list, description="List of relevant snippets from web search.") # NEW FIELD
    current_task_status: str = Field("Initialized", description="Current status of the planning process.")

