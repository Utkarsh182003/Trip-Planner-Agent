# main.py
import streamlit as st
from models import UserInput, AgentState, TravelPlan, DailyItinerary, ItineraryItem, ItineraryResponse
import json
import os
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime, timedelta
from rag_system import RAGSystem
from knowledge_graph import KnowledgeGraph
from web_search_tool import WebSearchTool
from typing import List, Dict, Any, Optional, Callable # Import Callable for tool functions

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please set it up.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize RAG System globally
@st.cache_resource
def load_rag_system():
    return RAGSystem()
rag_system = load_rag_system()

# Initialize Knowledge Graph globally
@st.cache_resource
def load_knowledge_graph():
    return KnowledgeGraph()
kg_system = load_knowledge_graph()

# Initialize Web Search Tool globally
@st.cache_resource
def load_web_search_tool():
    try:
        return WebSearchTool()
    except ValueError as e:
        st.error(f"Web Search Tool Error: {e}. Please ensure TAVILY_API_KEY is set in your .env file.")
        st.stop()
web_search_tool = load_web_search_tool()

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Travel Planner")


# --- Tool Definitions for LLM ---
# These functions will be exposed to the LLM
def rag_retrieve(query: str) -> List[str]:
    """
    Retrieves relevant document snippets from a local knowledge base (RAG).
    Useful for general factual information or common travel queries.
    Args:
        query (str): The search query for the RAG system.
    Returns:
        List[str]: A list of relevant document snippets.
    """
    return rag_system.retrieve(query)

def kg_get_entity_info(entity_name: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves structured information about a specific entity (e.g., attraction, city)
    from the Knowledge Graph.
    Args:
        entity_name (str): The name of the entity to retrieve information for (e.g., "Eiffel Tower", "Paris").
    Returns:
        Optional[Dict[str, Any]]: A dictionary of attributes for the entity, or None if not found.
    """
    return kg_system.get_entity_info(entity_name)

def kg_find_attractions(city: str, interests: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Finds attractions in a given city that match specified interests from the Knowledge Graph.
    Args:
        city (str): The city to search for attractions (e.g., "Paris").
        interests (Optional[List[str]]): A list of user interests (e.g., ["History", "Art"]).
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing attraction details.
    """
    if interests is None:
        interests = []
    return kg_system.find_attractions_by_city_and_interest(city, interests)

def web_search(query: str) -> List[str]:
    """
    Performs a real-time web search to get up-to-date information.
    Useful for current events, latest reviews, specific opening hours, or dynamic data.
    Args:
        query (str): The search query for the web.
    Returns:
        List[str]: A list of relevant content snippets from the web.
    """
    return web_search_tool.search(query)

# Map tool names to their functions
available_tools: Dict[str, Callable[..., Any]] = {
    "rag_retrieve": rag_retrieve,
    "kg_get_entity_info": kg_get_entity_info,
    "kg_find_attractions": kg_find_attractions,
    "web_search": web_search,
}

# Define Groq tool specifications from our Python functions
tool_specs = [
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve",
            "description": "Retrieves relevant document snippets from a local knowledge base (RAG). Useful for general factual information or common travel queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for the RAG system."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_get_entity_info",
            "description": "Retrieves structured information about a specific entity (e.g., attraction, city) from the Knowledge Graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "The name of the entity to retrieve information for (e.g., 'Eiffel Tower', 'Paris')."}
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_find_attractions",
            "description": "Finds attractions in a given city that match specified interests from the Knowledge Graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city to search for attractions (e.g., 'Paris')."},
                    "interests": {"type": "array", "items": {"type": "string"}, "description": "A list of user interests (e.g., ['History', 'Art'])."}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a real-time web search to get up-to-date information. Useful for current events, latest reviews, specific opening hours, or dynamic data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for the web."}
                },
                "required": ["query"],
            },
        },
    },
]


def run_llm_with_tools(messages: List[Dict[str, Any]], max_tool_iterations: int = 5) -> str:
    """
    Runs the LLM, handling tool calls iteratively.
    """
    tool_outputs = []
    for iteration in range(max_tool_iterations):
        st.info(f"LLM Tool Call Iteration: {iteration + 1}/{max_tool_iterations}")
        chat_completion = client.chat.completions.create(
            messages=messages + tool_outputs, # Include previous tool outputs
            model="llama3-8b-8192",
            tools=tool_specs, # Pass the tool specifications
            tool_choice="auto", # Let the LLM decide if it wants to use a tool
            temperature=0.7,
            max_tokens=4000,
        )

        response_message = chat_completion.choices[0].message
        
        # Check if the LLM wants to call a tool
        if response_message.tool_calls:
            st.info(f"LLM requested tool calls.")
            current_tool_outputs = []
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    st.error(f"LLM provided malformed JSON arguments for tool {function_name}: {tool_call.function.arguments}")
                    current_tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": "Error: Malformed JSON arguments.",
                    })
                    continue

                if function_name in available_tools:
                    st.info(f"Executing tool: `{function_name}` with args: `{function_args}`")
                    try:
                        tool_response = available_tools[function_name](**function_args)
                        current_tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": json.dumps(tool_response) if isinstance(tool_response, (dict, list)) else str(tool_response),
                        })
                        st.write(f"Tool `{function_name}` output: {current_tool_outputs[-1]['content'][:200]}...") # Show snippet of tool output
                    except Exception as tool_exec_e:
                        st.error(f"Error executing tool `{function_name}`: {tool_exec_e}")
                        current_tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"Error executing tool: {tool_exec_e}",
                        })
                else:
                    st.warning(f"LLM requested unknown tool: {function_name}")
                    current_tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": "Error: Unknown tool.",
                    })
            
            messages.append(response_message) # Add the message that requested the tool call
            tool_outputs.extend(current_tool_outputs) # Add the results of the tool calls
        else:
            # If no tool call, this is the final response
            return response_message.content
    
    st.warning("Max tool iterations reached. Could not get a final itinerary from LLM.")
    return "Could not generate a detailed itinerary after multiple tool calls."


def get_high_level_plan_from_llm(user_input: UserInput) -> str:
    """
    Generates a high-level travel plan using the Groq LLM based on user input.
    """
    prompt = f"""
    You are an expert travel planner. Your task is to create a concise, high-level travel outline
    for a trip based on the user's preferences. Do NOT include daily breakdowns or specific activities yet.
    Focus on the overall theme, key highlights, and flow of the trip.

    User Request:
    Destination: {user_input.destination}
    Start Date: {user_input.start_date}
    End Date: {user_input.end_date}
    Number of Travelers: {user_input.travelers}
    Interests: {', '.join(user_input.interests)}
    Budget: {user_input.budget}
    Preferences: {user_input.preferences if user_input.preferences else 'None'}

    Provide only the high-level outline, without any conversational filler or extra text.
    Example: "A 5-day historical and culinary adventure in Rome, exploring ancient ruins,
    Vatican City, and indulging in authentic Italian cuisine, with a focus on relaxed evenings."
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=200,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating high-level plan: {e}")
        return "Could not generate a high-level plan at this time."


def get_detailed_itinerary_with_tools(user_input: UserInput, high_level_outline: str,
                                       existing_itinerary: Optional[List[DailyItinerary]] = None,
                                       refinement_request: Optional[str] = None) -> List[DailyItinerary]:
    """
    Generates a detailed, day-by-day itinerary using the Groq LLM with tool calling.
    """
    start = datetime.strptime(user_input.start_date, "%Y-%m-%d")
    end = datetime.strptime(user_input.end_date, "%Y-%m-%d")
    num_days = (end - start).days + 1

    existing_itinerary_str = ""
    if existing_itinerary:
        existing_itinerary_str = "\n--- Current Itinerary to Refine (if applicable) ---"
        for day_plan in existing_itinerary:
            existing_itinerary_str += f"\nDate: {day_plan.date}"
            for activity in day_plan.activities:
                existing_itinerary_str += f"\n  - Time: {activity.time}, Name: {activity.name}, Description: {activity.description}, Location: {activity.location}, Category: {activity.category}, Duration: {activity.estimated_duration}"
        existing_itinerary_str += "\n---------------------------------------------------"

    refinement_str = ""
    if refinement_request:
        refinement_str = f"\n--- User Refinement Request ---\n{refinement_request}\n-------------------------------"

    # Initial system message to set the context for the LLM
    system_message = {
        "role": "system",
        "content": f"""
        You are an expert travel planner. Your task is to create a detailed, day-by-day travel itinerary.
        The itinerary MUST cover exactly {num_days} days, starting from {user_input.start_date} to {user_input.end_date}.
        For each day, include 3-5 distinct activities. Ensure each activity has a 'time', 'name', 'description', 'location', 'category', and 'estimated_duration'.
        Prioritize activities based on user interests, budget, and preferences.
        Ensure activities are logically grouped by location or theme for efficiency.

        You have access to tools to gather information: `rag_retrieve`, `kg_get_entity_info`, `kg_find_attractions`, and `web_search`.
        Use these tools whenever you need specific, up-to-date, or factual information to create a comprehensive and accurate itinerary.
        For example, if you need opening hours, search for current events, or find specific details about an attraction, use the appropriate tool.

        IMPORTANT: The final output MUST be a JSON object with a single key 'itinerary', whose value is an array of daily plans.
        Each daily plan in the array should conform to the structure of a DailyItinerary object.
        The 'itinerary' array MUST ALWAYS be sorted chronologically by the 'date' field.
        If a refinement request asks to 'swap' days or reorder activities, you must ensure the final output
        still presents the days in strict chronological order (e.g., Day 1: 2025-07-05, Day 2: 2025-07-06, etc.),
        even if it means moving the content of a specific date. Do NOT output dates out of sequence.

        Do NOT include any conversational text, markdown formatting (like ```json), or any other text outside the JSON object in your final response.
        Output ONLY the JSON object.

        Example of expected JSON structure for final output:
        {{
          "itinerary": [
            {{
              "date": "YYYY-MM-DD",
              "activities": [
                {{
                  "time": "9:00 AM",
                  "name": "Activity Name",
                  "description": "Description of activity.",
                  "location": "Location details",
                  "category": "Attraction",
                  "estimated_duration": "2 hours"
                }}
              ]
            }}
          ]
        }}
        """
    }

    # Initial user message
    user_message = {
        "role": "user",
        "content": f"""
        Please generate a detailed travel itinerary based on the following:

        User Request:
        Destination: {user_input.destination}
        Start Date: {user_input.start_date}
        End Date: {user_input.end_date}
        Number of Travelers: {user_input.travelers}
        Interests: {', '.join(user_input.interests)}
        Budget: {user_input.budget}
        Preferences: {user_input.preferences if user_input.preferences else 'None'}

        High-Level Outline:
        {high_level_outline}
        {existing_itinerary_str}
        {refinement_str}
        """
    }

    messages = [system_message, user_message]

    try:
        json_output = run_llm_with_tools(messages) # Call our new tool-running function
        
        parsed_response = ItineraryResponse.model_validate_json(json_output)
        parsed_response.itinerary.sort(key=lambda x: datetime.strptime(x.date, "%Y-%m-%d"))
        
        return parsed_response.itinerary

    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON from LLM. This often means the LLM did not return valid JSON. Error: {e}. Raw LLM output: {json_output}")
        # Print raw LLM output to console for debugging
        print(f"DEBUG: Raw LLM output that caused JSONDecodeError: {json_output}")
        return []
    except Exception as e:
        st.error(f"Error generating detailed plan with tools or Pydantic validation: {e}")
        return []


def app():
    st.title("✈️ AI-Powered Smart Travel Planner")
    st.markdown("Enter your travel details below to get a personalized itinerary!")

    # --- User Input Form ---
    st.header("Your Travel Request")
    with st.form("travel_input_form"):
        destination = st.text_input("Destination (e.g., Paris, France)", "Paris, France")
        col1, col2 = st.columns(2)
        
        today = datetime.now().date()
        start_date = col1.date_input("Start Date", value=today)
        end_date = col2.date_input("End Date", value=today + timedelta(days=2))

        travelers = st.number_input("Number of Travelers", min_value=1, value=1)
        
        start_date_str = start_date.strftime("%Y-%m-%d") if start_date else ""
        end_date_str = end_date.strftime("%Y-%m-%d") if end_date else ""

        interests_options = ["History", "Food", "Nature", "Art", "Shopping", "Adventure", "Relaxation", "Nightlife"]
        interests = st.multiselect("Interests", options=interests_options, default=["History", "Food"])
        
        budget_options = ["Budget-friendly", "Mid-range", "Luxury"]
        budget = st.selectbox("Budget", options=budget_options, index=1)
        
        preferences = st.text_area("Any specific preferences or requirements?", "Family-friendly activities, good local restaurants.")

        submitted = st.form_submit_button("Plan My Trip")
        
    st.markdown("---")
    if st.button("Start New Plan / Clear All"):
        st.session_state.agent_state = None
        st.rerun()

    if 'agent_state' not in st.session_state:
        st.session_state.agent_state = None

    if submitted:
        if not start_date or not end_date:
            st.error("Please select both a start date and an end date.")
            st.session_state.agent_state = None 
            return

        if start_date > end_date:
            st.error("End date cannot be before start date. Please adjust your dates.")
            st.session_state.agent_state = None
            return

        try:
            user_input = UserInput(
                destination=destination,
                start_date=start_date_str,
                end_date=end_date_str,
                travelers=travelers,
                interests=interests,
                budget=budget,
                preferences=preferences if preferences else None
            )

            st.session_state.agent_state = AgentState(user_input=user_input)

            st.success("Travel request received and validated!")

            status_placeholder = st.empty()

            with status_placeholder.container():
                st.info("Generating high-level plan...")
                high_level_outline = get_high_level_plan_from_llm(user_input)
                st.session_state.agent_state.travel_plan = TravelPlan(
                    high_level_outline=high_level_outline,
                    detailed_itinerary=[]
                )
                st.session_state.agent_state.current_task_status = "High-level plan generated"
                st.subheader("Generated High-Level Travel Plan:")
                st.write(st.session_state.agent_state.travel_plan.high_level_outline) 

                st.info("Initiating detailed itinerary generation with dynamic tool use...")

                # --- Phase 9: Generate Detailed Itinerary using LLM with Tool Calling ---
                detailed_itinerary = get_detailed_itinerary_with_tools(
                    user_input, 
                    high_level_outline, 
                    existing_itinerary=None, # No existing itinerary for initial generation
                    refinement_request=None
                )
                
                if detailed_itinerary:
                    st.session_state.agent_state.travel_plan.detailed_itinerary = detailed_itinerary
                    st.session_state.agent_state.current_task_status = "Detailed itinerary generated with tools"
                else:
                    st.error("Could not generate a detailed itinerary. Please try again or adjust your request.")

            status_placeholder.empty() 

        except Exception as e:
            st.error(f"An unexpected error occurred during planning: {e}")
            st.warning("Please ensure all required fields are filled correctly and your Groq API key is valid.")
            st.session_state.agent_state = None


    if st.session_state.agent_state and st.session_state.agent_state.travel_plan and st.session_state.agent_state.travel_plan.detailed_itinerary:
        st.subheader("Generated Detailed Itinerary:")
        for i, day_plan in enumerate(st.session_state.agent_state.travel_plan.detailed_itinerary):
            st.markdown(f"### 🗓️ Day {i+1}: {day_plan.date}")
            for activity in day_plan.activities:
                st.markdown(f"**⏰ {activity.time} - {activity.name}**")
                st.markdown(f"  - _Category:_ {activity.category}")
                st.markdown(f"  - _Location:_ {activity.location}")
                st.markdown(f"  - _Duration:_ {activity.estimated_duration}")
                st.write(f"  {activity.description}")
                st.markdown("---")

        st.subheader("Final Agent State (Pydantic Model):")
        st.json(st.session_state.agent_state.model_dump_json(indent=2))

        st.success("Initial itinerary generation complete!")

        st.header("Refine Your Itinerary")
        with st.form("refinement_form"):
            refinement_request = st.text_area(
                "What changes would you like to make to the itinerary?",
                "Can you suggest a different activity for Day 1 afternoon? Maybe something more active or outdoors."
            )
            refine_submitted = st.form_submit_button("Refine Itinerary")

            if refine_submitted and refinement_request:
                st.info("Refining itinerary based on your request...")
                refine_status_placeholder = st.empty()
                with refine_status_placeholder.container():
                    st.info("Re-generating itinerary with your feedback using tools...")
                    revised_itinerary = get_detailed_itinerary_with_tools( # Call the tool-enabled function
                        st.session_state.agent_state.user_input,
                        st.session_state.agent_state.travel_plan.high_level_outline,
                        existing_itinerary=st.session_state.agent_state.travel_plan.detailed_itinerary,
                        refinement_request=refinement_request
                    )
                
                if revised_itinerary:
                    st.session_state.agent_state.travel_plan.detailed_itinerary = revised_itinerary
                    st.session_state.agent_state.current_task_status = "Itinerary refined with tools"
                    st.success("Itinerary refined successfully!")
                    refine_status_placeholder.empty()
                    st.rerun()
                else:
                    st.error("Could not refine the itinerary. Please try again.")
            elif refine_submitted and not refinement_request:
                st.warning("Please enter your refinement request.")
    elif submitted:
        st.error("Failed to generate initial itinerary. Please check inputs and try again.")


if __name__ == "__main__":
    app()

