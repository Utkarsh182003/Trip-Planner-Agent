✈️ AI-Powered Smart Travel Planner
Unleash Your Inner Explorer with Intelligent Itinerary Generation
This project presents an AI-Powered Smart Travel Planner, a sophisticated application designed to revolutionize how you plan your trips. Leveraging cutting-edge Large Language Models (LLMs) and advanced information retrieval techniques, it acts as your personal, intelligent travel concierge, crafting personalized, detailed itineraries based on your unique preferences.

Forget endless hours of research – simply tell the planner where and when you want to go, your interests, and your budget, and watch as it dynamically generates a comprehensive travel plan, complete with daily activities, timings, and descriptions.

✨ Features
Intuitive User Interface (Streamlit): A clean and interactive web application built with Streamlit for easy input and itinerary visualization.

Intelligent Itinerary Generation (Groq LLM): Powered by Groq's high-performance LLMs (e.g., Llama3-8B), the core intelligence generates high-level outlines and detailed, day-by-day itineraries.

Structured Output (Pydantic): Ensures all generated itinerary data adheres to a strict, machine-readable JSON schema, making the output reliable and consistent.

Dynamic Tool Orchestration: The LLM intelligently decides when and how to use various information-gathering tools to provide the most accurate and up-to-date itinerary details. This is a key differentiator!

Retrieval-Augmented Generation (RAG): Integrates an in-memory RAG system to pull relevant information from a curated knowledge base (e.g., general facts about attractions).

Knowledge Graph (KG): Utilizes a simple in-memory Knowledge Graph for structured, precise facts about entities like cities, landmarks, and cuisines (e.g., Louvre Museum's closing days).

Real-time Web Search (Tavily API): Connects to the internet via Tavily API to fetch the latest information, such as current events, dynamic opening hours, or recent reviews.

Interactive Refinement: Allows users to provide feedback on the generated itinerary and request modifications, enabling the LLM to revise and adapt the plan dynamically.

Robust Error Handling: Provides clear feedback and graceful handling for common issues.

Session Management: Utilizes Streamlit's session state to maintain the planning context across interactions.

🧠 How It Works (The Intelligent Agent's Journey)
The heart of this planner is an intelligent agent driven by the Groq LLM, capable of dynamic decision-making and tool use.

User Request: You provide your travel details (destination, dates, interests, budget) via the Streamlit UI. This input is validated by Pydantic models.

High-Level Planning: The Groq LLM first generates a concise, high-level overview of your trip.

Intelligent Information Gathering (The Core Loop):

The LLM then receives your detailed request and the high-level plan.

It reasons about what specific information it needs to create a detailed itinerary (e.g., "What are the must-see historical sites in Paris?", "Are there any food festivals in Rome in July?", "What are the opening hours of the Eiffel Tower?").

Based on this reasoning, the LLM dynamically decides which tool to call:

rag_retrieve: For general, broad factual information from its internal document store.

kg_get_entity_info / kg_find_attractions: For structured, precise facts about known entities.

web_search: For real-time, highly specific, or dynamic information from the internet.

Our Python backend executes the tool call requested by the LLM.

The results from the tool (e.g., web snippets, KG data) are fed back to the LLM.

This process iterates until the LLM has gathered sufficient information to confidently generate the full itinerary.

Detailed Itinerary Generation: Once satisfied, the LLM generates the complete, day-by-day itinerary in a structured JSON format, which is then validated by Pydantic and displayed beautifully in the Streamlit UI.

Refinement Loop: If you request changes, your feedback is sent back to the LLM, along with the current itinerary and all previously gathered context. The LLM then intelligently revises the plan.

This dynamic tool orchestration allows the planner to adapt to diverse queries and provide highly relevant, context-aware itineraries.

🚀 Getting Started
Follow these steps to set up and run the AI-Powered Smart Travel Planner on your local machine.

Prerequisites
Python 3.8+

pip (Python package installer)

1. Clone the Repository
git clone <your-repository-url>
cd travel_planner_agent # Or whatever your project folder is named

2. Set Up Your Environment
It's highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install Dependencies
Install all required Python packages:

pip install -r requirements.txt

4. Configure API Keys
You'll need API keys for Groq and Tavily.

Groq API Key: Obtain one from Groq Console.

Tavily API Key: Obtain one from Tavily AI.

Create a file named .env in the root directory of your project (where main.py is located) and add your keys:

GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"

Replace the placeholder values with your actual API keys.

5. Run the Application
Once dependencies are installed and API keys are set, run the Streamlit app:

streamlit run main.py

Your browser should automatically open to the Streamlit application. If not, open your web browser and go to http://localhost:8501.

💡 Potential Future Enhancements
Advanced Knowledge Graph: Integrate with a persistent graph database (e.g., Neo4j) for more complex relationships and larger datasets.

Real-time Booking Integration: Connect to flight, hotel, and activity booking APIs.

User Profiles & History: Implement user accounts to save and retrieve past itineraries and learn user preferences over time.

Multimodality: Allow image inputs (e.g., "Plan a trip like this picture").

Cost Estimation: Provide estimated costs for activities, transport, and accommodation.

Local Event Integration: Dynamically pull local events and festivals for the travel dates.

🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests.

📄 License
This project is open-sourced under the MIT License. See the LICENSE file for more details.

Made with ❤️ by [Your Name/Your GitHub Handle]