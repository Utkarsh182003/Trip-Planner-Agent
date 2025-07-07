# ✈️ AI-Powered Smart Travel Planner

**Unleash Your Inner Explorer with Intelligent Itinerary Generation**

---

## Overview

The **AI-Powered Smart Travel Planner** is your personal, intelligent travel concierge. Powered by advanced Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), a Knowledge Graph, and real-time web search, this app crafts personalized, detailed itineraries based on your unique preferences—saving you hours of research and planning.

---

## ✨ Features

- **Intuitive UI**: Clean, interactive web app built with Streamlit.
- **LLM-Driven Planning**: Uses Groq’s Llama3-8B for high-level and detailed itinerary generation.
- **Structured Output**: All plans validated with Pydantic models for reliability.
- **Dynamic Tool Orchestration**: LLM decides when to use RAG, Knowledge Graph, or Web Search for the best results.
- **Retrieval-Augmented Generation (RAG)**: Pulls relevant info from a curated knowledge base.
- **Knowledge Graph**: Structured facts about cities, attractions, and cuisines.
- **Real-Time Web Search**: Fetches up-to-date info (e.g., opening hours, events) via Tavily API.
- **Interactive Refinement**: Users can request changes and instantly refine their itinerary.
- **Robust Error Handling**: Clear feedback for common issues.
- **Session Management**: Maintains planning context across interactions.

---

## 🧠 How It Works

1. **User Input**: Enter your destination, dates, interests, budget, and preferences.
2. **High-Level Planning**: The LLM generates a concise trip outline.
3. **Dynamic Information Gathering**: The LLM reasons about what it needs and calls:
   - `rag_retrieve`: For general facts from the internal knowledge base.
   - `kg_get_entity_info` / `kg_find_attractions`: For structured, precise facts.
   - `web_search`: For real-time, dynamic info from the web.
4. **Iterative Tool Use**: The LLM gathers info, refines its plan, and generates a detailed, day-by-day itinerary.
5. **Refinement Loop**: Users can request changes; the LLM revises the plan accordingly.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the Repository

```sh
git clone <your-repository-url>
cd travel_planner_agent