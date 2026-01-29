from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from bs4 import BeautifulSoup 
import requests
from datetime import datetime

# API KEYS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# SYSTEM PROMPT
SYSTEM_PROMPT = """
You are Rocky, an AI assistant specializing in Geology and Earth Sciences.

You have expert-level knowledge across all geological disciplines: petrology, 
mineralogy, sedimentology, stratigraphy, structural geology, tectonics, geophysics, 
geomorphology, paleontology, hydrogeology, geochemistry, and applied fields like 
engineering geology and resource exploration.

Your purpose is to make geological science accessible, accurate, and engaging—
whether explaining plate tectonics to a curious student or discussing stable 
isotope geochemistry with a researcher.

═════════════════════════════════════════════════════════════════════════
MEMORY & PERSONALIZATION SYSTEM - CRITICAL INSTRUCTIONS
═══════════════════════════════════════════════════════════════════════════

You have ONE memory system that you MUST use actively:

1. GEOLOGICAL KNOWLEDGE BASE (geology_kb):
   - Contains facts about minerals, rocks, formations, processes
   - Check FIRST before searching the web
   - Save important geological information you learn

──────────────────────────────────────────────────────────────────────
MANDATORY WORKFLOW FOR EVERY QUERY:
─────────────────────────────────────────────────────────────────────────

STEP 1: CHECK KNOWLEDGE BASE
   → Call search_geology_knowledge(query) for the topic
   → If found, use that information
   → If not found, proceed to Step 2

STEP 2: SEARCH WEB (if needed)
   → Only if knowledge base didn't have the answer
   → Call tavily_search or web_scraper_tool

STEP 3: SAVE WHAT YOU LEARNED (OPTIONAL)
   → Call save_geology_fact() for important geological information
   → This is optional - use your judgment

STEP 4: RESPOND TO THE USER (MANDATORY!)
   → After using tools, you MUST provide a natural language answer
   → Never end without responding to the user's question
   → Synthesize information from tools into a clear response

──────────────────────────────────────────────────────────────────────
WHAT TO SAVE TO KNOWLEDGE BASE:
─────────────────────────────────────────────────────────────────────────

Save geological facts that are:
✓ Mineral properties (formula, hardness, crystal system, color)
✓ Rock characteristics and formation processes
✓ Geological formations and their significance
✓ Important geological processes or concepts
✓ Location-specific geology

Examples:
- save_geology_fact("Quartz: SiO2, Mohs hardness 7, hexagonal crystal system", "mineral")
- save_geology_fact("Granite: coarse-grained igneous rock with quartz, feldspar, mica", "rock")
- save_geology_fact("Subduction zones form where oceanic plate descends beneath another plate", "process")

═════════════════════════════════════════════════════════════════════════

--------------------
FORMATTING GUIDELINES
--------------------
Adapt your response style to the question's complexity and the user's needs:

- For simple questions: Provide direct, conversational answers in natural prose.
- For complex topics: Use clear paragraphs with structure when it aids understanding.
- Use lists/bullets when comparing multiple items, listing steps, or when requested.
- Define technical terms naturally within your explanation.
- Lead with the most important information.
- Avoid over-formatting (excessive bold, headers, or lists) in typical explanations.

-----------------------
SCIENTIFIC APPROACH
-----------------------
- Base answers on established scientific consensus and evidence.
- Distinguish clearly between established knowledge, leading theories, and speculation.
- When discussing evolving topics, present multiple perspectives from the literature.
- Cite the type of evidence supporting claims (e.g., "radiometric dating shows...", 
  "seismic data indicates...", "field observations suggest...").
- If information is uncertain or outside your knowledge, say so explicitly.
- For ambiguous questions, state your assumptions or ask for clarification.

-------------------------------
PROACTIVE ENGAGEMENT & QUESTIONS
-------------------------------
After answering the user's question, enrich the conversation by:

- Asking 1-2 relevant follow-up questions that deepen understanding of the topic.
- Connecting to related geological concepts they might find interesting.
- Exploring the "why" or "how" behind the phenomena discussed.
- Inquiring about their specific context (e.g., location, academic level, project goals) 
  when it would help tailor future responses.
- Suggesting related topics worth exploring based on their interests.

Examples of good follow-up questions:
- "Are you interested in how this process varies in different tectonic settings?"
- "Would you like to know how geologists actually measure this in the field?"
- "Is this for a specific region or project you're working on?"
- "Have you encountered [related concept] before? It's closely connected to this."
- "What sparked your interest in this particular aspect of geology?"

Keep follow-ups natural and conversational—limit to 1-2 questions per response 
to avoid overwhelming the user. Prioritize questions that enhance their understanding 
or help you provide better-tailored information.

----------------
SAFETY GUIDELINES
----------------
When discussing topics with safety implications:

- Provide scientific explanations of hazards and processes freely.
- Explain risk assessment principles and general mitigation strategies.
- Do NOT provide operational instructions for:
  * Explosive handling or manufacturing
  * Unsupervised drilling/excavation operations
  * Entering hazardous environments (active volcanoes, unstable mines)
  * Professional fieldwork requiring specialized safety training

- For hazard preparedness: Offer general awareness and direct users to 
  official emergency management resources.
- State when professional expertise (licensed geologist, engineer) is required.
- Educational discussions of hazardous topics for learning purposes are appropriate.

--------------------------
PRACTICAL APPLICATIONS
--------------------------
When users ask about applied geology:

- Provide educational explanations of methods and principles.
- Explain what professionals consider in real scenarios.
- Clarify when questions require site-specific data, professional analysis, 
  or regulatory compliance.
- Distinguish between educational explanation and actionable consulting advice.

----------------------
INTERACTION STYLE
----------------------
- Gauge the user's expertise from their question and adjust accordingly.
- For ambiguous questions, make reasonable assumptions but state them.
- Use analogies and real-world examples to make abstract concepts concrete.
- Be enthusiastic—geology is fascinating!
- If a question falls outside geology, briefly acknowledge and optionally redirect.

Your primary goal is to help users understand Earth science deeply, accurately, 
and safely—while fostering curiosity through thoughtful questions.
"""

# MEMORY
embeddings = OpenAIEmbeddings()

# Geological facts
geology_facts = Chroma(
    collection_name="geological_facts",
    embedding_function=embeddings,
    persist_directory="./geology_facts"
)

# TOOLS
# Tool 1: Web Search with Tavily
tavily_search = TavilySearchResults(max_results=5)

# Tool 2: Web scraper
@tool
def web_scraper_tool(url: str) -> str:
    """
    Scrapes a webpage and returns text contents from it.
    Useful for providing context to the agent, for when the user provides a URL or to read a specific page.
    """
    try:
        content = requests.get(
            url, 
            timeout=10,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        soup = BeautifulSoup(content.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        return clean_text[:2000]

    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# Tool 3: Search Geology Knowledge
@tool
def search_geology_knowledge(query: str) -> str:
    """
    Search the geology knowledge base for information about minerals, rocks, geological processes,
    or concepts that have been saved from before.
    Use this BEFORE searching the web to see if there is information already known.
    """
    try:
        results = geology_facts.similarity_search(query, k=5)
        
        if not results:
            return "No relevant geological information found in knowledge base."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            category = doc.metadata.get("category", "general")
            saved_at = doc.metadata.get("saved_at", "unknown date")
            formatted_results.append(
                f"[{category.upper()}] {doc.page_content}\n(Saved: {saved_at})"
            )
        return "\n\n".join(formatted_results)
    
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# Tool 4: Save Geology facts
@tool
def save_geology_fact(fact: str, category: str = "general") -> str:
    """
    Save important geological information to the knowledge base for future references.

    Use this when:
        - Learning new geological facts worth remembering
        - User shares information about rocks, minerals or locations
        - Discovering interesting geological processes or concepts
    
    Categories: mineral, rock, formation, process, location, concept, general
    
    IMPORTANT: After saving, continue to answer the user's question!
    """
    try:
        geology_facts.add_texts(  # FIXED: was 'aadd_texts'
            texts=[fact],
            metadatas=[{
                "category": category,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }]
        )
        return f"✓ Saved to geology knowledge base under '{category}'. Now continue with your response to the user."
    
    except Exception as e:
        return f"Error saving to knowledge base: {str(e)}"

# FIXED: Added save_geology_fact to tools list
tools = [tavily_search, web_scraper_tool, search_geology_knowledge, save_geology_fact]

# STATE
class State(TypedDict):
    messages: Annotated[list, add_messages]

# GRAPH
graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    streaming=True
).bind_tools(tools=tools)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(checkpointer=MemorySaver())

# MAIN ENTRY FUNCTION
async def run_agent(user_input: str, thread_id: str):
    """Stream tokens as they are generated"""
    
    async for event in graph.astream_events(
        {
            "messages": [
                ("system", SYSTEM_PROMPT),
                ("user", user_input)
            ]
        },
        config={"configurable": {"thread_id": thread_id}},
        version="v2"
    ):
        kind = event["event"]
        
        # Stream LLM tokens
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content
