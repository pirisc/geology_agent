from typing import Annotated, TypedDict, AsyncGenerator
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from bs4 import BeautifulSoup 
import requests
import logging
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# API KEYS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Configuration constants
MAX_INPUT_LENGTH = 10000
WEB_SCRAPER_CHAR_LIMIT = 8000  # Increased for better content
WEB_SCRAPER_TIMEOUT = 20
DALLE_PROMPT_ENHANCEMENT = True  # Auto-enhance image prompts

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

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
- When you use the get_geological_image tool, the image will be displayed automatically 
  by the system. Simply mention that you're showing the image in your response, like:
  "Let me show you what granite looks like" or "Here's a real photograph of basalt columns"
- Use the image tool when it adds educational value:
  * GOOD: "Let me show you what granite actually looks like" → fetches real granite photo
  * GOOD: "Here's a real example of columnar basalt from Iceland" → shows actual formation
  * BAD: Don't show images for abstract concepts better explained with words
  * BAD: Don't show images when the user just wants text information
- When requesting images, be SPECIFIC to get the best photographic results:
  * Good: "pink granite with feldspar crystals"
  * Good: "basalt columnar joints Iceland"
  * Good: "sedimentary rock layers cliff face"
  * Bad: Generic like "rocks" or "minerals" - be specific about what to show!
- These are REAL photographs from Unsplash, so you can confidently say "Here's what X actually looks like"
  instead of "Here's an illustration of X"
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
- If 'tavily_search' is used, ALWAYS cite sources with author/publication when available.
- Prefer peer-reviewed sources and official geological surveys.

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

**STUDY MODE**: When users want to test their knowledge:
1. Generate 2-3 questions based on the discussed topic
2. Wait for their answers
3. Provide constructive feedback on each answer
4. Explain correct answers with context
5. Ask if they want more questions or to move to a new topic

Keep follow-ups natural and conversational—limit to 1-2 questions per response 
to avoid overwhelming the user.

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

# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

# Tool 1: Enhanced Web Search with Tavily
tavily_search = TavilySearchResults(
    max_results=5, 
    search_depth="advanced",
    include_images=True,
    include_answer=True,  # Get AI-generated answer
    include_raw_content=False
)

# Tool 2: Enhanced Web Scraper with Better Error Handling
@tool
def web_scraper_tool(url: str) -> str:
    """
    Scrapes a webpage and returns cleaned text content.
    Useful for reading research papers, articles, or documentation that users reference.
    
    Args:
        url: The URL to scrape (must be a valid http/https URL)
    
    Returns:
        Cleaned text content from the webpage or error message
    """
    # Validate URL
    if not url or not isinstance(url, str):
        return "Error: Invalid URL provided"
    
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        return f"Invalid URL: {url}. URL must start with http:// or https://"
    
    try:
        logger.info(f"Scraping URL: {url}")
        
        response = requests.get(
            url, 
            timeout=WEB_SCRAPER_TIMEOUT,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            },
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type and 'text/plain' not in content_type:
            return f"Error: URL returned non-text content type: {content_type}"
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace while preserving structure
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)

        # Return with character limit
        if len(clean_text) > WEB_SCRAPER_CHAR_LIMIT:
            truncated = clean_text[:WEB_SCRAPER_CHAR_LIMIT]
            return f"{truncated}\n\n[Content truncated to {WEB_SCRAPER_CHAR_LIMIT} characters. Original length: {len(clean_text)} characters]"
        
        return clean_text if clean_text else "No readable content found on page"

    except requests.Timeout:
        logger.warning(f"Timeout scraping {url}")
        return f"Error: Request timed out after {WEB_SCRAPER_TIMEOUT} seconds for {url}"
    except requests.RequestException as e:
        logger.error(f"Request error for {url}: {str(e)}")
        return f"Error fetching {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error scraping {url}: {str(e)}")
        return f"Unexpected error processing {url}: {str(e)}"
 
# Tool 3: Real Geological Image Search using Unsplash
@tool
def get_geological_image(description: str) -> str:
    """
    Fetch real, scientifically accurate geological photographs from Unsplash.
    Provides high-quality images of actual rocks, minerals, and geological formations.
    
    Use this when users want to see what geological features actually look like.
    Much better than AI-generated images for educational purposes.
    
    Best for:
    - Rock types and mineral samples (granite, basalt, quartz, etc.)
    - Geological structures (canyons, cliffs, mountains)
    - Natural formations (columnar basalt, sedimentary layers)
    - Landscapes showing geological features
    - Fossils and paleontological specimens
    
    Args:
        description: What geological feature to show (be specific: "pink granite with feldspar crystals")
    
    Returns: 
        URL of real geological photograph from Unsplash
    """
    try:
        logger.info(f"Searching for geological image: {description}")
        
        # Get Unsplash API key from environment
        import os
        unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
        
        if not unsplash_key:
            logger.warning("UNSPLASH_ACCESS_KEY not found in environment variables")
            # Fallback to web search
            return search_geological_image_fallback(description)
        
        # Search Unsplash for geological images
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": f"{description} geology rock mineral formation",
            "client_id": unsplash_key,
            "per_page": 3,  # Get top 3 to choose best
            "orientation": "landscape",
            "content_filter": "high"  # High quality only
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('results') and len(data['results']) > 0:
            # Get the best matching image
            best_image = data['results'][0]
            image_url = best_image['urls']['regular']
            photographer = best_image['user']['name']
            photo_link = best_image['links']['html']
            
            logger.info(f"Found Unsplash image by {photographer}: {image_url[:50]}...")
            
            # Return just the URL - the frontend will display it
            return image_url
        else:
            logger.warning(f"No Unsplash results for: {description}")
            return search_geological_image_fallback(description)
            
    except requests.RequestException as e:
        logger.error(f"Unsplash API error: {e}")
        return search_geological_image_fallback(description)
    except Exception as e:
        logger.error(f"Error fetching geological image: {e}")
        return f"Unable to fetch image: {str(e)}"

def search_geological_image_fallback(description: str) -> str:
    """
    Fallback method using web search when Unsplash is unavailable.
    Searches for geological images on the web.
    """
    try:
        logger.info(f"Using fallback web search for: {description}")
        
        # Use the existing tavily search
        search_query = f"{description} geology high resolution photo"
        results = tavily_search.invoke(search_query)
        
        # Try to extract image URLs from results
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    # Check if result has images
                    if 'images' in result and result['images']:
                        logger.info(f"Found image via web search: {result['images'][0][:50]}...")
                        return result['images'][0]
        
        # If no images found, return a helpful message
        logger.warning(f"No images found via fallback for: {description}")
        return f"NO_IMAGE_FOUND: {description}"
        
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        return f"IMAGE_ERROR: {str(e)}"

# Tool 4: Enhanced Quiz Generation
@tool
def generate_quiz_questions(topic: str, difficulty: str = "intermediate", num_questions: int = 2) -> str:
    return f"Generate {num_questions} {difficulty}-level questions about {topic} to test the user's understanding."
    
# Tool list
tools = [
    tavily_search, 
    web_scraper_tool,
    get_geological_image,  # Now using real photos instead of DALL-E
    generate_quiz_questions
]

# ═══════════════════════════════════════════════════════════════════════════
# STATE & GRAPH
# ═══════════════════════════════════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the graph
graph_builder = StateGraph(State)

# Initialize LLM with improved settings
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    presence_penalty=0.6,
    frequency_penalty=0.5,
    top_p=0.9
).bind_tools(tools=tools)

def chatbot(state: State):
    """Main chatbot node that processes messages with error handling."""
    try:
        messages = state["messages"]
        
        # Log conversation for debugging (first 100 chars of last message)
        if messages:
            last_msg = str(messages[-1])[:100]
            logger.info(f"Processing message: {last_msg}...")
        
        response = llm.invoke(messages)
        return {"messages": [response]}
        
    except Exception as e:
        logger.error(f"Error in chatbot node: {str(e)}", exc_info=True)
        raise RuntimeError(f"Error processing your request: {str(e)}") from e

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Add edges
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Compile graph with memory
graph = graph_builder.compile(checkpointer=MemorySaver())

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def validate_input(user_input: str) -> tuple[bool, str]:
    """
    Validate user input with comprehensive checks.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not user_input:
        return False, "Error: No input provided"
    
    if not isinstance(user_input, str):
        return False, "Error: Input must be a string"
    
    if not user_input.strip():
        return False, "Error: Input is empty or only whitespace"
    
    if len(user_input) > MAX_INPUT_LENGTH:
        return False, f"Error: Input exceeds maximum length of {MAX_INPUT_LENGTH} characters (got {len(user_input)})"
    
    return True, ""

async def run_agent(user_input: str, thread_id: str) -> AsyncGenerator[str, None]:
    """
    Stream tokens as they are generated from the Rocky geology chatbot.
    
    Args:
        user_input: The user's question or message
        thread_id: Unique identifier for the conversation thread
    
    Yields:
        String tokens as they are generated
    """
    # Validate input
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        logger.warning(f"Invalid input: {error_msg}")
        yield error_msg
        return
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        logger.info(f"Starting agent run for thread {thread_id}")

        # Check if this is a new conversation
        state = graph.get_state(config)
        is_new_conversation = not state.values.get("messages")

        # Only include system prompt on first message
        if is_new_conversation:
            messages = [
                ("system", SYSTEM_PROMPT),
                ("user", user_input)
            ]
            logger.info(f"New conversation started for thread {thread_id}")
        else:
            messages = [("user", user_input)]

        token_count = 0
        async for event in graph.astream_events(
            {"messages": messages},
            config=config,
            version="v2"
        ):
            kind = event["event"]

            # Stream LLM tokens
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    token_count += 1
                    yield content

            # Capture tool outputs (like image urls)
            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                
                # Handle the get_geological_image tool
                if tool_name == "get_geological_image":
                    image_url = event["data"]["output"]
                    if hasattr(image_url, "content"):
                        image_url = image_url.content
                    image_url = str(image_url).strip()
                    
                    # Only yield if it's a valid URL
                    if image_url.startswith("http"):
                        yield f"\n\n![Geological Image]({image_url})\n\n"
                        logger.info(f"Displaying geological image: {image_url[:50]}...")
                    elif "NO_IMAGE_FOUND" in image_url or "IMAGE_ERROR" in image_url:
                        logger.warning(f"Image fetch failed: {image_url}")
                        # Don't show error to user, let the LLM handle it
        
        logger.info(f"Completed agent run for thread {thread_id}, tokens: {token_count}")

    except Exception as e:
        error_message = f"\n\n❌ An error occurred: {str(e)}\n\nPlease try rephrasing your question or start a new chat."
        logger.error(f"Error in run_agent for thread {thread_id}: {str(e)}", exc_info=True)
        yield error_message
