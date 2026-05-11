### IMPORTS###

# 1. Standard Library
import logging
import os
from typing import Annotated, AsyncGenerator, TypedDict

# 2. Third-Party Libraries
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
import requests
import psycopg

# 3. LangChain & LangGraph
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

# 4. Local applications
from build_knowlege_base import load_vector_store


# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

# This way is to allow change settings from the .env file (faster)
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", 1000))
WEB_SCRAPER_CHAR_LIMIT = int(os.getenv("WEB_SCRAPER_CHAR_LIMIT", 8000))
WEB_SCRAPER_TIMEOUT = int(os.getenv("WEB_SCRAPER_TIMEOUT", 20))
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "geology_kb")

# Load vector store when the server starts
try:    
    VECTOR_DB = load_vector_store(VECTOR_DB_DIR)
    logger.info(f"✅ Rocky has successfully connected to the Geology Knowledge Base at: {VECTOR_DB_DIR}")
except Exception as e:
    VECTOR_DB = None
    logger.error(f"❌ Rocky could not find the database at {VECTOR_DB_DIR}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
You are Rocky, an expert AI Geologist. You bridge the gap between complex Earth Science and accessible education.

CORE OPERATING PRINCIPLES:
1. TOOL FIRST: For any factual geological query, use 'search_geology_knowledge_base'. If that yields no results, use 'tavily_search'.
2. ACCURACY: Distinguish between scientific consensus and theory. Cite evidence (e.g., "seismic data suggests").
3. VISUALS: Proactively use 'find_geological_images' when describing physical objects (rocks, minerals) or processes (subduction).
4. SAFETY: Discuss hazards (volcanoes, quakes) educationally. Never provide DIY instructions for explosives or hazardous site entry.

RESPONSE STRUCTURE:
- Simple Questions: Conversational prose. Define technical terms in-line.
- Complex Topics: Use markdown headers (##) and bullet points for scannability.
- Synthesis: Do not just dump tool outputs. Summarize the findings from the knowledge base into a cohesive narrative.

ENGAGEMENT:
Always end with 1-2 natural follow-up questions to spark curiosity. 
Example: "Would you like to know how this formation looks in a different tectonic setting?"

STUDY MODE:
If a user wants to test knowledge, use 'start_quiz_mode' to structure the interaction.
"""


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

# Tool 1: Web Search
tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)


# Tool 2: Web Scrapper 
@tool
def web_scraper_tool(url: str) -> str:
    """Scrape a webpage and return cleaned text content."""
    if not url or not isinstance(url, str):
        return "Error: Invalid URL provided"

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        return f"Invalid URL: {url}. URL must start with http:// or https://"

    try:
        response = requests.get(
            url,
            timeout=WEB_SCRAPER_TIMEOUT,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                )
            },
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return f"Error: URL returned non-text content type: {content_type}"

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) > WEB_SCRAPER_CHAR_LIMIT:
            truncated = clean_text[:WEB_SCRAPER_CHAR_LIMIT]
            return (
                f"{truncated}\n\n"
                f"[Content truncated to {WEB_SCRAPER_CHAR_LIMIT} characters. "
                f"Original length: {len(clean_text)} characters]"
            )

        return clean_text if clean_text else "No readable content found on page"

    except requests.Timeout:
        return f"Error: Request timed out after {WEB_SCRAPER_TIMEOUT} seconds for {url}"
    except requests.RequestException as exc:
        return f"Error fetching {url}: {str(exc)}"
    except Exception as exc:
        return f"Unexpected error processing {url}: {str(exc)}"


# Tool 3: Find geological images from Google images
@tool
def find_geological_images(topic: str) -> str:
    """
    Find and display geological images from Google Images.
    
    Args:
        topic: What to search for (e.g., "basalt", "subduction zone diagram", "quartz crystal")
    
    Returns:
        Markdown-formatted image or fallback link
    """
    if not topic.strip():
        return "Please provide a valid geology topic"
    
    topic_lower = topic.lower()
    
    # Build a targeted search query
    if "mineral" in topic_lower or "crystal" in topic_lower or "gem" in topic_lower:
        search_query = f"{topic} mineral geology specimen"
    elif any(word in topic_lower for word in ["diagram", "process", "cycle", "cross-section", "cross section", "plate", "boundary", "structure", "how"]):
        search_query = f"{topic} geology diagram illustration"
    elif any(word in topic_lower for word in ["rock", "outcrop", "formation", "sample", "stone"]):
        search_query = f"{topic} rock geology sample"
    else:
        search_query = f"{topic} geology"
    
    # Try to get image from Google using Tavily (which can search Google)
    try:
        # Use tavily to search, which will return images from various sources including Google
        results = tavily_search.invoke({"query": search_query})
        
        # Try to extract an image URL from results
        image_url = None
        if isinstance(results, dict):
            # Check for images in the response
            if "images" in results and results["images"]:
                image_url = results["images"][0]
            # Also check in results items
            elif "results" in results and results["results"]:
                for result in results["results"]:
                    if isinstance(result, dict) and "images" in result and result["images"]:
                        image_url = result["images"][0]
                        break
        
        if image_url and image_url.startswith("http"):
            # Return the image in markdown format so it displays in chat
            encoded_query = search_query.replace(' ', '+')
            google_search_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
            return (
                f"![{topic}]({image_url})\n\n"
                f"_See more images: [Google Images search for '{topic}']({google_search_url})_"
            )
    
    except Exception as e:
        logger.warning(f"Failed to fetch image for '{topic}': {e}")
    
    # Fallback: just provide the Google Images link
    encoded_query = search_query.replace(' ', '+')
    google_images_url = f"https://www.google.com/search?q={encoded_query}&tbm=isch"
    
    return (
        f"📸 **[View images of '{topic}' on Google Images]({google_images_url})**\n\n"
        f"_Click to see high-quality geological images._"
    )


# Tool 4: Quiz mode
@tool
def start_quiz_mode(topic: str, difficulty: str = "intermediate", num_questions: int = 3) -> str:
    """
    Start an interactive geology quiz on a specific topic.
    
    Args:
        topic: The geological topic to quiz on (e.g., "plate tectonics", "igneous rocks")
        difficulty: Difficulty level - "easy", "intermediate", or "advanced"
        num_questions: How many questions to generate (1-5)
    
    Returns:
        Instructions for generating quiz questions
    """
    # Validate inputs
    num_questions = max(1, min(5, int(num_questions)))
    difficulty = (difficulty or "intermediate").lower()
    if difficulty not in {"easy", "intermediate", "advanced"}:
        difficulty = "intermediate"
    
    return (
        "🎯 QUIZ MODE ACTIVATED\n\n"
        f"Generate {num_questions} {difficulty}-level multiple-choice questions about '{topic}'.\n\n"
        "Format:\n"
        "- Provide 4 options (A, B, C, D) for each  question\n"
        "- Make questions test understanding, not just memorization\n"
        "- After presenting all questions, WAIT for the user to answer\n"
        "- Then provide feedback and explanations for each answer\n\n"
        "End with: 'Take your time! Reply with your answers (e.g., 1-A, 2-C, 3-B) when ready.'"
    )


# Tool 5: Search geology knowledge
@tool
def search_geology_knowledge_base(query: str)->str:
    """Search Rocky's geology knowledge base for relevant information.
        Sources: geological associations and other public domain geological resources 
        Args:
            query: what the user want to search
        
        Returns:
            relevant information with source citations about the user's query
    """
    if VECTOR_DB is None:
        return "⚠️ Error: The internal knowledge base is not currently loaded."

    try:
        #Search for relevant chunks
        results = VECTOR_DB.similarity_search_with_relevance_scores(query, k = 3)

        # Format results
        if not results:
            return "🔍 I searched the internal records but couldn't find a direct match for that query."
        
        formatted_results = "🔍 **Relevant information found it in Rocky's knowledge base:**\n\n"
        for i, result in enumerate(results, 1):
            source_page = result.metadata.get("page", "N/A")
            source_title = result.metadata.get("title", "Geological document")
            
            formatted_results += f"**Source {i}** found it in {source_title}, page {source_page}:\n"
            formatted_results += f"{result.page_content}\n\n"
            formatted_results += "---\n"
        
        return formatted_results
        
    
    except Exception as e:
        return f"❌ Error searching knowledge database: {str(e)}"


# Tool 6: Get recent information about earthquakes
@tool
def get_earthquake_data(location: str = "worldwide", magnitude_min: float = 4.5)-> str:
    """Get recent eathquakes data from USGS GeoJSON feed.
        Args:
            location: location where the user want to search
            magnitude_min: the minimum magnitude for the earthquakes to appear in the search
        
        Return:
            recent earthquake information from USGS.
    """

    try:
        minmag = max(0.0, float(magnitude_min))
        feed_url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query"
            "?format=geojson&orderby=time&limit=10"
            f"&minmagnitude={minmag}"
        )

        data = requests.get(feed_url, timeout=15).json()
        features = data.get("features", [])
        if not features:
            return f"No recent earthquakes found with magnitude >= {minmag}."

        lines = [f"Recent earthquakes (USGS, magnitude >= {minmag}):"]
        location_filter = location.strip().lower()

        for item in features:
            props = item.get("properties", {})
            place = props.get("place", "Unknown location")
            mag = props.get("mag", "?")
            ts_ms = props.get("time")
            usgs_url = props.get("url", "")

            if location_filter != "worldwide" and location_filter not in place.lower():
                continue

            lines.append(f"- M{mag} | {place} | {ts_ms} | {usgs_url}")

        if len(lines) == 1:
            return f"No recent events matched location='{location}'."

        lines.append("Source: USGS Earthquake Hazards Program")
        return "\n".join(lines)

    except Exception as exc: 
        logger.exception("USGS lookup failed")
        return f"Unable to fetch earthquake data: {exc}"


# Register all tools
tools = [
    tavily_search,
    web_scraper_tool,
    find_geological_images,
    start_quiz_mode,
    search_geology_knowledge_base,
    get_earthquake_data
]


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH SETUP
# ═══════════════════════════════════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Get the database url
db_url = os.getenv('DATABASE_URL')

if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    try:
        conn = psycopg.connect(db_url)
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        logger.info('✓ Using PostgreSQL checkpointer')
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")
        checkpointer = MemorySaver()
        logger.warning("⚠️ Falling back to MemorySaver (no persistence)")

else:
    # Local development: Use SQLite
    db_path = os.getenv("ROCKY_DB_PATH", "rocky_conversations.db")
    checkpointer = SqliteSaver.from_conn_string(db_path)
    logger.info("✓ Using SQLite checkpointer (local only)")


graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    presence_penalty=0.6,
    frequency_penalty=0.5,
    top_p=0.9,
).bind_tools(tools=tools)


def chatbot(state: State):
    """Main chatbot node.
        Injects the SYSTEM_PROMPT into every turn to ensure the persona doesn't drift.
        It also handles potential API/Network error
    """

    try:
        messages = state["messages"]

        # Ensure the system prompt is always at the top
        # check the first message to avoid duplicatin it
        if not messages or not (isinstance(messages[0], tuple) and messages[0][0] == "system"):
            messages = [("system", SYSTEM_PROMPT)] + messages
        
        # call the llm
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    except Exception as e:
        logger.error(f"Chatbot Node Error: {str(e)}")

        # Return friendly message to user
        error_message = (
            "🪨 **Rocky here!** I've encountered a bit of a landslide in my circuits. "
            "My connection to the geological database was briefly interrupted. "
            "Could you please try your question again?"
        )
        return {"messages": [("assistant", error_message)]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer = checkpointer)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def validate_input(user_input: str) -> tuple[bool, str]:
    """Validate user input."""
    if not user_input:
        return False, "Error: No input provided"
    if not isinstance(user_input, str):
        return False, "Error: Input must be a string"
    if not user_input.strip():
        return False, "Error: Input is empty or only whitespace"
    if len(user_input) > MAX_INPUT_LENGTH:
        return False, (
            f"Error: Input exceeds maximum length of {MAX_INPUT_LENGTH} characters "
            f"(got {len(user_input)})"
        )
    return True, ""


async def run_agent(user_input: str, thread_id: str) -> AsyncGenerator[str, None]:
    """Stream generated tokens for the geology chatbot."""
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        logger.warning("Invalid input: %s", error_msg)
        yield error_msg
        return

    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        is_new_conversation = not state.values.get("messages")

        if is_new_conversation:
            messages = [("system", SYSTEM_PROMPT), ("user", user_input)]
        else:
            messages = [("user", user_input)]

        async for event in graph.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            kind = event.get("event")

            # Stream LLM text output
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content
            
            # Handle tool outputs (especially images)
            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                if tool_name == "find_geological_images":
                    # Get the tool output
                    output = event["data"].get("output")
                    if hasattr(output, "content"):
                        output = output.content
                    output = str(output).strip()
                    
                    # Stream the tool output (which includes image markdown or links)
                    if output:
                        yield f"\n\n{output}\n\n"

    except Exception as exc:
        error_message = (
            f"\n\n❌ An error occurred: {str(exc)}\n\n"
            "Please try rephrasing your question or start a new chat."
        )
        logger.error("Error in run_agent for thread %s: %s", thread_id, str(exc), exc_info=True)
        yield error_message
