from typing import Annotated, TypedDict, AsyncGenerator
import logging
import os
import re

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

-----------------------
USING IMAGES
-----------------------
You have a tool called 'get_geological_image' that automatically fetches the right type of image:

**The tool is smart - just describe what you want:**
- "tectonic plate boundaries diagram" → fetches diagram
- "basalt rock sample" → fetches photo  
- "rock cycle diagram" → fetches diagram
- "granite outcrop" → fetches photo

**Be specific in your descriptions:**
- Good: "subduction zone cross-section diagram"
- Good: "sedimentary rock layers in cliff"
- Bad: Generic terms like "rocks" or "geology"

**How to use it naturally:**
- Simply mention you're showing an image: "Let me show you a diagram of plate boundaries"
- The image will display automatically - you don't need to describe it in detail
- Use images to enhance understanding, especially for visual concepts

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
# SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

load_dotenv(find_dotenv(), override=True)

MAX_INPUT_LENGTH = 10000
WEB_SCRAPER_CHAR_LIMIT = 8000
WEB_SCRAPER_TIMEOUT = 20


# Tool 1: Web Search with Tavily
tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)


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


@tool
def get_geological_image(description: str) -> str:
    """
    Fetch geological images - automatically chooses photos or diagrams based on description.
    
    Fetches diagrams for: processes, cross-sections, cycles, plate tectonics, structures
    Fetches photos for: rock samples, minerals, formations, outcrops
    
    Args:
        description: What to search for (e.g., 'granite rock sample', 'subduction zone diagram')
    
    Returns:
        Image URL or error message
    """
    try:
        # Simple auto-detection: look for diagram keywords
        diagram_keywords = ["diagram", "cross-section", "cycle", "process", "plate", "boundary", "structure"]
        needs_diagram = any(keyword in description.lower() for keyword in diagram_keywords)
        
        image_type = "diagram" if needs_diagram else "photo"
        logger.info(f"Fetching {image_type} for: {description}")
        
        if image_type == "diagram":
            return _fetch_from_wikimedia(description)
        else:
            return _fetch_from_unsplash(description)
            
    except Exception as exc:
        logger.error(f"Image fetch error: {exc}")
        return f"NO_IMAGE_FOUND: {description}"


def _fetch_from_unsplash(description: str) -> str:
    """Fetch photos from Unsplash API."""
    unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
    if not unsplash_key:
        logger.warning("No Unsplash key found")
        return f"NO_IMAGE_FOUND: {description}"
    
    try:
        response = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": f"{description} geology",
                "client_id": unsplash_key,
                "per_page": 5,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if results:
            return results[0]["urls"]["regular"]
        
        logger.info(f"No Unsplash results for: {description}")
        return f"NO_IMAGE_FOUND: {description}"
        
    except Exception as exc:
        logger.error(f"Unsplash error: {exc}")
        return f"NO_IMAGE_FOUND: {description}"


def _fetch_from_wikimedia(description: str) -> str:
    """Fetch diagrams from Wikimedia Commons."""
    try:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrsearch": f"{description} geology diagram",
                "gsrlimit": 5,
                "prop": "imageinfo",
                "iiprop": "url",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo", [])
            if imageinfo:
                url = imageinfo[0].get("url", "")
                if url.startswith("http"):
                    return url
        
        logger.info(f"No Wikimedia results for: {description}")
        return f"NO_IMAGE_FOUND: {description}")
        
    except Exception as exc:
        logger.error(f"Wikimedia error: {exc}")
        return f"NO_IMAGE_FOUND: {description}"


@tool
def generate_quiz_questions(topic: str, difficulty: str = "intermediate", num_questions: int = 2) -> str:
    """Generate quiz questions prompt for the model."""
    return f"Generate {num_questions} {difficulty}-level questions about {topic} to test the user's understanding."


tools = [
    tavily_search,
    web_scraper_tool,
    get_geological_image,
    generate_quiz_questions,
]


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    presence_penalty=0.6,
    frequency_penalty=0.5,
    top_p=0.9,
).bind_tools(tools=tools)


def chatbot(state: State):
    """Main chatbot node."""
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as exc:
        logger.error("Error in chatbot node: %s", str(exc), exc_info=True)
        raise RuntimeError(f"Error processing your request: {str(exc)}") from exc


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=MemorySaver())


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

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

            elif kind == "on_tool_end":
                tool_name = event.get("name", "")
                if tool_name == "get_geological_image":
                    image_url = event["data"]["output"]
                    if hasattr(image_url, "content"):
                        image_url = image_url.content
                    image_url = str(image_url).strip()

                    if image_url.startswith("http"):
                        yield f"\n\n![Geological Image]({image_url})\n\n"
                    elif "NO_IMAGE_FOUND" in image_url:
                        logger.warning("Image fetch failed: %s", image_url)

    except Exception as exc:
        error_message = (
            f"\n\n❌ An error occurred: {str(exc)}\n\n"
            "Please try rephrasing your question or start a new chat."
        )
        logger.error("Error in run_agent for thread %s: %s", thread_id, str(exc), exc_info=True)
        yield error_message
