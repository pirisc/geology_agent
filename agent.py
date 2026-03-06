from typing import Annotated, TypedDict, AsyncGenerator, Any, Optional
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
# SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

load_dotenv(find_dotenv(), override=True)

MAX_INPUT_LENGTH = 10000
WEB_SCRAPER_CHAR_LIMIT = 8000
WEB_SCRAPER_TIMEOUT = 20
IMAGE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

SYSTEM_PROMPT = """
You are Rocky, an AI assistant specializing in Geology and Earth Sciences.
Be accurate, clear, and educational.
"""

# Tool 1: Web Search with Tavily
# NOTE: include_images=True is required for image fallback extraction.
tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_images=True,
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
def get_geological_image(description: str, image_type: str = "auto") -> str:
    """
    Fetch geological images (photos and diagrams).

    image_type: "photo", "diagram", or "auto"
    """
    try:
        if image_type == "auto":
            diagram_keywords = [
                "diagram", "cross-section", "cross section", "plate", "tectonic",
                "cycle", "process", "structure of", "layers", "boundary", "subduction",
                "illustration", "schematic", "model", "system", "formation process",
                "crystal structure", "fault", "fold", "stratigraphic", "earth interior",
                "mantle", "crust", "core", "convection", "how", "mechanism",
            ]
            description_lower = description.lower()
            image_type = "diagram" if any(k in description_lower for k in diagram_keywords) else "photo"

        if image_type == "diagram":
            return fetch_geological_diagram(description)
        return fetch_geological_photo(description)

    except Exception as exc:
        logger.error("Error in get_geological_image: %s", exc)
        return f"IMAGE_ERROR: {str(exc)}"


def fetch_geological_photo(description: str) -> str:
    """Fetch real photographs from Unsplash; fallback to diagram search."""
    try:
        unsplash_key = os.getenv("UNSPLASH_ACCESS_KEY")
        if not unsplash_key:
            logger.warning("UNSPLASH_ACCESS_KEY not found, using diagram/web fallback")
            return fetch_geological_diagram(description)

        response = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": f"{description} geology rock mineral",
                "client_id": unsplash_key,
                "per_page": 3,
                "orientation": "landscape",
                "content_filter": "high",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results") or []
        if results:
            return results[0]["urls"]["regular"]

        return fetch_geological_diagram(description)

    except Exception as exc:
        logger.error("Unsplash error: %s", exc)
        return fetch_geological_diagram(description)


def fetch_geological_diagram(description: str) -> str:
    """Fetch geological diagrams from Wikimedia, then Tavily fallback."""
    try:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrsearch": f"{description} geology diagram",
                "gsrlimit": "5",
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": "1000",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            imageinfo = page.get("imageinfo") or []
            if not imageinfo:
                continue
            info = imageinfo[0]
            image_url = info.get("thumburl") or info.get("url")
            if isinstance(image_url, str) and IMAGE_URL_RE.match(image_url):
                return image_url

        return search_geological_image_fallback(description, prefer_diagrams=True)

    except Exception as exc:
        logger.error("Wikimedia error: %s", exc)
        return search_geological_image_fallback(description, prefer_diagrams=True)


def _first_valid_image_url(payload: Any) -> Optional[str]:
    """Extract first plausible image URL from Tavily output (dict or list)."""
    candidates: list[str] = []

    if isinstance(payload, dict):
        top_images = payload.get("images")
        if isinstance(top_images, list):
            candidates.extend([x for x in top_images if isinstance(x, str)])

        results = payload.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    imgs = item.get("images")
                    if isinstance(imgs, list):
                        candidates.extend([x for x in imgs if isinstance(x, str)])

    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                imgs = item.get("images")
                if isinstance(imgs, list):
                    candidates.extend([x for x in imgs if isinstance(x, str)])
            elif isinstance(item, str):
                candidates.append(item)

    for url in candidates:
        if IMAGE_URL_RE.match(url):
            return url.strip()
    return None


def search_geological_image_fallback(description: str, prefer_diagrams: bool = False) -> str:
    """Fallback image search using Tavily with robust response parsing."""
    try:
        if prefer_diagrams:
            search_query = f"{description} diagram illustration geology educational"
        else:
            search_query = f"{description} geology photo high resolution"

        # Use explicit object payload for consistency.
        results = tavily_search.invoke({"query": search_query})
        image_url = _first_valid_image_url(results)
        if image_url:
            return image_url

        return f"NO_IMAGE_FOUND: {description}"

    except Exception as exc:
        logger.error("Fallback search error: %s", exc)
        return f"IMAGE_ERROR: {str(exc)}"


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

                    if IMAGE_URL_RE.match(image_url):
                        yield f"\n\n![Geological Image]({image_url})\n\n"
                    elif "NO_IMAGE_FOUND" in image_url or "IMAGE_ERROR" in image_url:
                        logger.warning("Image fetch failed: %s", image_url)

    except Exception as exc:
        error_message = (
            f"\n\n❌ An error occurred: {str(exc)}\n\n"
            "Please try rephrasing your question or start a new chat."
        )
        logger.error("Error in run_agent for thread %s: %s", thread_id, str(exc), exc_info=True)
        yield error_message
