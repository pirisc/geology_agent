from typing import Annotated, TypedDict, AsyncGenerator, Any, Optional
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

MAX_INPUT_LENGTH = 10_000
WEB_SCRAPER_CHAR_LIMIT = 8_000
WEB_SCRAPER_TIMEOUT = 20
HTTP_TIMEOUT = 10
IMAGE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT - Updated comprehensive version
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
- If an image can't be found, I'll let you know and continue with the explanation

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
    if not isinstance(url, str) or not url.strip():
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

        clean_text = "\n".join(line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip())
        if len(clean_text) > WEB_SCRAPER_CHAR_LIMIT:
            return (
                f"{clean_text[:WEB_SCRAPER_CHAR_LIMIT]}\n\n"
                f"[Content truncated to {WEB_SCRAPER_CHAR_LIMIT} characters. Original length: {len(clean_text)} characters]"
            )
        return clean_text or "No readable content found on page"

    except requests.Timeout:
        return f"Error: Request timed out after {WEB_SCRAPER_TIMEOUT} seconds for {url}"
    except requests.RequestException as exc:
        return f"Error fetching {url}: {exc}"
    except Exception as exc:
        return f"Unexpected error processing {url}: {exc}"


@tool
def get_geological_image(description: str, image_type: str = "auto") -> str:
    """
    Fetch a geological image URL from reliable sources.
    
    Args:
        description: What to search for (e.g., "basalt thin section", "fault line diagram")
        image_type: Type of image - "photo", "diagram", or "auto" (default: auto-detects)
    
    Returns:
        Either a valid image URL, or a user-friendly error message explaining what happened
    """
    if image_type not in {"auto", "photo", "diagram"}:
        image_type = "auto"

    if image_type == "auto":
        image_type = "diagram" if _looks_like_diagram_request(description) else "photo"

    # Build search query
    query = f"{description} geology"
    if image_type == "diagram":
        query += " diagram schematic illustration"
    else:
        query += " photograph"

    # Try Wikimedia first (best for scientific content)
    wikimedia_url = _wikimedia_image_search(query)
    if wikimedia_url:
        logger.info(f"Found image on Wikimedia for: {description}")
        return wikimedia_url

    # Fallback to Tavily
    tavily_url = _tavily_image_search(query)
    if tavily_url:
        logger.info(f"Found image via Tavily for: {description}")
        return tavily_url

    # No image found - return helpful message
    logger.warning(f"No image found for: {description}")
    return f"IMAGE_NOT_FOUND: I couldn't find a suitable {image_type} for '{description}'. I'll continue explaining without the image."


def _looks_like_diagram_request(description: str) -> bool:
    """Check if the description suggests a diagram/illustration vs a photo."""
    keywords = {
        "diagram", "cross-section", "cross section", "plate", "tectonic", "cycle", "process",
        "structure", "layers", "boundary", "subduction", "illustration", "schematic", "model",
        "formation", "crystal structure", "fault", "fold", "stratigraphic", "earth interior",
        "mantle", "crust", "core", "convection", "mechanism", "how",
    }
    desc = description.lower()
    return any(word in desc for word in keywords)


def _wikimedia_image_search(query: str) -> Optional[str]:
    """Search Wikimedia Commons for geological images."""
    try:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrsearch": query,
                "gsrlimit": "8",
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": "1200",
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        pages = (response.json().get("query") or {}).get("pages") or {}

        for page in pages.values():
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            mime = (info.get("mime") or "").lower()
            if not mime.startswith("image/"):
                continue

            image_url = info.get("thumburl") or info.get("url")
            if isinstance(image_url, str) and IMAGE_URL_RE.match(image_url):
                return image_url.strip()
    except Exception as exc:
        logger.warning("Wikimedia search failed: %s", exc)

    return None


def _tavily_image_search(query: str) -> Optional[str]:
    """Search for images using Tavily as fallback."""
    try:
        payload = tavily_search.invoke({"query": query})
        candidates: list[str] = []

        if isinstance(payload, dict):
            if isinstance(payload.get("images"), list):
                candidates.extend([x for x in payload["images"] if isinstance(x, str)])
            if isinstance(payload.get("results"), list):
                for item in payload["results"]:
                    if isinstance(item, dict) and isinstance(item.get("images"), list):
                        candidates.extend([x for x in item["images"] if isinstance(x, str)])
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    candidates.append(item)
                elif isinstance(item, dict) and isinstance(item.get("images"), list):
                    candidates.extend([x for x in item["images"] if isinstance(x, str)])

        for url in candidates:
            if IMAGE_URL_RE.match(url):
                return url.strip()
    except Exception as exc:
        logger.warning("Tavily image search failed: %s", exc)

    return None


@tool
def start_quiz_mode(topic: str, difficulty: str = "intermediate", num_questions: int = 3) -> str:
    """
    Start an interactive geology quiz on a specific topic.
    
    Args:
        topic: The geological topic to quiz on (e.g., "plate tectonics", "igneous rocks", "minerals")
        difficulty: Difficulty level - "easy", "intermediate", or "advanced" (default: intermediate)
        num_questions: How many questions to generate, 1-5 (default: 3)
    
    Returns:
        Instructions telling the AI to generate quiz questions and wait for answers
    """
    # Simple validation
    if num_questions < 1:
        num_questions = 1
    elif num_questions > 5:
        num_questions = 5
    
    difficulty = difficulty.lower()
    if difficulty not in ["easy", "intermediate", "advanced"]:
        difficulty = "intermediate"
    
    return (
        f"🎯 QUIZ MODE ACTIVATED\n\n"
        f"Generate {num_questions} {difficulty}-level multiple-choice questions about '{topic}'.\n\n"
        f"Format:\n"
        f"- Number each question clearly\n"
        f"- Provide 4 options (A, B, C, D) for each\n"
        f"- Make questions test understanding, not just memorization\n"
        f"- After presenting all questions, WAIT for the user to answer\n"
        f"- Then provide feedback and explanations for each answer\n\n"
        f"End with: 'Take your time! Reply with your answers (e.g., 1-A, 2-C, 3-B) when ready.'"
    )


# Register all tools
tools = [tavily_search, web_scraper_tool, get_geological_image, start_quiz_mode]


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
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=MemorySaver())


def validate_input(user_input: str) -> tuple[bool, str]:
    """Simple validation to catch empty or too-long inputs."""
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
    Main function that runs the Rocky agent and streams responses.
    
    This handles:
    - Input validation
    - Streaming chat responses from the LLM
    - Displaying images when the image tool is called
    - Gracefully handling errors
    """
    is_valid, error_msg = validate_input(user_input)
    if not is_valid:
        yield error_msg
        return

    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        messages = [("user", user_input)] if state.values.get("messages") else [("system", SYSTEM_PROMPT), ("user", user_input)]

        async for event in graph.astream_events({"messages": messages}, config=config, version="v2"):
            kind = event.get("event")

            # Stream text from the LLM
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

            # Handle image tool results
            elif kind == "on_tool_end" and event.get("name") == "get_geological_image":
                image_result = str(getattr(event["data"].get("output"), "content", event["data"].get("output"))).strip()
                
                # Check if we got a real image URL
                if IMAGE_URL_RE.match(image_result):
                    # Display the image in markdown
                    yield f"\n\n![Geological Image]({image_result})\n\n"
                
                # Check if image search failed
                elif "IMAGE_NOT_FOUND" in image_result:
                    # Extract the friendly message and show it to user
                    message = image_result.replace("IMAGE_NOT_FOUND: ", "")
                    yield f"\n\n_({message})_\n\n"

    except Exception as exc:
        logger.error("Error in run_agent for thread %s: %s", thread_id, exc, exc_info=True)
        yield f"\n\n❌ An error occurred: {exc}\n\nPlease try rephrasing your question or start a new chat."
