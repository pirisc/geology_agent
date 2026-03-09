from __future__ import annotations

import logging
from typing import Annotated, AsyncGenerator, TypedDict
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
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

MAX_INPUT_LENGTH = 10_000
WEB_SCRAPER_CHAR_LIMIT = 8_000
WEB_SCRAPER_TIMEOUT = 20

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
PROVIDING IMAGE RESOURCES
-----------------------
You have a tool called 'find_geological_images' that provides links to trusted geology image sources:

**How to use it:**
- Use it when visual examples would help (rocks, minerals, diagrams, processes)
- Be specific in your description: "basalt thin section" not just "rock"
- The tool automatically picks the best resource (Mindat for minerals, USGS for diagrams, etc.)

**Example usage:**
- User asks about granite → explain, then mention: "Let me point you to some images of granite"
- User asks about plate boundaries → explain, then: "I can show you where to find diagrams of this"

**What it provides:**
- Direct links to professional geology databases
- High-quality, scientifically accurate images
- Resources maintained by geologists and institutions

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
""".strip()


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


def _safe_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
        element.decompose()

    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


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

        clean_text = _safe_text_from_html(response.text)
        if not clean_text:
            return "No readable content found on page"

        if len(clean_text) > WEB_SCRAPER_CHAR_LIMIT:
            truncated = clean_text[:WEB_SCRAPER_CHAR_LIMIT]
            return (
                f"{truncated}\n\n"
                f"[Content truncated to {WEB_SCRAPER_CHAR_LIMIT} characters. "
                f"Original length: {len(clean_text)} characters]"
            )

        return clean_text

    except requests.Timeout:
        return f"Error: Request timed out after {WEB_SCRAPER_TIMEOUT} seconds for {url}"
    except requests.RequestException as exc:
        return f"Error fetching {url}: {exc}"
    except Exception as exc:  # defensive catch for parser edge-cases
        return f"Unexpected error processing {url}: {exc}"


@tool
def find_geological_images(topic: str) -> str:
    """Provide a Google Images search link for geological topics."""
    if not isinstance(topic, str) or not topic.strip():
        return "Error: Topic must be a non-empty string"

    topic = topic.strip()
    topic_lower = topic.lower()

    if any(k in topic_lower for k in ["mineral", "crystal", "gem"]):
        search_query = f"{topic} mineral geology specimen"
        context = "You'll find photos of actual mineral specimens and crystals."
    elif any(
        word in topic_lower
        for word in [
            "diagram",
            "process",
            "cycle",
            "cross-section",
            "cross section",
            "plate",
            "boundary",
            "structure",
            "how",
        ]
    ):
        search_query = f"{topic} geology diagram illustration"
        context = "You'll find educational diagrams and illustrations."
    elif any(word in topic_lower for word in ["rock", "outcrop", "formation", "sample", "stone"]):
        search_query = f"{topic} rock geology sample"
        context = "You'll find photos of real rock samples and formations."
    else:
        search_query = f"{topic} geology"
        context = "You'll find a variety of relevant geological images."

    google_images_url = f"https://www.google.com/search?q={quote_plus(search_query)}&tbm=isch"
    return (
        f"📸 **Search Google Images for '{topic}'**\n\n"
        f"{google_images_url}\n\n"
        f"_{context} Click the link to see high-quality images._"
    )


@tool
def start_quiz_mode(topic: str, difficulty: str = "intermediate", num_questions: int = 3) -> str:
    """Start an interactive geology quiz on a specific topic."""
    num_questions = min(max(num_questions, 1), 5)

    difficulty = difficulty.lower().strip()
    if difficulty not in ["easy", "intermediate", "advanced"]:
        difficulty = "intermediate"

    return (
        "🎯 QUIZ MODE ACTIVATED\n\n"
        f"Generate {num_questions} {difficulty}-level multiple-choice questions about '{topic}'.\n\n"
        "Format:\n"
        "- Number each question clearly\n"
        "- Provide 4 options (A, B, C, D) for each\n"
        "- Make questions test understanding, not just memorization\n"
        "- After presenting all questions, WAIT for the user to answer\n"
        "- Then provide feedback and explanations for each answer\n\n"
        "End with: 'Take your time! Reply with your answers (e.g., 1-A, 2-C, 3-B) when ready.'"
    )


tools = [tavily_search, web_scraper_tool, find_geological_images, start_quiz_mode]


# ═══════════════════════════════════════════════════════════════════════════
# LANGGRAPH SETUP
# ═══════════════════════════════════════════════════════════════════════════

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
        logger.error("Error in chatbot node: %s", exc, exc_info=True)
        raise RuntimeError(f"Error processing your request: {exc}") from exc


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════════════
# MAIN AGENT FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def validate_input(user_input: str) -> tuple[bool, str]:
    """Validate user input."""
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

        messages = [("user", user_input)]
        if is_new_conversation:
            messages.insert(0, ("system", SYSTEM_PROMPT))

        async for event in graph.astream_events({"messages": messages}, config=config, version="v2"):
            if event.get("event") == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield content

    except Exception as exc:
        error_message = (
            f"\n\n❌ An error occurred: {exc}\n\n"
            "Please try rephrasing your question or start a new chat."
        )
        logger.error("Error in run_agent for thread %s: %s", thread_id, exc, exc_info=True)
        yield error_message
