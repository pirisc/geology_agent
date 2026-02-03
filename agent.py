from typing import Annotated, TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from bs4 import BeautifulSoup 
import requests
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════
# SETUP & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# API KEYS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Configuration constants
MAX_INPUT_LENGTH = 10000  # Maximum characters for user input
WEB_SCRAPER_CHAR_LIMIT = 5000  # Increased from 2000
WEB_SCRAPER_TIMEOUT = 15  # Increased timeout

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
SCIENTIFIC APPROACH
-----------------------
- Base answers on established scientific consensus and evidence.
- Distinguish clearly between established knowledge, leading theories, and speculation.
- When discussing evolving topics, present multiple perspectives from the literature.
- Cite the type of evidence supporting claims (e.g., "radiometric dating shows...", 
  "seismic data indicates...", "field observations suggest...").
- If information is uncertain or outside your knowledge, say so explicitly.
- For ambiguous questions, state your assumptions or ask for clarification.
- If 'tavily_search' is used, ALWAYS provide the source of the information find

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

Also add a "Study Mode". If the user wants to be tested on something they learned, generate 
2 question based on the topic and wait for their answer. After the user anwsers the questions,
provide feedback and continue with this loop of questions and answers.


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


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════

# Tool 1: Web Search with Tavily
tavily_search = TavilySearchResults(
    max_results=5, 
    search_depth = "advanced",
    include_images = True,
    include_favicon = True
)

# Tool 2: Enhanced Web Scraper
@tool
def web_scraper_tool(url: str) -> str:
    """
    Scrapes a webpage and returns text contents from it.
    Useful for providing context to the agent, for when the user provides a URL 
    or to read a specific page.
    
    Args:
        url: The URL to scrape (must be a valid http/https URL)
    
    Returns:
        Cleaned text content from the webpage or error message
    """
    # Validate URL
    if not url.startswith(('http://', 'https://')):
        return f"Invalid URL: {url}. URL must start with http:// or https://"
    
    try:
        response = requests.get(
            url, 
            timeout=WEB_SCRAPER_TIMEOUT,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        # Return with character limit
        if len(clean_text) > WEB_SCRAPER_CHAR_LIMIT:
            return clean_text[:WEB_SCRAPER_CHAR_LIMIT] + f"\n\n[Content truncated - showed first {WEB_SCRAPER_CHAR_LIMIT} characters]"
        
        return clean_text

    except requests.Timeout:
        return f"Error: Request timed out after {WEB_SCRAPER_TIMEOUT} seconds for {url}"
    except requests.RequestException as e:
        return f"Error fetching {url}: {str(e)}"
 
# Tool 3: Create geological images
@tool
def create_geological_images(prompt:str):
    """ Generate a geological ilustration or diagam based on the users prompt.
    Useful for understanding geological concepts and visualizing structures.

    Args:
        prompt: users prompt about what they want to be create
    
    Returns: 
        an explicative image or ilustration about geological concepts.
    """

    client = OpenAI()
    response = client.images.generate(
        prompt = prompt,
        model = "dall-e-3", # best for science
        n = 1, # number of images
        size= "1024x1024",
        response_format= "url")

    return response.data[0].url

# Tool 4: Geneate quiz questions
@tool
def generate_quiz_questions(topic:str, difficulty: str = "intermediate") -> str:
    """ Generates a pair of 2 questions to help the user understand and learn a specific geological
    topic. 
    """
    return f"Generate 2 {difficulty} questions about {topic}"
    
# Tool list
tools = [
    tavily_search, 
    web_scraper_tool,
    create_geological_images,
    generate_quiz_questions
]

# ═══════════════════════════════════════════════════════════════════════════
# STATE & GRAPH
# ═══════════════════════════════════════════════════════════════════════════

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Build the graph
graph_builder = StateGraph(State)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    streaming=True
).bind_tools(tools=tools)

def chatbot(state: State):
    """Main chatbot node that processes messages."""
    try:
        return {"messages": [llm.invoke(state["messages"])]}
    except Exception as e:
        raise RuntimeError(f"Error in chatbot node: {str(e)}") from e
        

# Add nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# Add edges
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Compile graph
graph = graph_builder.compile(checkpointer=MemorySaver())

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def validate_input(user_input: str) -> tuple[bool, str]:
    """
    Validate user input.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not user_input:
        return False, "Error: No input provided"
    
    if not user_input.strip():
        return False, "Error: Input is empty or only whitespace"
    
    if len(user_input) > MAX_INPUT_LENGTH:
        return False, f"Error: Input exceeds maximum length of {MAX_INPUT_LENGTH} characters"
    
    return True, ""

async def run_agent(user_input: str, thread_id: str):
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
        yield error_msg
        return
    
    try:
        config = {"configurable": {"thread_id": thread_id}}

        # Retrieve the full conversation history from the checkpointer
        state = graph.get_state(config)
        history = state.values.get("messages", [])

        # Always build the full message list:
        messages = [("system", SYSTEM_PROMPT)] + history + [("user", user_input)]

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
                    yield content

            # Capture tool outputs (like image urls)
            elif kind == "on_tool_end":
                if event["name"] == "create_geological_images":
                    image_url = event["data"]["output"]
                    yield f"Image created: {image_url}"

    except Exception as e:
        error_message = f"\n\n❌ An error occurred: {str(e)}"
        yield error_message
