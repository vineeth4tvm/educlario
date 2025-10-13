import os
import json
import re
import google.generativeai as genai
from pathlib import Path

# --- Configuration ---
PROMPTS_DIR = Path(__file__).resolve().parent / 'prompts'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-1.5-pro-latest")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash-latest")

if not GEMINI_API_KEY:
    print("WARNING: AI_SERVICE - GEMINI_API_KEY is not set. AI functions will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize models to be used by the service functions
pro_model = genai.GenerativeModel(GEMINI_PRO_MODEL)
flash_model = genai.GenerativeModel(GEMINI_FLASH_MODEL)

# --- Helper Functions ---

def _load_prompt(prompt_name, **kwargs):
    """Loads a prompt from the prompts directory and formats it."""
    try:
        with open(PROMPTS_DIR / prompt_name, 'r') as f:
            prompt_template = f.read()
        return prompt_template.format(**kwargs)
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found: {prompt_name}")
        return None
    except KeyError as e:
        print(f"ERROR: Missing placeholder in prompt {prompt_name}: {e}")
        return None

def _clean_json_response(text):
    """Extracts a JSON object from a string, removing markdown code blocks."""
    match = re.search(r'```(json)?(.*)```', text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return text.strip()

# --- Service Functions for Stage 2 Logic ---

def get_user_context_summary(profile_data):
    """Generates a personalized learning context summary from user profile data."""
    prompt = _load_prompt(
        'generate_user_context.txt',
        academic_level=profile_data.get('academic_level'),
        interests=profile_data.get('interests'),
        learning_style=profile_data.get('learning_style'),
        location=profile_data.get('location'),
        explanation_style=profile_data.get('explanation_style')
    )
    if not prompt:
        return None
    response = flash_model.generate_content(prompt)
    return response.text

def get_book_overview_and_chapters(filepath, original_filename, user_context_text):
    """Generates chapter list and simplified overview for a book."""
    context_prompt_addition = ""
    if user_context_text:
        context_prompt_addition = f"""
        **IMPORTANT PERSONALIZATION INSTRUCTIONS:**
        You MUST tailor the simplified content based on the following user learning context:
        ---
        {user_context_text}
        ---
        """

    prompt = _load_prompt(
        'generate_book_overview.txt',
        original_filename=original_filename,
        context_prompt_addition=context_prompt_addition
    )
    if not prompt:
        return None, None

    uploaded_file = genai.upload_file(path=filepath, display_name=original_filename)
    response = pro_model.generate_content([prompt, uploaded_file])
    # Return the file handle so it can be reused for other calls
    return json.loads(_clean_json_response(response.text)), uploaded_file

def detect_subject_from_book(uploaded_file):
    """Analyzes a book to detect its primary academic subject."""
    prompt = _load_prompt('detect_book_subject.txt')
    if not prompt:
        return "General Studies" # Fallback subject

    # Use the flash model for this quick classification task
    response = flash_model.generate_content([prompt, uploaded_file])
    return response.text.strip()

# --- Service Functions for Stage 3 ---

def get_course_context(course_name, provider):
    """Generates a structured context for a given course."""
    prompt = _load_prompt(
        'generate_course_context.txt',
        course_name=course_name,
        provider=provider or "Not specified"
    )
    if not prompt:
        return None
    response = flash_model.generate_content(prompt)
    return json.loads(_clean_json_response(response.text))

def get_book_preface(uploaded_file):
    """Generates a book preface."""
    prompt = _load_prompt('generate_book_preface.txt')
    if not prompt:
        return None
    response = pro_model.generate_content([prompt, uploaded_file])
    return response.text

def get_assessment_for_chapter(chapter_content):
    """Generates a set of assessment questions for a given chapter's content."""
    prompt = _load_prompt(
        'generate_assessment.txt',
        chapter_content=chapter_content
    )
    if not prompt:
        return None

    # Use the Pro model as this is a complex generation task
    response = pro_model.generate_content(prompt)
    return json.loads(_clean_json_response(response.text))

def get_book_summary(uploaded_file):
    """Generates a book summary."""
    prompt = _load_prompt('generate_book_summary.txt')
    if not prompt:
        return None
    response = pro_model.generate_content([prompt, uploaded_file])
    return response.text

def get_book_context(uploaded_file):
    """Generates structured context (themes, structure) for a book."""
    prompt = _load_prompt('generate_book_context.txt')
    if not prompt:
        return None
    response = pro_model.generate_content([prompt, uploaded_file])
    return json.loads(_clean_json_response(response.text))

def get_deep_dive_content(chapter, book, user_context, course_context_db, book_context_db):
    """
    Generates rich, context-aware content for a chapter using a two-step
    "generate and refine" process to improve quality and accuracy.
    """
    filepath = os.path.join('uploads', book.filename)
    uploaded_file = genai.upload_file(path=filepath, display_name=book.original_name)

    # --- Step 1: Generate the initial deep dive ---
    user_context_prompt = f"USER CONTEXT: {user_context.generated_context_text if user_context else 'Not provided.'}"
    course_context_prompt = ""
    if course_context_db:
        cc = course_context_db
        course_context_prompt = f"COURSE CONTEXT: Subject: {cc.subject_analysis}. Prerequisites: {cc.prerequisites}. Real-world applications: {cc.real_world_apps}."
    book_context_prompt = ""
    if book_context_db:
        bc = book_context_db
        book_context_prompt = f"BOOK CONTEXT: Themes: {bc.themes}. Structure: {bc.structure}."

    initial_prompt = _load_prompt(
        'generate_deep_dive.txt',
        chapter_title=chapter.title,
        user_context_prompt=user_context_prompt,
        course_context_prompt=course_context_prompt,
        book_context_prompt=book_context_prompt
    )
    if not initial_prompt:
        raise Exception("Could not load the initial deep dive prompt.")

    generation_config = genai.types.GenerationConfig(max_output_tokens=8192)
    initial_response = pro_model.generate_content(
        [initial_prompt, uploaded_file],
        generation_config=generation_config
    )
    initial_ai_content = initial_response.text

    # --- Step 2: Extract the original text for the same chapter ---
    # This is a simpler, direct prompt to get the source material.
    text_extraction_prompt = f"From the provided PDF, extract and return only the raw, complete, and unedited text for the chapter titled '{chapter.title}'. Do not summarize, simplify, or format it in any way."
    original_text_response = pro_model.generate_content([text_extraction_prompt, uploaded_file])
    original_chapter_text = original_text_response.text

    # --- Step 3: Refine the content using the original text as a reference ---
    refinement_prompt = _load_prompt(
        'refine_deep_dive.txt',
        original_chapter_text=original_chapter_text,
        initial_ai_content=initial_ai_content
    )
    if not refinement_prompt:
        raise Exception("Could not load the refinement prompt.")

    final_response = pro_model.generate_content(refinement_prompt)

    return final_response.text