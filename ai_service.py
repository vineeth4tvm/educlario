import os
import json
import re
import time
import functools
import google.generativeai as genai
from google.api_core import exceptions as core_exceptions
from pathlib import Path
from pdf_utils import trim_pdf, cleanup_temp_file, extract_text_from_pdf

# --- Configuration ---
PROMPTS_DIR = Path(__file__).resolve().parent / 'prompts'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-1.5-pro-latest")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash-latest")

if not GEMINI_API_KEY:
    print("WARNING: AI_SERVICE - GEMINI_API_KEY is not set. AI functions will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize models
pro_model = genai.GenerativeModel(GEMINI_PRO_MODEL)
flash_model = genai.GenerativeModel(GEMINI_FLASH_MODEL)

# --- Helper Functions ---
def _load_prompt(prompt_name, **kwargs):
    try:
        with open(PROMPTS_DIR / prompt_name, 'r') as f:
            return f.read().format(**kwargs)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading prompt {prompt_name}: {e}")
        return None

def _clean_json_response(text):
    match = re.search(r'```(json)?(.*)```', text, re.DOTALL)
    return match.group(2).strip() if match else text.strip()

def retry_on_rate_limit(max_retries=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except core_exceptions.ResourceExhausted as e:
                    retries += 1
                    if retries >= max_retries: raise e
                    delay_match = re.search(r'retry_delay {\s*seconds: (\d+)\s*}', str(e))
                    wait_time = int(delay_match.group(1)) + 1 if delay_match else (2 ** retries)
                    print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

def _create_pdf_part(filepath):
    return {'mime_type': 'application/pdf', 'data': Path(filepath).read_bytes()}

# --- Main Service Functions ---
@retry_on_rate_limit()
def get_user_context_summary(profile_data):
    prompt = _load_prompt('generate_user_context.txt', **profile_data)
    if not prompt: return None
    response = flash_model.generate_content(prompt)
    return response.text

@retry_on_rate_limit()
def get_book_chapter_list(filepath, original_filename):
    prompt = _load_prompt('get_chapter_list.txt', original_filename=original_filename)
    if not prompt: return None
    pdf_part = _create_pdf_part(filepath)
    response = pro_model.generate_content([prompt, pdf_part])
    return json.loads(_clean_json_response(response.text))

@retry_on_rate_limit()
def get_overview_for_trimmed_chapter(chapter, book, user_context, all_chapter_titles, chart_id):
    trimmed_filepath = None
    try:
        if not chapter.page_range:
            raise Exception("Cannot generate overview without a page range.")

        original_filepath = os.path.join('uploads', book.filename)
        trimmed_filepath = trim_pdf(original_filepath, chapter.page_range)
        if not trimmed_filepath:
            raise Exception("Failed to trim PDF for overview generation.")

        user_context_text = user_context.generated_context_text if user_context else ""
        context_prompt_addition = f"For context, the book's chapters are: {', '.join(all_chapter_titles)}."
        if user_context_text:
            context_prompt_addition += f"\n\n**USER CONTEXT:**\n{user_context_text}"

        prompt = _load_prompt('generate_chapter_overview.txt', chapter_title=chapter.title, context_prompt_addition=context_prompt_addition, chart_id=chart_id)
        if not prompt: return None

        pdf_part = _create_pdf_part(trimmed_filepath)
        response = pro_model.generate_content([prompt, pdf_part])
        return response.text
    finally:
        if trimmed_filepath:
            cleanup_temp_file(trimmed_filepath)

@retry_on_rate_limit()
def detect_subject_from_book(filepath):
    prompt = _load_prompt('detect_book_subject.txt')
    if not prompt: return "General Studies"
    pdf_part = _create_pdf_part(filepath)
    response = flash_model.generate_content([prompt, pdf_part])
    return response.text.strip()

@retry_on_rate_limit()
def get_course_context(course_name, provider):
    prompt = _load_prompt('generate_course_context.txt', course_name=course_name, provider=provider or "Not specified")
    if not prompt: return None
    response = flash_model.generate_content(prompt)
    return json.loads(_clean_json_response(response.text))

@retry_on_rate_limit()
def get_book_preface(filepath):
    prompt = _load_prompt('generate_book_preface.txt')
    if not prompt: return None
    pdf_part = _create_pdf_part(filepath)
    response = pro_model.generate_content([prompt, pdf_part])
    return response.text

@retry_on_rate_limit()
def get_book_summary(filepath):
    prompt = _load_prompt('generate_book_summary.txt')
    if not prompt: return None
    pdf_part = _create_pdf_part(filepath)
    response = pro_model.generate_content([prompt, pdf_part])
    return response.text

@retry_on_rate_limit()
def get_book_context(filepath):
    prompt = _load_prompt('generate_book_context.txt')
    if not prompt: return None
    pdf_part = _create_pdf_part(filepath)
    response = pro_model.generate_content([prompt, pdf_part])
    return json.loads(_clean_json_response(response.text))

def _check_recitation(response):
    """Checks if the model's response was blocked for recitation."""
    if not response.parts and response.candidates[0].finish_reason.name == "RECITATION":
        raise Exception("Content generation failed due to the model's safety filters detecting potential recitation from copyrighted material. Please try a different chapter or book.")

@retry_on_rate_limit()
def get_deep_dive_content(chapter, book, user_context, course_context_db, book_context_db, chart_id):
    original_filepath = os.path.join('uploads', book.filename)
    trimmed_filepath = None
    try:
        if not chapter.page_range: raise Exception("Cannot perform deep dive without a page range.")
        trimmed_filepath = trim_pdf(original_filepath, chapter.page_range)
        if not trimmed_filepath: raise Exception("Failed to trim PDF for deep dive.")

        original_chapter_text = extract_text_from_pdf(trimmed_filepath)
        if not original_chapter_text:
            raise Exception("Could not extract text from the chapter PDF.")

        pdf_part = _create_pdf_part(trimmed_filepath)
        user_context_prompt = f"USER CONTEXT: {user_context.generated_context_text if user_context else 'Not provided.'}"
        course_context_prompt = f"COURSE CONTEXT: Subject: {course_context_db.subject_analysis if course_context_db else 'N/A'}."
        book_context_prompt = f"BOOK CONTEXT: Themes: {book_context_db.themes if book_context_db else 'N/A'}."
        initial_prompt = _load_prompt('generate_deep_dive.txt', chapter_title=chapter.title, user_context_prompt=user_context_prompt, course_context_prompt=course_context_prompt, book_context_prompt=book_context_prompt, chart_id=chart_id)
        if not initial_prompt: raise Exception("Could not load deep dive prompt.")

        generation_config = genai.types.GenerationConfig(max_output_tokens=8192)
        initial_response = pro_model.generate_content([initial_prompt, pdf_part], generation_config=generation_config)
        _check_recitation(initial_response)
        initial_ai_content = initial_response.text

        refinement_prompt = _load_prompt('refine_deep_dive.txt', original_chapter_text=original_chapter_text, initial_ai_content=initial_ai_content)
        if not refinement_prompt: raise Exception("Could not load refinement prompt.")

        final_response = pro_model.generate_content(refinement_prompt)
        _check_recitation(final_response)
        return final_response.text
    finally:
        if trimmed_filepath: cleanup_temp_file(trimmed_filepath)

@retry_on_rate_limit()
def get_assessment_for_chapter(chapter, book):
    trimmed_filepath = None
    try:
        if not chapter.page_range: raise Exception("Cannot generate assessment without a page range.")
        original_filepath = os.path.join('uploads', book.filename)
        trimmed_filepath = trim_pdf(original_filepath, chapter.page_range)
        if not trimmed_filepath: raise Exception("Failed to trim PDF for assessment generation.")

        content_to_assess = extract_text_from_pdf(trimmed_filepath)
        if not content_to_assess:
            raise Exception("Could not extract text from the chapter PDF for assessment.")

        prompt = _load_prompt('generate_assessment.txt', chapter_content=content_to_assess)
        if not prompt: return None

        response = pro_model.generate_content(prompt)
        _check_recitation(response)
        return json.loads(_clean_json_response(response.text))
    finally:
        if trimmed_filepath: cleanup_temp_file(trimmed_filepath)

@retry_on_rate_limit()
def get_flashcards_for_chapter(chapter, book, user_context):
    trimmed_filepath = None
    try:
        if not chapter.page_range: raise Exception("Cannot generate flashcards without a page range.")
        original_filepath = os.path.join('uploads', book.filename)
        trimmed_filepath = trim_pdf(original_filepath, chapter.page_range)
        if not trimmed_filepath: raise Exception("Failed to trim PDF for flashcard generation.")

        pdf_part = _create_pdf_part(trimmed_filepath)
        user_context_text = user_context.generated_context_text if user_context else ""
        all_chapter_titles = [c.title for c in book.chapters]
        context_prompt_addition = f"For context, the book's chapters are: {', '.join(all_chapter_titles)}."
        if user_context_text:
            context_prompt_addition += f"\n\n**USER CONTEXT:**\n{user_context_text}"

        prompt = _load_prompt('generate_flashcards.txt', chapter_title=chapter.title, context_prompt_addition=context_prompt_addition)
        if not prompt: return None

        response = pro_model.generate_content([prompt, pdf_part])
        return json.loads(_clean_json_response(response.text))
    finally:
        if trimmed_filepath: cleanup_temp_file(trimmed_filepath)

@retry_on_rate_limit()
def get_mind_map_for_chapter(chapter, book, user_context):
    trimmed_filepath = None
    try:
        if not chapter.page_range: raise Exception("Cannot generate mind map without a page range.")
        original_filepath = os.path.join('uploads', book.filename)
        trimmed_filepath = trim_pdf(original_filepath, chapter.page_range)
        if not trimmed_filepath: raise Exception("Failed to trim PDF for mind map generation.")

        pdf_part = _create_pdf_part(trimmed_filepath)
        user_context_text = user_context.generated_context_text if user_context else ""
        all_chapter_titles = [c.title for c in book.chapters]
        context_prompt_addition = f"For context, the book's chapters are: {', '.join(all_chapter_titles)}."
        if user_context_text:
            context_prompt_addition += f"\n\n**USER CONTEXT:**\n{user_context_text}"

        prompt = _load_prompt('generate_mind_map.txt', chapter_title=chapter.title, context_prompt_addition=context_prompt_addition)
        if not prompt: return None

        response = pro_model.generate_content([prompt, pdf_part])
        return json.loads(_clean_json_response(response.text))
    finally:
        if trimmed_filepath: cleanup_temp_file(trimmed_filepath)