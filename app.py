import os
import json
import re
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from dotenv import load_dotenv
from datetime import datetime

from models import db, User, Book, Chapter, GeneratedContent

# --- App Initialization ---
app = Flask(__name__)

# --- Configuration ---
load_dotenv()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_PRO_MODEL = os.getenv("GEMINI_PRO_MODEL", "gemini-1.5-pro-latest")
GEMINI_FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-1.5-flash-latest")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Content generation will fail.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Database and Extensions ---
db.init_app(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Create Uploads Directory ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Routes ---
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists.')
            return redirect(url_for('signup'))
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash('Please check your login details and try again.')
            return redirect(url_for('login'))
        login_user(user, remember=remember)
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    courses = Course.query.filter_by(user_id=current_user.id).order_by(Course.created_at.desc()).all()
    # Eagerly load books for each course to avoid separate queries in the template
    books_by_course = {course.id: course.books for course in courses}

    # Also get books that are not assigned to any course
    standalone_books = Book.query.filter_by(user_id=current_user.id, course_id=None).order_by(Book.uploaded_at.desc()).all()

    return render_template('dashboard.html', name=current_user.email, courses=courses, books_by_course=books_by_course, standalone_books=standalone_books)

@app.route('/profile')
@login_required
def profile():
    user_context = UserContext.query.filter_by(user_id=current_user.id).first()
    return render_template('profile.html', user_context=user_context)

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user_context = UserContext.query.filter_by(user_id=current_user.id).first()

    if request.method == 'POST':
        # Get data from form
        academic_level = request.form.get('academic_level')
        interests = request.form.get('interests')
        learning_style = request.form.get('learning_style')
        location = request.form.get('location')
        explanation_style = request.form.get('explanation_style')

        if not user_context:
            user_context = UserContext(user_id=current_user.id)
            db.session.add(user_context)

        user_context.academic_level = academic_level
        user_context.interests = interests
        user_context.learning_style = learning_style
        user_context.location = location
        user_context.explanation_style = explanation_style

        # --- AI Context Generation ---
        try:
            model = genai.GenerativeModel(GEMINI_FLASH_MODEL)
            prompt = f"""
            Based on the following user profile, generate a concise, one-paragraph summary of their personalized learning context. This summary will guide the content generation AI.

            - Academic Level: {academic_level}
            - Subject Interests: {interests}
            - Preferred Learning Style: {learning_style}
            - Location for Context: {location}
            - Preferred Explanation Style: {explanation_style}

            Example Output: "The user is an intermediate learner from India, interested in engineering. They prefer content with practical examples and a conversational tone."

            Generate the summary now:
            """
            response = model.generate_content(prompt)
            user_context.generated_context_text = response.text
            flash('Successfully generated your personalized learning context.')
        except Exception as e:
            flash(f'Could not generate AI context: {e}')

        db.session.commit()
        flash('Your profile has been updated.')
        return redirect(url_for('profile'))

    return render_template('edit_profile.html', user_context=user_context)

@app.route('/create_course', methods=['GET', 'POST'])
@login_required
def create_course():
    if request.method == 'POST':
        name = request.form.get('name')
        provider = request.form.get('provider')
        if not name:
            flash('Course name is required.')
        else:
            new_course = Course(user_id=current_user.id, name=name, provider=provider)
            db.session.add(new_course)
            db.session.commit()
            flash(f'Course "{name}" has been created successfully.')
            return redirect(url_for('dashboard'))
    return render_template('create_course.html')

def clean_json_from_response(text):
    """Extracts a JSON object from a string, removing markdown code blocks."""
    match = re.search(r'```(json)?(.*)```', text, re.DOTALL)
    if match:
        return match.group(2).strip()
    return text.strip()

@app.route('/upload_book', methods=['POST'])
@login_required
def upload_book():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('dashboard'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('dashboard'))

    if file and file.filename.endswith('.pdf'):
        original_filename = secure_filename(file.filename)
        filename = f"{current_user.id}_{int(datetime.now().timestamp())}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get course_id from form
        course_id = request.form.get('course_id')
        if course_id:
            course_id = int(course_id)
            # Security check: ensure the course belongs to the current user
            course = Course.query.filter_by(id=course_id, user_id=current_user.id).first()
            if not course:
                flash("Invalid course selected.")
                return redirect(url_for('dashboard'))
        else:
            course_id = None

        # Create Book record first
        new_book = Book(
            user_id=current_user.id,
            course_id=course_id,
            filename=filename,
            original_name=original_filename
        )
        db.session.add(new_book)
        db.session.commit()
        flash(f'Book "{original_filename}" uploaded. Processing with AI...')

        try:
            # Fetch the user's learning context
            user_context = UserContext.query.filter_by(user_id=current_user.id).first()
            context_prompt_addition = ""
            if user_context and user_context.generated_context_text:
                context_prompt_addition = f"""
                **IMPORTANT PERSONALIZATION INSTRUCTIONS:**
                You MUST tailor the simplified content based on the following user learning context:
                ---
                {user_context.generated_context_text}
                ---
                For example, if the user is from India and interested in engineering, use relevant Indian examples and engineering applications in your explanations.
                """

            # Upload the file to Gemini
            uploaded_file = genai.upload_file(path=filepath, display_name=original_filename)

            # Create the prompt for the model
            prompt = f"""
            You are an expert in educational content creation. Analyze the provided PDF file, which is a textbook titled "{original_filename}".

            Your task is to perform the following actions and return the result as a single, valid JSON object:

            1.  **Identify Chapters**: Scan the entire document and identify all the main chapters or sections. Extract the chapter number and the exact title for each one.
            2.  **Simplify Content**: For each chapter you identify, read its content and generate a simplified explanation suitable for a student. The explanation should be formatted in simple, clean HTML (e.g., using `<p>`, `<h1>`, `<h2>`, `<ul>`, `<li>`, `<strong>`).

            {context_prompt_addition}

            The final JSON output should follow this exact structure:

            {{
              "book_title": "The Title of the Book",
              "chapters": [
                {{
                  "chapter_number": 1,
                  "title": "Introduction to Subject",
                  "simplified_html_content": "<h1>Introduction to Subject</h1><p>This is the simplified content for the first chapter...</p>"
                }}
              ]
            }}

            Ensure the `simplified_html_content` is a single string containing valid HTML. Do not include any text or markdown formatting outside of the main JSON object.
            """

            # Call the model
            model = genai.GenerativeModel(GEMINI_PRO_MODEL)
            response = model.generate_content([prompt, uploaded_file])

            # Clean and parse the JSON response
            cleaned_json_str = clean_json_from_response(response.text)
            ai_data = json.loads(cleaned_json_str)

            # Populate chapters and content from AI response
            for chapter_data in ai_data.get('chapters', []):
                new_chapter = Chapter(
                    book_id=new_book.id,
                    chapter_number=chapter_data.get('chapter_number'),
                    title=chapter_data.get('title', f"Chapter {chapter_data.get('chapter_number')}")
                )
                db.session.add(new_chapter)
                db.session.commit() # Commit to get the chapter ID

                new_content = GeneratedContent(
                    chapter_id=new_chapter.id,
                    html_content=chapter_data.get('simplified_html_content', '<p>Content not available.</p>')
                )
                db.session.add(new_content)

            db.session.commit()
            flash('AI processing complete. Your book is ready!')

        except Exception as e:
            db.session.rollback() # Rollback changes for this book
            db.session.delete(new_book)
            db.session.commit()
            flash(f'An error occurred during AI processing: {e}')

        return redirect(url_for('dashboard'))

    else:
        flash('Only PDF files are allowed.')
        return redirect(url_for('dashboard'))

@app.route('/book/<int:book_id>')
@login_required
def book_details(book_id):
    book = Book.query.get_or_404(book_id)
    if book.user_id != current_user.id:
        flash("You do not have permission to view this book.")
        return redirect(url_for('dashboard'))

    chapters = Chapter.query.filter_by(book_id=book.id).order_by(Chapter.chapter_number).all()
    return render_template('book_details.html', book=book, chapters=chapters)

@app.route('/chapter/<int:chapter_id>')
@login_required
def chapter_view(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    # Security check: ensure the chapter belongs to a book owned by the current user
    if chapter.book.user_id != current_user.id:
        flash("You do not have permission to view this chapter.")
        return redirect(url_for('dashboard'))

    content = GeneratedContent.query.filter_by(chapter_id=chapter.id).first()
    return render_template('chapter_view.html', chapter=chapter, content=content)

@app.route('/regenerate_content/<int:chapter_id>', methods=['POST'])
@login_required
def regenerate_content(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    # Security check
    if chapter.book.user_id != current_user.id:
        flash("You do not have permission to modify this content.")
        return redirect(url_for('dashboard'))

    try:
        # Fetch user context
        user_context = UserContext.query.filter_by(user_id=current_user.id).first()
        context_prompt_addition = ""
        if user_context and user_context.generated_context_text:
            context_prompt_addition = f"""
            **IMPORTANT PERSONALIZATION INSTRUCTIONS:**
            You MUST tailor the simplified content based on the following user learning context:
            ---
            {user_context.generated_context_text}
            ---
            """

        # Get the file and upload it
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], chapter.book.filename)
        uploaded_file = genai.upload_file(path=filepath, display_name=chapter.book.original_name)

        # Create a more targeted prompt for single-chapter regeneration
        prompt = f"""
        You are an expert in educational content creation. From the provided PDF file, focus ONLY on the chapter titled "{chapter.title}".

        Your task is to generate a new, simplified explanation for this specific chapter. The explanation should be formatted in simple, clean HTML.

        {context_prompt_addition}

        Return only the raw HTML content for this chapter, with no other text, explanation, or markdown.
        """

        # Call the model
        model = genai.GenerativeModel(GEMINI_PRO_MODEL)
        response = model.generate_content([prompt, uploaded_file])

        # Update the existing content record or create a new one
        content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first()
        if content_record:
            content_record.html_content = response.text
            flash(f'Content for "{chapter.title}" has been regenerated successfully.')
        else:
            content_record = GeneratedContent(chapter_id=chapter.id, html_content=response.text)
            db.session.add(content_record)
            flash(f'Content for "{chapter.title}" has been newly generated.')

        db.session.commit()

    except Exception as e:
        flash(f'An error occurred during content regeneration: {e}')

    return redirect(url_for('chapter_view', chapter_id=chapter.id))

# --- Main Execution ---
def create_tables():
    with app.app_context():
        # Check if the migrations directory exists and has been stamped
        if os.path.exists(os.path.join(app.root_path, 'migrations')):
            from flask_migrate import upgrade
            upgrade()
        else:
            # Fallback for initial setup if migrations aren't initialized yet
            db.create_all()

if __name__ == '__main__':
    create_tables()
    app.run(debug=True)