import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from werkzeug.utils import secure_filename
from flask_migrate import Migrate
from dotenv import load_dotenv
from datetime import datetime

from models import db, User, Book, Chapter, GeneratedContent, Course, UserContext, CourseContext, BookContext, BookPreface, BookSummary, Semester
import ai_service
import pdf_utils

# --- App Initialization ---
app = Flask(__name__)

# --- Custom Jinja Filter ---
def from_json(json_string):
    if json_string:
        return json.loads(json_string)
    return None

app.jinja_env.filters['fromjson'] = from_json

# --- Configuration ---
load_dotenv()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_development')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///db.sqlite3')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

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

# --- Main Routes ---
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    courses = Course.query.filter_by(user_id=current_user.id).order_by(Course.created_at.desc()).all()
    standalone_books = Book.query.filter_by(user_id=current_user.id, course_id=None).order_by(Book.uploaded_at.desc()).all()
    return render_template('dashboard.html', name=current_user.email, courses=courses, standalone_books=standalone_books)

# --- Auth Routes ---
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
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
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
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

# --- Profile Routes ---
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
        profile_data = {
            'academic_level': request.form.get('academic_level'), 'interests': request.form.get('interests'),
            'learning_style': request.form.get('learning_style'), 'location': request.form.get('location'),
            'explanation_style': request.form.get('explanation_style')
        }
        if not user_context:
            user_context = UserContext(user_id=current_user.id)
            db.session.add(user_context)

        for key, value in profile_data.items():
            setattr(user_context, key, value)

        try:
            generated_text = ai_service.get_user_context_summary(profile_data)
            if generated_text:
                user_context.generated_context_text = generated_text
                flash('Successfully generated your personalized learning context.')
        except Exception as e:
            flash(f'Could not generate AI context: {e}')

        db.session.commit()
        flash('Your profile has been updated.')
        return redirect(url_for('profile'))
    return render_template('edit_profile.html', user_context=user_context)

# --- Course and Book Routes ---
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
            flash(f'Course "{name}" created. Now generating course context...')
            try:
                context_data = ai_service.get_course_context(name, provider)
                if context_data:
                    new_context = CourseContext(
                        course_id=new_course.id, subject_analysis=context_data.get('subject_analysis'),
                        prerequisites=json.dumps(context_data.get('prerequisites')),
                        real_world_apps=json.dumps(context_data.get('real_world_apps')),
                        cultural_connections=json.dumps(context_data.get('cultural_connections'))
                    )
                    db.session.add(new_context)
                    db.session.commit()
                    flash('Course context generated successfully.')
            except Exception as e:
                flash(f'Could not generate AI course context: {e}')
            return redirect(url_for('dashboard'))
    return render_template('create_course.html')

@app.route('/upload_book', methods=['POST'])
@login_required
def upload_book():
    if 'file' not in request.files:
        flash('No file part'); return redirect(request.referrer or url_for('dashboard'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file'); return redirect(request.referrer or url_for('dashboard'))
    if file and file.filename.endswith('.pdf'):
        original_filename = secure_filename(file.filename)
        filename = f"{current_user.id}_{int(datetime.now().timestamp())}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        course_id = request.form.get('course_id')
        semester_id = request.form.get('semester_id')

        # Validation
        if course_id:
            course = Course.query.filter_by(id=course_id, user_id=current_user.id).first()
            if not course:
                flash("Invalid course selected."); return redirect(url_for('dashboard'))
        else: course_id = None
        if semester_id:
            semester = Semester.query.filter_by(id=semester_id, course_id=course_id).first()
            if not semester:
                flash("Invalid semester for the chosen course."); return redirect(url_for('dashboard'))
        else: semester_id = None

        new_book = Book(user_id=current_user.id, course_id=course_id, semester_id=semester_id, filename=filename, original_name=original_filename)
        db.session.add(new_book)
        db.session.commit()
        flash(f'Book "{original_filename}" uploaded. Processing with AI...')

        try:
            ai_service.process_new_book(new_book.id)
            flash('Your new book has been fully processed!')
        except Exception as e:
            db.session.rollback()
            db.session.delete(new_book)
            db.session.commit()
            flash(f'An error occurred during AI processing: {e}')

        return redirect(url_for('course_details', course_id=course_id) if course_id else url_for('dashboard'))
    else:
        flash('Only PDF files are allowed.')
        return redirect(request.referrer or url_for('dashboard'))

@app.route('/course/<int:course_id>')
@login_required
def course_details(course_id):
    course = Course.query.filter_by(id=course_id, user_id=current_user.id).first_or_404()
    books = Book.query.filter_by(course_id=course.id).order_by(Book.semester_id, Book.original_name).all()
    return render_template('course_details.html', course=course, books=books)

@app.route('/course/<int:course_id>/add_semester', methods=['POST'])
@login_required
def add_semester(course_id):
    course = Course.query.filter_by(id=course_id, user_id=current_user.id).first_or_404()
    name = request.form.get('name')
    if name:
        db.session.add(Semester(name=name, course_id=course.id))
        db.session.commit()
        flash(f'Semester "{name}" has been added to {course.name}.')
    else:
        flash("Semester name cannot be empty.")
    return redirect(url_for('course_details', course_id=course_id))

@app.route('/course/<int:course_id>/book/<int:book_id>')
@login_required
def book_details(course_id, book_id):
    book = Book.query.filter_by(id=book_id, course_id=course_id, user_id=current_user.id).first_or_404()
    chapters = Chapter.query.filter_by(book_id=book.id).order_by(Chapter.chapter_number).all()
    return render_template('book_details.html', book=book, chapters=chapters)

@app.route('/course/<int:course_id>/book/<int:book_id>/chapter/<int:chapter_id>')
@login_required
def chapter_view(course_id, book_id, chapter_id):
    book = Book.query.filter_by(id=book_id, course_id=course_id, user_id=current_user.id).first_or_404()
    chapter = Chapter.query.filter_by(id=chapter_id, book_id=book.id).first_or_404()
    content = GeneratedContent.query.filter_by(chapter_id=chapter.id).first()
    return render_template('chapter_view.html', book=book, chapter=chapter, content=content)

# --- Generation Routes ---
@app.route('/course/<int:course_id>/book/<int:book_id>/chapter/<int:chapter_id>/generate_overview', methods=['POST'])
@login_required
def generate_overview(course_id, book_id, chapter_id):
    book = Book.query.filter_by(id=book_id, user_id=current_user.id).first_or_404()
    chapter = Chapter.query.filter_by(id=chapter_id, book_id=book.id).first_or_404()
    content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first_or_404()
    try:
        user_context = UserContext.query.filter_by(user_id=current_user.id).first()

        # Fetch all chapter titles to provide context to the AI
        all_chapters_for_book = Chapter.query.filter_by(book_id=book.id).order_by(Chapter.chapter_number).all()
        all_chapter_titles = [c.title for c in all_chapters_for_book]

        overview_html = ai_service.get_overview_for_trimmed_chapter(chapter, book, user_context, all_chapter_titles)
        content_record.html_content = overview_html or "<p>Content generation failed.</p>"
        db.session.commit()
        flash("Chapter overview generated successfully!")
    except Exception as e:
        flash(f"An error occurred during overview generation: {e}")
    return redirect(url_for('chapter_view', course_id=course_id, book_id=book_id, chapter_id=chapter_id))

@app.route('/course/<int:course_id>/book/<int:book_id>/chapter/<int:chapter_id>/generate_study_aids', methods=['POST'])
@login_required
def generate_study_aids(course_id, book_id, chapter_id):
    book = Book.query.filter_by(id=book_id, user_id=current_user.id).first_or_404()
    chapter = Chapter.query.filter_by(id=chapter_id, book_id=book.id).first_or_404()
    content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first_or_404()
    if "Content generation is pending" in content_record.html_content:
        flash("Please generate the chapter overview before creating study aids.")
        return redirect(url_for('chapter_view', course_id=course_id, book_id=book_id, chapter_id=chapter_id))
    try:
        user_context = UserContext.query.filter_by(user_id=current_user.id).first()
        flashcards_json = ai_service.get_flashcards_for_chapter(chapter, book, user_context)
        content_record.flashcards_json = json.dumps(flashcards_json) if flashcards_json else None
        mind_map_json = ai_service.get_mind_map_for_chapter(chapter, book, user_context)
        content_record.mind_map_json = json.dumps(mind_map_json) if mind_map_json else None
        db.session.commit()
        flash("Study aids generated successfully!")
    except Exception as e:
        flash(f"An error occurred during study aid generation: {e}")
    return redirect(url_for('chapter_view', course_id=course_id, book_id=book_id, chapter_id=chapter_id))

@app.route('/course/<int:course_id>/book/<int:book_id>/chapter/<int:chapter_id>/generate_assessment', methods=['POST'])
@login_required
def generate_assessment(course_id, book_id, chapter_id):
    book = Book.query.filter_by(id=book_id, user_id=current_user.id).first_or_404()
    chapter = Chapter.query.filter_by(id=chapter_id, book_id=book.id).first_or_404()
    try:
        assessment_json = ai_service.get_assessment_for_chapter(chapter, book)
        if assessment_json:
            content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first_or_404()
            content_record.questions_json = json.dumps(assessment_json)
            db.session.commit()
            flash("Assessment generated successfully!")
        else:
            flash("Failed to generate assessment from AI service.")
    except Exception as e:
        flash(f"An error occurred during assessment generation: {e}")
    return redirect(url_for('chapter_view', course_id=course_id, book_id=book_id, chapter_id=chapter_id))

@app.route('/course/<int:course_id>/book/<int:book_id>/chapter/<int:chapter_id>/deep_dive', methods=['POST'])
@login_required
def deep_dive_content(course_id, book_id, chapter_id):
    book = Book.query.filter_by(id=book_id, user_id=current_user.id).first_or_404()
    chapter = Chapter.query.filter_by(id=chapter_id, book_id=book.id).first_or_404()
    try:
        user_context = UserContext.query.filter_by(user_id=current_user.id).first()
        course_context_db = chapter.book.course.context if chapter.book.course else None
        book_context_db = chapter.book.context
        rich_content_html = ai_service.get_deep_dive_content(chapter, book, user_context, course_context_db, book_context_db)
        content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first_or_404()
        content_record.rich_html_content = rich_content_html
        db.session.commit()
        flash(f'Deep dive for "{chapter.title}" has been generated!')
    except Exception as e:
        flash(f'An error occurred during the deep dive: {e}')
    return redirect(url_for('chapter_view', course_id=course_id, book_id=book_id, chapter_id=chapter_id))

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        from flask_migrate import upgrade
        upgrade()
    app.run(debug=True)