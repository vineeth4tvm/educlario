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
    standalone_books = Book.query.filter_by(user_id=current_user.id, course_id=None).order_by(Book.uploaded_at.desc()).all()
    return render_template('dashboard.html', name=current_user.email, courses=courses, standalone_books=standalone_books)

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
            'academic_level': request.form.get('academic_level'),
            'interests': request.form.get('interests'),
            'learning_style': request.form.get('learning_style'),
            'location': request.form.get('location'),
            'explanation_style': request.form.get('explanation_style')
        }
        if not user_context:
            user_context = UserContext(user_id=current_user.id)
            db.session.add(user_context)

        user_context.academic_level = profile_data['academic_level']
        user_context.interests = profile_data['interests']
        user_context.learning_style = profile_data['learning_style']
        user_context.location = profile_data['location']
        user_context.explanation_style = profile_data['explanation_style']

        try:
            generated_text = ai_service.get_user_context_summary(profile_data)
            if generated_text:
                user_context.generated_context_text = generated_text
                flash('Successfully generated your personalized learning context.')
            else:
                flash('There was an issue generating the AI context.')
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
            flash(f'Course "{name}" created. Now generating course context...')
            try:
                context_data = ai_service.get_course_context(name, provider)
                if context_data:
                    new_context = CourseContext(
                        course_id=new_course.id,
                        subject_analysis=context_data.get('subject_analysis'),
                        prerequisites=json.dumps(context_data.get('prerequisites')),
                        real_world_apps=json.dumps(context_data.get('real_world_apps')),
                        cultural_connections=json.dumps(context_data.get('cultural_connections'))
                    )
                    db.session.add(new_context)
                    db.session.commit()
                    flash('Course context generated successfully.')
                else:
                    flash('Could not generate course context.')
            except Exception as e:
                flash(f'Could not generate AI course context: {e}')
            return redirect(url_for('dashboard'))
    return render_template('create_course.html')

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

        course_id = request.form.get('course_id')
        semester_id = request.form.get('semester_id')

        if course_id:
            course_id = int(course_id)
            course = Course.query.filter_by(id=course_id, user_id=current_user.id).first()
            if not course:
                flash("Invalid course selected.")
                return redirect(url_for('dashboard'))
        else:
            course_id = None
        if semester_id:
            semester_id = int(semester_id)
            semester = Semester.query.filter_by(id=semester_id, course_id=course_id).first()
            if not semester:
                flash("Invalid semester selected for the chosen course.")
                return redirect(url_for('dashboard'))
        else:
            semester_id = None

        new_book = Book(user_id=current_user.id, course_id=course_id, semester_id=semester_id, filename=filename, original_name=original_filename)
        db.session.add(new_book)
        db.session.commit()
        flash(f'Book "{original_filename}" uploaded. Processing with AI...')

        try:
            # 1. Get chapter list and save them
            chapter_list_data = ai_service.get_book_chapter_list(filepath, original_filename)
            if not chapter_list_data or 'chapters' not in chapter_list_data:
                raise Exception("AI service failed to return a valid chapter list.")

            all_chapter_titles = [ch.get('title', '') for ch in chapter_list_data['chapters']]
            for chapter_data in chapter_list_data['chapters']:
                db.session.add(Chapter(book_id=new_book.id, chapter_number=chapter_data.get('chapter_number'), title=chapter_data.get('title'), page_range=chapter_data.get('page_range')))
            db.session.commit()

            # 2. Generate overview for each chapter using trimmed PDFs
            user_context = UserContext.query.filter_by(user_id=current_user.id).first()
            user_context_text = user_context.generated_context_text if user_context else ""

            chapters = Chapter.query.filter_by(book_id=new_book.id).all()
            for chapter in chapters:
                trimmed_filepath = None
                try:
                    trimmed_filepath = pdf_utils.trim_pdf(filepath, chapter.page_range)
                    if not trimmed_filepath: continue

                    overview_html = ai_service.get_overview_for_trimmed_chapter(trimmed_filepath, chapter.title, user_context_text, all_chapter_titles)
                    db.session.add(GeneratedContent(chapter_id=chapter.id, html_content=overview_html or "<p>Content generation failed.</p>"))
                finally:
                    if trimmed_filepath: pdf_utils.cleanup_temp_file(trimmed_filepath)
            db.session.commit()
            flash('Chapter overviews generated successfully.')

            # 3. Generate Book-Level Content (using full PDF)
            full_uploaded_file = ai_service.genai.upload_file(path=filepath, display_name=original_filename)
            new_book.subject = ai_service.detect_subject_from_book(full_uploaded_file)
            db.session.add(BookPreface(book_id=new_book.id, html_content=ai_service.get_book_preface(full_uploaded_file)))
            db.session.add(BookSummary(book_id=new_book.id, html_content=ai_service.get_book_summary(full_uploaded_file)))
            context_data = ai_service.get_book_context(full_uploaded_file)
            db.session.add(BookContext(book_id=new_book.id, themes=json.dumps(context_data.get('themes')), structure=context_data.get('structure'), complexity_map=context_data.get('complexity_map')))
            db.session.commit()
            flash('Book-level content and subject detected.')

        except Exception as e:
            db.session.rollback()
            db.session.delete(new_book)
            db.session.commit()
            flash(f'An error occurred during AI processing: {e}')
        return redirect(url_for('dashboard'))
    else:
        flash('Only PDF files are allowed.')
        return redirect(url_for('dashboard'))

@app.route('/get_semesters_for_course/<int:course_id>')
@login_required
def get_semesters_for_course(course_id):
    course = Course.query.get_or_404(course_id)
    if course.user_id != current_user.id:
        return json.dumps({'error': 'Permission denied'}), 403
    semesters = [{'id': s.id, 'name': s.name} for s in course.semesters]
    return json.dumps(semesters)

@app.route('/course/<int:course_id>')
@login_required
def course_details(course_id):
    course = Course.query.get_or_404(course_id)
    if course.user_id != current_user.id:
        flash("You do not have permission to view this course.")
        return redirect(url_for('dashboard'))
    return render_template('course_details.html', course=course)

@app.route('/course/<int:course_id>/add_semester', methods=['POST'])
@login_required
def add_semester(course_id):
    course = Course.query.get_or_404(course_id)
    if course.user_id != current_user.id:
        flash("You do not have permission to modify this course.")
        return redirect(url_for('dashboard'))
    name = request.form.get('name')
    if name:
        new_semester = Semester(name=name, course_id=course.id)
        db.session.add(new_semester)
        db.session.commit()
        flash(f'Semester "{name}" has been added to {course.name}.')
    else:
        flash("Semester name cannot be empty.")
    return redirect(url_for('course_details', course_id=course_id))

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
    if chapter.book.user_id != current_user.id:
        flash("You do not have permission to view this chapter.")
        return redirect(url_for('dashboard'))
    content = GeneratedContent.query.filter_by(chapter_id=chapter.id).first()
    return render_template('chapter_view.html', chapter=chapter, content=content)

@app.route('/chapter/<int:chapter_id>/generate_assessment', methods=['POST'])
@login_required
def generate_assessment(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    if chapter.book.user_id != current_user.id:
        flash("You do not have permission to modify this content.")
        return redirect(url_for('dashboard'))
    try:
        assessment_json = ai_service.get_assessment_for_chapter(chapter, chapter.book)
        if assessment_json:
            content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first()
            if content_record:
                content_record.questions_json = json.dumps(assessment_json)
                db.session.commit()
                flash("Assessment generated successfully!")
            else:
                flash("Could not find content record to save assessment.")
        else:
            flash("Failed to generate assessment from AI service.")
    except Exception as e:
        flash(f"An error occurred during assessment generation: {e}")
    return redirect(url_for('chapter_view', chapter_id=chapter.id))

@app.route('/chapter/<int:chapter_id>/deep_dive', methods=['POST'])
@login_required
def deep_dive_content(chapter_id):
    chapter = Chapter.query.get_or_404(chapter_id)
    if chapter.book.user_id != current_user.id:
        flash("You do not have permission to modify this content.")
        return redirect(url_for('dashboard'))
    try:
        user_context = UserContext.query.filter_by(user_id=current_user.id).first()
        course_context_db = chapter.book.course.context if chapter.book.course else None
        book_context_db = chapter.book.context

        rich_content_html = ai_service.get_deep_dive_content(chapter, chapter.book, user_context, course_context_db, book_context_db)

        content_record = GeneratedContent.query.filter_by(chapter_id=chapter.id).first()
        if content_record:
            content_record.rich_html_content = rich_content_html
            db.session.commit()
            flash(f'Deep dive for "{chapter.title}" has been generated!')
        else:
            flash('Error: Could not find the content record to update.')
    except Exception as e:
        flash(f'An error occurred during the deep dive: {e}')
    return redirect(url_for('chapter_view', chapter_id=chapter.id))

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)