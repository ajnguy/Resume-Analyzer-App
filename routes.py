from app import app, db
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from models import Todo
from forms import TodoForm
from werkzeug.utils import secure_filename
import os
from analysis import Analyze

@app.route('/', methods=["GET", "POST"])
def home():
    resumes = os.listdir(app.config['UPLOAD_FOLDER'])  # List uploaded resumes
    cosine_similarity = 0
    semantic_similarity = 0
    boosted_semantic_similarity = 0
    missing_keywords = []
    
    resume_class = ""
    resume_fit_score = None
    
    if request.method == 'POST':
        if 'file' in request.files:  # Handle file upload
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
            elif file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File uploaded successfully!', 'success')
            return redirect(url_for('home'))
        # Handle resume selection (selecting a resume from the dropdown)
        selected_resume = request.form.get('resume')
        if selected_resume:
            selected_resume_path = os.path.join(app.config['UPLOAD_FOLDER'], selected_resume)
            # print(f"Selected resume: {selected_resume_path}")  # This is where you get the path to the selected resume
            
        # Handle job description
        job_description = request.form.get('job_description')
        if job_description:
            pass
            # print(f"Job Description: {job_description}")  # You can process or analyze the job description here

        analyze = Analyze(selected_resume_path, job_description)
        cosine_similarity = analyze.calculate_cosine_similarity()
        semantic_similarity = analyze.calculate_semantic_similarity()
        boosted_semantic_similarity = analyze.calculate_boosted_semantic_similarity()
        missing_keywords = analyze.missing_keywords
        resume_class, resume_fit_score = analyze.fit_resume()

        print(cosine_similarity)
        print(semantic_similarity)
        print(boosted_semantic_similarity)
        print(missing_keywords)
        print(resume_class, resume_fit_score)
        
        

    return render_template('home.html', resumes=resumes, 
                           cosine_similarity=cosine_similarity, 
                           semantic_similarity=semantic_similarity, 
                           boosted_semantic_similarity=boosted_semantic_similarity, 
                           missing_keywords=missing_keywords,
                           resume_class=resume_class,
                           resume_fit_score=resume_fit_score)

# Verify that file is a PDF
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Fetch resume from folder
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(app.root_path, 'uploads')
    return send_from_directory(uploads_dir, filename)

@app.route('/sample', methods=["GET", "POST"])
def sample():
    return render_template('index.html')


@app.route('/samplegeneric', methods=["GET", "POST"])
def sampleGeneric():
    return render_template('generic.html')

@app.route('/sampleElements', methods=["GET", "POST"])
def sampleElements():
    return render_template('elements.html')