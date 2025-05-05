# Resume Analyzer App

A web-based application built with Flask that allows users to upload their resume to compare against job descriptions and get a similarly score rating.

## Features

- Upload and analyze resumes in PDF format
- Score resumes based on:
  - Cosine Similarity
  - TF-IDF Similairity
  - Hugging Face's Setence Transformer
  - Hugging Face's Zero-shot classifier 

## Tech Stack

- Python
- Flask
- HTML, CSS, JavaScript
- SQLite

## Getting Started

1. Clone the repo:
```
bash
git clone https://github.com/ajnguy/Resume-Analyzer-App.git
cd Resume-Analyzer-App
```
3. Set up virtual environment
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
5. Run Flask
```
flask run
```

## Using the App
- Upload your resume to be stored locally
- Copy and paste a job description to be compared
- Compare and see similarity ratings
