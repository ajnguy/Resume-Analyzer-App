# Resume Analyzer App

A web-based application built with Flask that allows users to upload their resume to compare against job descriptions and get a similarly score rating.

## Features

- Upload and analyze resumes in PDF format
- Score resumes based on:
  - Cosine and TF/IDF Similarity
  - Hugging Face's Setence Transformer
  - Hugging Face's Zero-shot classifier
- Finds missing keywords

## How it works

- The core scoring mechanics is found in analysis.py
- Cosine Similarity: Calculates the cosine of the angle between TF-IDF vectors of the resume and job description to assess textual similarity.
- Sentence Transformer Similarity: Uses Hugging Face's pretrained transformer model to embed both the resume and job description into vector representations, then computes cosine similarity between them to measure semantic similarity.
- Zero-Shot Classification: Leverages Hugging Face's transformer model to predict relevance by classifying the resume according to job-specific labels without needing prior training on those labels.

## Tech Stack

- Python
- Flask
- HTML, CSS, JavaScript
- SQLite

## Getting Started

1. Clone the repo:
```bash
git clone https://github.com/ajnguy/Resume-Analyzer-App.git
cd Resume-Analyzer-App
```
2. Set up virtual environment
- Windows 
```bash
python -m venv venv
venv\Scripts\activate
```
- macOS / Linux 
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all packages  
```bash
pip install -r requirements.txt
```
3. Install language model
```bash
python -m spacy download en_core_web_sm
```
4. Run Flask
```bash
flask run
```
5. Navigate to generated link

## Using the App
- Upload your resume to be stored locally
- Copy and paste a job description to be compared
- Compare and see similarity ratings
- Check missing keywords


