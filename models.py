from app import db

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    todo_text = db.Column(db.String(100), index = True)